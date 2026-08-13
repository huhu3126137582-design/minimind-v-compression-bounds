from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import numpy as np

from minimind_v_bound.certificate.sampling import read_cluster_hashes
from minimind_v_bound.compression.artifact import sha256_file


ROOT = Path(__file__).resolve().parent
DOMAIN = b"minimind-v-bound-certificate-sample-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently replay and verify the formal certificate sample"
    )
    parser.add_argument(
        "--sample-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_sample_n10000",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "runs/s4k_formal_v1/certificate_sample_n10000_verification.json",
    )
    return parser.parse_args()


def replay_indices(seed: bytes, population_size: int, sample_size: int) -> np.ndarray:
    """Independent implementation of the registered sampling byte protocol."""
    limit = (1 << 64) - ((1 << 64) % population_size)
    replay = np.empty(sample_size, dtype=np.uint32)
    accepted = 0
    counter = 0
    while accepted < sample_size:
        digest = hmac.new(
            seed, DOMAIN + counter.to_bytes(8, byteorder="big"), hashlib.sha256
        ).digest()
        counter += 1
        words = [
            int.from_bytes(digest[start : start + 8], byteorder="big")
            for start in range(0, 32, 8)
        ]
        for word in words:
            if word < limit:
                replay[accepted] = word % population_size
                accepted += 1
                if accepted == sample_size:
                    break
    return replay


def independent_commitment(
    descriptor: dict[str, object], seed: bytes, indices: np.ndarray
) -> str:
    digest = hashlib.sha256()
    payloads = (
        b"minimind-v-bound-certificate-sampling-commitment-v1",
        json.dumps(
            descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        seed,
        indices.astype("<u4", copy=False).tobytes(order="C"),
    )
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sample_directory = args.sample_directory.resolve()
    report_path = args.report.resolve()
    if "locked_test" in sample_directory.parts or "locked_test" in report_path.parts:
        raise ValueError("certificate verification must not access the locked-test area")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    manifest_path = sample_directory / "manifest.json"
    indices_path = sample_directory / "sample_indices.npy"
    clusters_path = sample_directory / "sampled_clusters.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "certificate_sample_frozen_before_loss_evaluation":
        raise RuntimeError("certificate sample has an unexpected status")
    if manifest["certificate_losses_evaluated"] or manifest["alpha_selected"]:
        raise RuntimeError("sample manifest claims later experimental stages already ran")
    if manifest["locked_test_accessed"]:
        raise RuntimeError("sample manifest claims locked-test access")
    if sha256_file(indices_path) != manifest["hashes"]["sample_indices_sha256"]:
        raise RuntimeError("sample index file hash mismatch")
    if sha256_file(clusters_path) != manifest["hashes"]["sampled_clusters_sha256"]:
        raise RuntimeError("sample cluster file hash mismatch")

    descriptor = manifest["sampling_descriptor"]
    if descriptor["algorithm_version"] != "hmac-sha256-counter-u64be-rejection-v1":
        raise RuntimeError("unknown sampling algorithm")
    train_manifest_path = ROOT / "dataset/manifests/train_clusters.jsonl"
    quantized_manifest_path = ROOT / "runs/s4k_formal_v1/quantized_q11/manifest.json"
    if sha256_file(train_manifest_path) != descriptor["train_manifest_sha256"]:
        raise RuntimeError("training manifest hash mismatch")
    if sha256_file(quantized_manifest_path) != descriptor["quantized_manifest_sha256"]:
        raise RuntimeError("quantized model manifest hash mismatch")

    indices = np.load(indices_path, allow_pickle=False)
    population_size = int(descriptor["population_size"])
    sample_size = int(descriptor["sample_size"])
    if indices.dtype != np.uint32 or indices.shape != (sample_size,):
        raise RuntimeError("sample index array has the wrong dtype or shape")
    if np.any(indices >= population_size):
        raise RuntimeError("sample index lies outside the training population")
    seed = bytes.fromhex(manifest["seed"]["hex"])
    if len(seed) != 32 or manifest["seed"]["bits"] != 256:
        raise RuntimeError("certificate seed is not 256 bits")
    replay = replay_indices(seed, population_size, sample_size)
    if not np.array_equal(replay, indices):
        raise RuntimeError("independent sampling replay does not match stored indices")
    commitment = independent_commitment(descriptor, seed, indices)
    if commitment != manifest["hashes"]["sampling_commitment_sha256"]:
        raise RuntimeError("independent sampling commitment mismatch")

    cluster_hashes = read_cluster_hashes(train_manifest_path)
    if len(cluster_hashes) != population_size:
        raise RuntimeError("training population size mismatch")
    record_count = 0
    with clusters_path.open("r", encoding="utf-8") as source:
        for draw_position, line in enumerate(source):
            if draw_position >= sample_size:
                raise RuntimeError("sample cluster file contains too many records")
            record = json.loads(line)
            index = int(indices[draw_position])
            expected = {
                "draw_position": draw_position,
                "train_manifest_index": index,
                "cluster_sha256": cluster_hashes[index],
            }
            if record != expected:
                raise RuntimeError(f"sample cluster mismatch at draw {draw_position}")
            record_count += 1
    if record_count != sample_size:
        raise RuntimeError("sample cluster file contains too few records")

    unique_count = int(np.unique(indices).size)
    expected_statistics = {
        "draw_count": sample_size,
        "unique_cluster_count": unique_count,
        "duplicate_draw_count": sample_size - unique_count,
        "minimum_index": int(indices.min()),
        "maximum_index": int(indices.max()),
    }
    if manifest["sample_statistics"] != expected_statistics:
        raise RuntimeError("sample statistics mismatch")
    report = {
        "schema_version": 1,
        "status": "independent_replay_verified",
        "sample_manifest_sha256": sha256_file(manifest_path),
        "sampling_commitment_sha256": commitment,
        "population_size": population_size,
        "sample_size": sample_size,
        "unique_cluster_count": unique_count,
        "duplicate_draw_count": sample_size - unique_count,
        "all_indices_in_range": True,
        "sampled_cluster_mapping_verified": True,
        "locked_test_accessed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    report_path.chmod(0o444)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
