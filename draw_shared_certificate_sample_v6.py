from __future__ import annotations

import json
import secrets
import shutil
import tempfile
from pathlib import Path

import numpy as np

from minimind_v_bound.certificate.sampling import (
    SAMPLING_ALGORITHM_VERSION,
    read_cluster_hashes,
    sample_uniform_with_replacement,
    sampling_commitment_sha256,
)
from minimind_v_bound.compression.artifact import sha256_file
from minimind_v_bound.configuration_v6_compressibility import (
    load_v6_registry,
    resolve_v6_candidate,
    verify_v6_protocol_freeze,
)

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "configs/experiment_registry_v6_compressibility.yaml"
REVISION_MANIFEST = ROOT / "configs/v6_q11_only_protocol_revision_manifest.json"
TRAIN_MANIFEST = ROOT / "dataset/manifests/train_clusters.jsonl"
OUTPUT = ROOT / "runs/v6_compressibility/shared_certification_sample_n10000"
FAMILY = ROOT / "runs/v6_compressibility/quantized_family_manifest.json"
VERIFICATION = ROOT / "runs/v6_compressibility/quantized_family_verification.json"


def main() -> None:
    if OUTPUT.exists() or FAMILY.exists() or VERIFICATION.exists():
        raise FileExistsError("v6 certification/family output already exists")
    verify_v6_protocol_freeze(root=ROOT, registry_path=REGISTRY_PATH)
    registry = load_v6_registry(REGISTRY_PATH)
    candidates = [
        resolve_v6_candidate(registry, item["configuration_id"], root=ROOT)
        for item in registry["candidate_family"]["candidates"]
    ]
    if len(candidates) != 9 or any(c.quantization_levels != 11 for c in candidates):
        raise RuntimeError("v6 candidate family is not exactly nine Q=11 models")

    hypotheses = []
    for candidate in candidates:
        artifact = candidate.output_directory / "quantized_q11"
        manifest_path = artifact / "manifest.json"
        verification_path = candidate.output_directory / "quantized_q11_verification.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "v6_qat_hypothesis_frozen_before_fresh_certification_sampling":
            raise RuntimeError(f"artifact is not frozen: {candidate.configuration_id}")
        if verification.get("status") != "v6_qat_hypothesis_independently_verified_before_sampling":
            raise RuntimeError(f"artifact is not independently verified: {candidate.configuration_id}")
        hypotheses.append({
            "configuration_id": candidate.configuration_id,
            "candidate_id_code": candidate.candidate_id,
            "parameterization": "projector_sublora" if candidate.parameterization == "sublora" else "subspace",
            "lora_rank": candidate.rank,
            "intrinsic_dimension": candidate.intrinsic_dimension,
            "artifact_directory": str(artifact.relative_to(ROOT)),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "compressed_model_descriptor": manifest["compressed_model_descriptor"],
            "description_length": manifest["description_length"],
            "hashes": manifest["hashes"],
        })

    family = {
        "schema_version": 1,
        "status": "v6_nine_q11_quantized_hypotheses_frozen_before_shared_sampling",
        "experiment_id": registry["experiment_id"],
        "candidate_order": [item["configuration_id"] for item in hypotheses],
        "model_count": 9,
        "hypotheses": hypotheses,
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "revision_manifest_sha256": sha256_file(REVISION_MANIFEST),
        "fresh_shared_sample_seed": None,
    }
    family_hash = sha256_file_from_bytes((json.dumps(family, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    verification = {
        "schema_version": 1,
        "status": "v6_nine_q11_quantized_hypotheses_independently_verified_before_sampling",
        "family_manifest_sha256": family_hash,
        "model_count": 9,
        "hypotheses": [
            {
                "configuration_id": item["configuration_id"],
                "K_VLM_upper_bits": item["description_length"]["K_VLM_upper_bits"],
                "quantized_hypothesis_sha256": item["hashes"]["quantized_hypothesis_sha256"],
            }
            for item in hypotheses
        ],
        "fresh_shared_sample_seed": None,
    }
    train_hash = sha256_file(TRAIN_MANIFEST)
    cluster_hashes = read_cluster_hashes(TRAIN_MANIFEST)
    if len(cluster_hashes) != 561359:
        raise RuntimeError("unexpected training population")
    seed = secrets.token_bytes(32)
    descriptor = {
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "drawer_source_sha256": sha256_file(Path(__file__)),
        "sampling_source_sha256": sha256_file(ROOT / "minimind_v_bound/certificate/sampling.py"),
        "sampling": "uniform_with_replacement",
        "population_size": len(cluster_hashes),
        "sample_size": 10000,
        "train_manifest_sha256": train_hash,
        "v6_registry_sha256": sha256_file(REGISTRY_PATH),
        "v6_revision_manifest_sha256": sha256_file(REVISION_MANIFEST),
        "family_manifest_sha256": family_hash,
        "family_verification_sha256": sha256_file_from_bytes((json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()),
        "models": hypotheses,
        "model_count": 9,
        "delta_model": 0.025,
        "delta_subsample_family_total": 0.025,
        "per_model_subsample_correction": "ln_9_over_delta_subsample",
        "existing_certificate_samples_reused": False,
    }
    indices = sample_uniform_with_replacement(seed=seed, population_size=len(cluster_hashes), sample_size=10000)
    commitment = sampling_commitment_sha256(descriptor=descriptor, seed=seed, indices=indices)
    temporary = Path(tempfile.mkdtemp(prefix=".v6_shared_certificate_sample.", dir=OUTPUT.parent))
    try:
        np.save(temporary / "sample_indices.npy", indices, allow_pickle=False)
        with (temporary / "sampled_clusters.jsonl").open("x", encoding="utf-8") as out:
            for position, index in enumerate(indices):
                out.write(json.dumps({"draw_position": position, "train_manifest_index": int(index), "cluster_sha256": cluster_hashes[int(index)]}, sort_keys=True, separators=(",", ":")) + "\n")
        sample_manifest = {
            "schema_version": 1,
            "status": "v6_shared_certificate_sample_frozen_before_loss_evaluation",
            "seed": {"source": "python_secrets_token_bytes_backed_by_os_csprNG", "bits": 256, "hex": seed.hex()},
            "sampling_descriptor": descriptor,
            "sample_statistics": {"draw_count": 10000, "unique_cluster_count": int(np.unique(indices).size), "duplicate_draw_count": 10000 - int(np.unique(indices).size)},
            "hashes": {"sampling_commitment_sha256": commitment, "sample_indices_sha256": sha256_file(temporary / "sample_indices.npy"), "sampled_clusters_sha256": sha256_file(temporary / "sampled_clusters.jsonl")},
            "certificate_losses_evaluated": False,
            "alpha_selected": False,
            "heldout_validation_accessed": False,
            "existing_certificate_sample_accessed": False,
        }
        (temporary / "manifest.json").write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        FAMILY.write_text(json.dumps(family, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        VERIFICATION.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(OUTPUT), "model_count": 9, "sample_size": 10000, "population_size": len(cluster_hashes), "unique_cluster_count": int(np.unique(indices).size), "commitment": commitment}, indent=2, sort_keys=True))


def sha256_file_from_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
