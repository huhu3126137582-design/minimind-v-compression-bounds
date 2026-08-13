from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from minimind_v_bound.certificate.sampling import (
    SAMPLING_ALGORITHM_VERSION,
    read_cluster_hashes,
    sample_uniform_with_replacement,
    sampling_commitment_sha256,
)
from minimind_v_bound.compression.artifact import (
    canonical_compressed_model_sha256,
    canonical_reconstructed_coordinate_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.compression.description_length import (
    description_length_upper_bound,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the one-time formal certificate sample after model freezing"
    )
    parser.add_argument(
        "--quantized-artifact",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/quantized_q11",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_sample_n10000",
    )
    return parser.parse_args()


def require_read_only_tree(directory: Path) -> None:
    if directory.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("quantized artifact directory is not mode-level read-only")
    for path in directory.iterdir():
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"quantized artifact is not mode-level read-only: {path}")


def main() -> None:
    args = parse_args()
    quantized_directory = args.quantized_artifact.resolve()
    output_directory = args.output_directory.resolve()
    if "locked_test" in quantized_directory.parts or "locked_test" in output_directory.parts:
        raise ValueError("certificate sampling must not access the locked-test area")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    registry_path = ROOT / "configs/experiment_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    certificate_config = registry["certificate"]
    if certificate_config != {
        "sample_size": 10000,
        "sampling": "uniform_with_replacement",
        "sampling_seed_policy": "fresh_os_csprng_after_quantized_model_and_hash_are_final",
        "delta_model": 0.025,
        "delta_subsample": 0.025,
        "model_count": 1,
        "alpha_search": {
            "family": "reduced_dyadic",
            "max_denominator_bits": 16,
            "numerator": "odd",
            "code_length": "b + 2*floor(log2(b))",
        },
    }:
        raise RuntimeError("certificate registry is not the expected preregistered contract")

    require_read_only_tree(quantized_directory)
    quantized_manifest, quantized = load_and_verify_quantized_arrays(
        quantized_directory
    )
    if quantized_manifest["status"] != "quantized_model_frozen_before_certificate_sampling":
        raise RuntimeError("quantized model was not frozen before certificate sampling")
    if quantized_manifest["certificate_sample_seed"] is not None:
        raise RuntimeError("quantized manifest already contains a certificate seed")
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=quantized_manifest["compressed_model_descriptor"],
    )
    coordinate_hash = canonical_reconstructed_coordinate_sha256(
        quantized.reconstructed
    )
    if compressed_hash != quantized_manifest["hashes"]["compressed_model_sha256"]:
        raise RuntimeError("compressed model hash mismatch")
    if coordinate_hash != quantized_manifest["hashes"]["reconstructed_coordinate_sha256"]:
        raise RuntimeError("reconstructed coordinate hash mismatch")
    length = description_length_upper_bound(
        quantized.counts,
        dimension=registry["model"]["intrinsic_dimension"],
        codebook_bits_per_center=registry["description_length"][
            "codebook_bits_per_center"
        ],
        structure_choice_bits=registry["description_length"]["structure_choice_bits"],
    )
    if length["K_VLM_upper_bits"] != quantized_manifest["description_length"][
        "K_VLM_upper_bits"
    ]:
        raise RuntimeError("description length no longer matches the frozen artifact")

    train_manifest_path = ROOT / "dataset/manifests/train_clusters.jsonl"
    final_checkpoint_path = ROOT / quantized_manifest["source"]["formal_checkpoint"]
    final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu", weights_only=False)
    train_manifest_hash = sha256_file(train_manifest_path)
    if final_checkpoint["contract"]["train_manifest_sha256"] != train_manifest_hash:
        raise RuntimeError("training manifest does not match the formal checkpoint")
    cluster_hashes = read_cluster_hashes(train_manifest_path)
    population_size = len(cluster_hashes)
    if population_size != 561359:
        raise RuntimeError(f"unexpected training cluster population: {population_size}")

    descriptor = {
        "schema_version": 1,
        "algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "sampling": "uniform_with_replacement",
        "indexing": "zero_based_train_manifest_line_index",
        "population_size": population_size,
        "sample_size": certificate_config["sample_size"],
        "train_manifest_sha256": train_manifest_hash,
        "experiment_registry_sha256": sha256_file(registry_path),
        "quantized_manifest_sha256": sha256_file(quantized_directory / "manifest.json"),
        "compressed_model_sha256": compressed_hash,
        "quantized_hypothesis_sha256": quantized_manifest["hashes"][
            "quantized_hypothesis_sha256"
        ],
        "K_VLM_upper_bits": length["K_VLM_upper_bits"],
        "model_count": certificate_config["model_count"],
        "delta_model": certificate_config["delta_model"],
        "delta_subsample": certificate_config["delta_subsample"],
    }

    # This is deliberately the first randomness request in the program and no
    # CLI seed override exists, preventing accidental seed selection.
    seed = secrets.token_bytes(32)
    indices = sample_uniform_with_replacement(
        seed=seed,
        population_size=population_size,
        sample_size=certificate_config["sample_size"],
    )
    commitment = sampling_commitment_sha256(
        descriptor=descriptor, seed=seed, indices=indices
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".certificate_sample_n10000.", dir=output_directory.parent)
    )
    try:
        indices_path = temporary / "sample_indices.npy"
        np.save(indices_path, indices, allow_pickle=False)
        clusters_path = temporary / "sampled_clusters.jsonl"
        with clusters_path.open("x", encoding="utf-8", newline="\n") as output:
            for draw_position, cluster_index in enumerate(indices):
                output.write(
                    json.dumps(
                        {
                            "draw_position": draw_position,
                            "train_manifest_index": int(cluster_index),
                            "cluster_sha256": cluster_hashes[int(cluster_index)],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        unique_count = int(np.unique(indices).size)
        manifest = {
            "schema_version": 1,
            "status": "certificate_sample_frozen_before_loss_evaluation",
            "seed": {
                "source": "python_secrets_token_bytes_backed_by_os_csprng",
                "bits": 256,
                "hex": seed.hex(),
            },
            "sampling_descriptor": descriptor,
            "sample_statistics": {
                "draw_count": len(indices),
                "unique_cluster_count": unique_count,
                "duplicate_draw_count": len(indices) - unique_count,
                "minimum_index": int(indices.min()),
                "maximum_index": int(indices.max()),
            },
            "hashes": {
                "sampling_commitment_sha256": commitment,
                "sample_indices_sha256": sha256_file(indices_path),
                "sampled_clusters_sha256": sha256_file(clusters_path),
            },
            "certificate_losses_evaluated": False,
            "alpha_selected": False,
            "locked_test_accessed": False,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.rename(temporary, output_directory)
        for artifact in output_directory.iterdir():
            artifact.chmod(0o444)
        output_directory.chmod(0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "population_size": population_size,
                "sample_size": len(indices),
                "unique_cluster_count": unique_count,
                "duplicate_draw_count": len(indices) - unique_count,
                "sampling_commitment_sha256": commitment,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

