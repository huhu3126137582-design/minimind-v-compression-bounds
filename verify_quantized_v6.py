from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from finalize_quantized_v6 import build_materialized_hashes, validate_completed_checkpoint
from minimind_v_bound.compression.artifact import (
    canonical_compressed_model_sha256,
    canonical_json_bytes,
    canonical_reconstructed_coordinate_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.compression.description_length import description_length_upper_bound
from minimind_v_bound.compression.qat_quantization_v6 import finalize_qat_quantization
from minimind_v_bound.configuration_v6_compressibility import (
    V6_CONFIGURATION_IDS,
    load_v6_registry,
    resolve_v6_candidate,
    verify_v6_protocol_freeze,
)


ROOT = Path(__file__).resolve().parent
REGISTRY_DEFAULT = ROOT / "configs/experiment_registry_v6_compressibility.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify one v6 QAT hypothesis")
    parser.add_argument("--configuration-id", choices=V6_CONFIGURATION_IDS, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    verify_v6_protocol_freeze(root=ROOT, registry_path=registry_path)
    registry = load_v6_registry(registry_path)
    candidate = resolve_v6_candidate(registry, args.configuration_id, root=ROOT)
    checkpoint_path = candidate.output_directory / "training/final.pt"
    artifact_directory = candidate.output_directory / f"quantized_q{candidate.quantization_levels}"
    report_path = candidate.output_directory / f"quantized_q{candidate.quantization_levels}_verification.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    manifest, stored = load_and_verify_quantized_arrays(artifact_directory)
    if manifest.get("status") != "v6_qat_hypothesis_frozen_before_fresh_certification_sampling":
        raise RuntimeError("v6 quantized hypothesis has the wrong frozen status")
    if manifest.get("fresh_certification_sample_seed") is not None:
        raise RuntimeError("v6 fresh certification sampling occurred too early")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_completed_checkpoint(
        checkpoint=checkpoint,
        candidate=candidate,
        registry=registry,
        registry_path=registry_path,
    )
    independent = finalize_qat_quantization(
        checkpoint["subspace_params"], checkpoint["quantization_centers"]
    )
    for name in ("centers", "assignments", "counts", "reconstructed"):
        if not np.array_equal(getattr(stored, name), getattr(independent, name)):
            raise RuntimeError(f"independent v6 quantization mismatch: {name}")
    description = registry["description_length"]
    length = description_length_upper_bound(
        stored.counts,
        dimension=candidate.intrinsic_dimension,
        codebook_bits_per_center=description["codebook_bits_per_center"],
        structure_choice_bits=description["structure_choice_bits"],
    )
    if length != {key: manifest["description_length"][key] for key in length}:
        raise RuntimeError("independent v6 description length differs")
    descriptor = manifest["compressed_model_descriptor"]
    compressed_hash = canonical_compressed_model_sha256(
        centers=stored.centers, assignments=stored.assignments, descriptor=descriptor
    )
    coordinate_hash = canonical_reconstructed_coordinate_sha256(stored.reconstructed)
    initial_hash, factor_hash, materialized_hash = build_materialized_hashes(
        stored.reconstructed, candidate=candidate, registry=registry
    )
    expected_hashes = {
        "projector_initialization_sha256": initial_hash,
        "factor_initialization_sha256": factor_hash,
        "reconstructed_coordinate_sha256": coordinate_hash,
        "materialized_projector_parameters_sha256": materialized_hash,
        "compressed_model_sha256": compressed_hash,
    }
    for key, expected in expected_hashes.items():
        if manifest["hashes"][key] != expected:
            raise RuntimeError(f"independent v6 hash differs: {key}")
    hypothesis_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
                "compressed_model_sha256": compressed_hash,
                "materialized_projector_parameters_sha256": materialized_hash,
            }
        )
    ).hexdigest()
    if manifest["hashes"]["quantized_hypothesis_sha256"] != hypothesis_hash:
        raise RuntimeError("independent v6 hypothesis hash differs")
    report = {
        "schema_version": 1,
        "status": "v6_qat_hypothesis_independently_verified_before_sampling",
        "configuration_id": candidate.configuration_id,
        "candidate_id": candidate.candidate_id,
        "artifact_manifest_sha256": sha256_file(artifact_directory / "manifest.json"),
        "formal_checkpoint_sha256": sha256_file(checkpoint_path),
        "K_VLM_upper_bits": length["K_VLM_upper_bits"],
        "compressed_model_sha256": compressed_hash,
        "quantized_hypothesis_sha256": hypothesis_hash,
        "fresh_certification_sample_seed": None,
        "heldout_validation_accessed": False,
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    report_path.chmod(0o444)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
