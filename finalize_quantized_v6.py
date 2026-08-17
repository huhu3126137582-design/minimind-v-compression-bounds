from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream/minimind-v"))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402

from minimind_v_bound.compression.artifact import (  # noqa: E402
    canonical_compressed_model_sha256,
    canonical_json_bytes,
    canonical_named_tensor_sha256,
    canonical_reconstructed_coordinate_sha256,
    projector_initialization_sha256,
    sha256_file,
    write_quantized_arrays,
)
from minimind_v_bound.compression.description_length import (  # noqa: E402
    DESCRIPTION_LENGTH_VERSION,
    description_length_upper_bound,
)
from minimind_v_bound.compression.qat_quantization_v6 import (  # noqa: E402
    QAT_QUANTIZATION_VERSION,
    finalize_qat_quantization,
)
from minimind_v_bound.configuration_v6_compressibility import (  # noqa: E402
    V6_CONFIGURATION_IDS,
    load_v6_registry,
    resolve_v6_candidate,
    verify_v6_protocol_freeze,
)
from minimind_v_bound.models.qat_projector_v6 import (  # noqa: E402
    QAT_PROJECTOR_VERSION,
    prepare_qat_projector_v6,
)
from minimind_v_bound.models.structured_projection_v6 import (  # noqa: E402
    PROJECTION_V6_VERSION,
)


FINALIZER_VERSION = "v6-qat-fp16-finalizer-v1"
REGISTRY_DEFAULT = ROOT / "configs/experiment_registry_v6_compressibility.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize one completed v6 QAT hypothesis")
    parser.add_argument("--configuration-id", choices=V6_CONFIGURATION_IDS, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_DEFAULT)
    return parser.parse_args()


def validate_completed_checkpoint(
    *, checkpoint: dict, candidate, registry: dict, registry_path: Path
) -> None:
    training = registry["training"]
    expected_progress = {
        "epoch": training["epochs"],
        "next_batch_in_epoch": 0,
        "global_step": training["expected_total_steps"],
        "total_steps": training["expected_total_steps"],
    }
    if checkpoint.get("progress") != expected_progress:
        raise RuntimeError("v6 checkpoint is not the registered completed final epoch")
    coordinate = checkpoint.get("subspace_params")
    centers = checkpoint.get("quantization_centers")
    if (
        not isinstance(coordinate, torch.Tensor)
        or coordinate.dtype != torch.float32
        or coordinate.shape != (candidate.intrinsic_dimension,)
        or not torch.isfinite(coordinate).all()
    ):
        raise RuntimeError("completed v6 coordinate is invalid")
    if (
        not isinstance(centers, torch.Tensor)
        or centers.dtype != torch.float32
        or centers.shape != (candidate.quantization_levels,)
        or not torch.isfinite(centers).all()
    ):
        raise RuntimeError("completed v6 learned centers are invalid")
    if checkpoint.get("qat_initialized") is not True:
        raise RuntimeError("completed v6 checkpoint did not enter QAT")
    contract = checkpoint.get("contract", {})
    required = {
        "registry_sha256": registry["frozen_inputs"][
            "training_contract_registry_sha256"
        ],
        "configuration_id": candidate.configuration_id,
        "candidate_id": candidate.candidate_id,
        "parameterization": candidate.parameterization,
        "lora_rank": candidate.rank,
        "intrinsic_dimension": candidate.intrinsic_dimension,
        "quantization_levels": candidate.quantization_levels,
        "total_steps": training["expected_total_steps"],
        "qat_start_step": registry["qat"]["expected_qat_start_step"],
        "development_qat_start_override": False,
        "heldout_validation_accessed": False,
        "existing_certification_sample_accessed": False,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"completed v6 checkpoint contract mismatch: {key}")


def build_materialized_hashes(
    coordinate: np.ndarray, *, candidate, registry: dict
) -> tuple[str, str | None, str]:
    training = registry["training"]
    torch.manual_seed(training["projector_constructor_seed"])
    config = VLMConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=450,
        use_moe=False,
        dropout=0.0,
    )
    model = MiniMindVLM(
        config, vision_model_path=str(ROOT / "model/siglip2-base-p32-256-ve")
    )
    initial_hash = projector_initialization_sha256(
        list(model.vision_proj.named_parameters())
    )
    common = registry["candidate_family"]["common_side_information"]
    wrapper = prepare_qat_projector_v6(
        model,
        parameterization=candidate.parameterization,
        intrinsic_dimension=candidate.intrinsic_dimension,
        quantization_levels=candidate.quantization_levels,
        rank=candidate.rank,
        lora_scale=common["lora_scale"],
        subspace_seed=common["subspace_seed"],
        factor_init_seed=common["factor_init_seed"],
    )
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(coordinate))
    if candidate.parameterization == "subspace":
        named = list(wrapper.materialized_parameters().items())
        factor_hash = None
    else:
        named = list(wrapper.materialized_projector_parameters().items())
        factor_hash = wrapper.factor_initialization_sha256()
    return initial_hash, factor_hash, canonical_named_tensor_sha256(named)


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    verify_v6_protocol_freeze(root=ROOT, registry_path=registry_path)
    registry = load_v6_registry(registry_path)
    candidate = resolve_v6_candidate(registry, args.configuration_id, root=ROOT)
    checkpoint_path = candidate.output_directory / "training/final.pt"
    output_directory = candidate.output_directory / f"quantized_q{candidate.quantization_levels}"
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_completed_checkpoint(
        checkpoint=checkpoint,
        candidate=candidate,
        registry=registry,
        registry_path=registry_path,
    )
    coordinate = checkpoint["subspace_params"].detach().cpu()
    learned_centers = checkpoint["quantization_centers"].detach().cpu()
    quantized = finalize_qat_quantization(coordinate, learned_centers)
    description = registry["description_length"]
    length = description_length_upper_bound(
        quantized.counts,
        dimension=candidate.intrinsic_dimension,
        codebook_bits_per_center=description["codebook_bits_per_center"],
        structure_choice_bits=description["structure_choice_bits"],
    )
    descriptor = {
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "candidate_id": candidate.candidate_id,
        "candidate_id_bits": 4,
        "configuration_id": candidate.configuration_id,
        "parameterization": candidate.parameterization,
        "lora_rank": candidate.rank,
        "intrinsic_dimension": candidate.intrinsic_dimension,
        "quantization_levels": candidate.quantization_levels,
        "qat_quantization_version": QAT_QUANTIZATION_VERSION,
        "qat_projector_version": QAT_PROJECTOR_VERSION,
        "projection_version": PROJECTION_V6_VERSION,
        "v6_registry_sha256": sha256_file(registry_path),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "projector_init_manifest_sha256": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "implementation_contract_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["implementation_contract"]
        ),
    }
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=descriptor,
    )
    coordinate_hash = canonical_reconstructed_coordinate_sha256(quantized.reconstructed)
    initial_hash, factor_hash, materialized_hash = build_materialized_hashes(
        quantized.reconstructed, candidate=candidate, registry=registry
    )
    hypothesis_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
                "compressed_model_sha256": compressed_hash,
                "materialized_projector_parameters_sha256": materialized_hash,
            }
        )
    ).hexdigest()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{candidate.configuration_id.lower()}.finalize.", dir=output_directory.parent)
    )
    try:
        file_hashes = write_quantized_arrays(temporary, quantized)
        values = coordinate.numpy().astype(np.float64)
        manifest = {
            "schema_version": 1,
            "status": "v6_qat_hypothesis_frozen_before_fresh_certification_sampling",
            "finalizer_version": FINALIZER_VERSION,
            "source": {
                "formal_checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "formal_checkpoint_sha256": sha256_file(checkpoint_path),
                "v6_registry_sha256": sha256_file(registry_path),
                "heldout_validation_accessed": False,
                "existing_certification_sample_accessed": False,
            },
            "quantization": {
                "algorithm_version": QAT_QUANTIZATION_VERSION,
                "dimension": candidate.intrinsic_dimension,
                "levels": candidate.quantization_levels,
                "center_dtype": "float16",
                "assignment_dtype": "uint16",
                "tie_break": "lowest_center_index",
                "recompute_assignments_after_fp16_cast": True,
                "qat_enabled": True,
                "centers": [float(value) for value in quantized.centers],
                "counts": [int(value) for value in quantized.counts],
                "coordinate_mse": float(
                    np.mean((values - quantized.reconstructed.astype(np.float64)) ** 2)
                ),
                "coordinate_max_abs_error": float(
                    np.max(np.abs(values - quantized.reconstructed.astype(np.float64)))
                ),
            },
            "description_length": {"version": DESCRIPTION_LENGTH_VERSION, **length},
            "hashes": {
                "projector_initialization_sha256": initial_hash,
                "factor_initialization_sha256": factor_hash,
                "reconstructed_coordinate_sha256": coordinate_hash,
                "materialized_projector_parameters_sha256": materialized_hash,
                "compressed_model_sha256": compressed_hash,
                "quantized_hypothesis_sha256": hypothesis_hash,
            },
            "compressed_model_descriptor": descriptor,
            "files": file_hashes,
            "fresh_certification_sample_seed": None,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
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
                "configuration_id": candidate.configuration_id,
                "K_VLM_upper_bits": length["K_VLM_upper_bits"],
                "quantized_hypothesis_sha256": hypothesis_hash,
                "output_directory": str(output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
