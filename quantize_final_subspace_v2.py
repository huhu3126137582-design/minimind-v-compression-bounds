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
from minimind_v_bound.compression.quantization import (  # noqa: E402
    QUANTIZATION_ALGORITHM_VERSION,
    quantize_equal_width_bin_means,
)
from minimind_v_bound.configuration import (  # noqa: E402
    load_v2_subspace_registry,
    resolve_subspace_candidate,
    verify_v2_protocol_freeze,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
)


QUANTIZER_VERSION = "subspace-v2-configurable-q11-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and freeze a completed v2 Subspace candidate"
    )
    parser.add_argument("--configuration-id", choices=["S-1K", "S-16K"], required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/experiment_registry_v2_subspace.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_v2_final_checkpoint(
    *,
    checkpoint: dict,
    checkpoint_path: Path,
    registry_path: Path,
    registry: dict,
    configuration_id: str,
    intrinsic_dimension: int,
) -> None:
    training = registry["training"]
    expected_total_steps = 71708
    expected_progress = {
        "epoch": training["epochs"],
        "next_batch_in_epoch": 0,
        "global_step": expected_total_steps,
        "total_steps": expected_total_steps,
    }
    if checkpoint.get("progress") != expected_progress:
        raise RuntimeError(
            f"checkpoint is not the completed registered v2 model: {checkpoint.get('progress')}"
        )
    coordinate = checkpoint.get("subspace_params")
    if (
        not isinstance(coordinate, torch.Tensor)
        or coordinate.dtype != torch.float32
        or coordinate.shape != (intrinsic_dimension,)
        or not torch.isfinite(coordinate).all()
    ):
        raise RuntimeError("v2 final checkpoint has an invalid intrinsic coordinate")
    contract = checkpoint.get("contract", {})
    implementation_path = ROOT / registry["frozen_inputs"]["implementation_contract"]
    required = {
        "experiment_registry_sha256": sha256_file(registry_path),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "projector_init_manifest_sha256": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "implementation_contract_sha256": sha256_file(implementation_path),
        "train_manifest_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["train_manifest"]
        ),
        "train_index_metadata_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["train_index"] / "metadata.json"
        ),
        "configuration_id": configuration_id,
        "intrinsic_dimension": intrinsic_dimension,
        "subspace_seed": registry["model_family"]["structured_projection"]["seed"],
        "world_size": training["planned_world_size"],
        "epochs": training["epochs"],
        "total_steps": expected_total_steps,
        "old_v1_test_accessed": False,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"v2 final checkpoint contract mismatch: {key}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)


def build_quantized_projector_hashes(
    coordinate: np.ndarray, *, intrinsic_dimension: int, subspace_seed: int
) -> tuple[str, str]:
    torch.manual_seed(20260813)
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
    projector_manifest = json.loads(
        (ROOT / "configs/projector_init_manifest.json").read_text(encoding="utf-8")
    )
    if initial_hash != projector_manifest["canonical_state_sha256"]:
        raise RuntimeError("Projector initialization differs from its frozen manifest")
    wrapper = prepare_intrinsic_projector(
        model,
        intrinsic_dimension=intrinsic_dimension,
        seed=subspace_seed,
    )
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(coordinate))
    materialized_hash = canonical_named_tensor_sha256(
        list(wrapper.materialized_parameters().items())
    )
    return initial_hash, materialized_hash


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    verify_v2_protocol_freeze(root=ROOT, registry_path=registry_path)
    registry = load_v2_subspace_registry(registry_path)
    candidate = resolve_subspace_candidate(
        registry, args.configuration_id, root=ROOT
    )
    if candidate.action != "train_new":
        raise RuntimeError(f"{candidate.configuration_id} must be imported, not quantized here")
    registered_checkpoint = candidate.output_directory / "training/final.pt"
    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint is not None else registered_checkpoint
    )
    if checkpoint_path != registered_checkpoint:
        raise ValueError("formal v2 quantization requires the registered final checkpoint")
    registered_output = candidate.output_directory / "quantized_q11"
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else registered_output
    )
    if output_directory != registered_output:
        raise ValueError("formal v2 quantization requires the registered output directory")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_v2_final_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        registry_path=registry_path,
        registry=registry,
        configuration_id=candidate.configuration_id,
        intrinsic_dimension=candidate.intrinsic_dimension,
    )
    coordinate = checkpoint["subspace_params"].detach().cpu().numpy()
    quantized = quantize_equal_width_bin_means(
        coordinate, levels=candidate.quantization_levels
    )
    structure_bits = registry["description_length"]["structure_choice_bits"]
    length = description_length_upper_bound(
        quantized.counts,
        dimension=candidate.intrinsic_dimension,
        codebook_bits_per_center=registry["description_length"][
            "codebook_bits_per_center"
        ],
        structure_choice_bits=structure_bits,
    )
    subspace_seed = registry["model_family"]["structured_projection"]["seed"]
    descriptor = {
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "configuration_id": candidate.configuration_id,
        "candidate_id_code_bits": structure_bits,
        "parameterization": registry["model_family"]["parameterization"],
        "intrinsic_dimension": candidate.intrinsic_dimension,
        "subspace_seed": subspace_seed,
        "levels": candidate.quantization_levels,
        "quantization_algorithm_version": QUANTIZATION_ALGORITHM_VERSION,
        "v2_registry_sha256": sha256_file(registry_path),
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
    coordinate_hash = canonical_reconstructed_coordinate_sha256(
        quantized.reconstructed
    )
    initial_hash, materialized_hash = build_quantized_projector_hashes(
        quantized.reconstructed,
        intrinsic_dimension=candidate.intrinsic_dimension,
        subspace_seed=subspace_seed,
    )
    hypothesis_descriptor = {
        "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "materialized_projector_parameters_sha256": materialized_hash,
    }
    hypothesis_hash = sha256_json(hypothesis_descriptor)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{candidate.configuration_id.lower()}_quantized_q11.",
            dir=output_directory.parent,
        )
    )
    try:
        file_hashes = write_quantized_arrays(temporary, quantized)
        manifest = {
            "schema_version": 1,
            "status": "v2_quantized_model_frozen_before_shared_certificate_sampling",
            "quantizer_version": QUANTIZER_VERSION,
            "source": {
                "formal_checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "formal_checkpoint_sha256": sha256_file(checkpoint_path),
                "formal_coordinate_sha256": hashlib.sha256(
                    coordinate.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
                "v2_registry_sha256": sha256_file(registry_path),
                "old_v1_test_accessed_by_training_or_quantization": False,
            },
            "quantization": {
                "algorithm_version": QUANTIZATION_ALGORITHM_VERSION,
                "dimension": candidate.intrinsic_dimension,
                "levels": candidate.quantization_levels,
                "center_dtype": "float16",
                "assignment_dtype": "uint16",
                "tie_break": "lowest_center_index",
                "recompute_assignments_after_fp16_cast": True,
                "qat_enabled": False,
                "centers": [float(value) for value in quantized.centers],
                "counts": [int(value) for value in quantized.counts],
                "assignments_changed_after_fp16_cast": int(
                    np.count_nonzero(
                        quantized.assignments != quantized.initial_assignments
                    )
                ),
                "coordinate_mse": float(
                    np.mean(
                        (coordinate.astype(np.float64) - quantized.reconstructed) ** 2
                    )
                ),
                "coordinate_max_abs_error": float(
                    np.max(
                        np.abs(
                            coordinate.astype(np.float64) - quantized.reconstructed
                        )
                    )
                ),
            },
            "description_length": {
                "version": DESCRIPTION_LENGTH_VERSION,
                **length,
            },
            "hashes": {
                "projector_initialization_sha256": initial_hash,
                "reconstructed_coordinate_sha256": coordinate_hash,
                "materialized_projector_parameters_sha256": materialized_hash,
                "compressed_model_sha256": compressed_hash,
                "quantized_hypothesis_sha256": hypothesis_hash,
            },
            "compressed_model_descriptor": descriptor,
            "files": file_hashes,
            "shared_certificate_sample_seed": None,
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
                "configuration_id": candidate.configuration_id,
                "K_VLM_upper_bits": length["K_VLM_upper_bits"],
                "compressed_model_sha256": compressed_hash,
                "quantized_hypothesis_sha256": hypothesis_hash,
                "output_directory": str(output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
