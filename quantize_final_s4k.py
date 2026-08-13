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
import yaml


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
from minimind_v_bound.models.intrinsic_projector import prepare_intrinsic_projector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and freeze the formal S-4K model")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs/s4k_formal_v1/final.pt"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/quantized_q11",
    )
    return parser.parse_args()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_final_checkpoint(checkpoint: dict, registry: dict, checkpoint_path: Path) -> None:
    progress = checkpoint.get("progress", {})
    if progress != {
        "epoch": 2,
        "next_batch_in_epoch": 0,
        "global_step": 71708,
        "total_steps": 71708,
    }:
        raise RuntimeError(f"checkpoint is not the completed preregistered model: {progress}")
    coordinate = checkpoint.get("subspace_params")
    expected_dimension = registry["model"]["intrinsic_dimension"]
    if (
        not isinstance(coordinate, torch.Tensor)
        or coordinate.dtype != torch.float32
        or coordinate.shape != (expected_dimension,)
        or not torch.isfinite(coordinate).all()
    ):
        raise RuntimeError("final checkpoint has an invalid intrinsic coordinate")
    contract = checkpoint.get("contract", {})
    required_hashes = {
        "experiment_registry_sha256": sha256_file(ROOT / "configs/experiment_registry.yaml"),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "implementation_contract_sha256": sha256_file(
            ROOT / "configs/implementation_contract.json"
        ),
        "train_manifest_sha256": sha256_file(
            ROOT / "dataset/manifests/train_clusters.jsonl"
        ),
        "train_index_metadata_sha256": sha256_file(
            ROOT / "dataset/manifests/train_index/metadata.json"
        ),
    }
    for key, expected in required_hashes.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"final checkpoint contract hash mismatch: {key}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)


def build_quantized_projector_hash(coordinate: np.ndarray) -> tuple[str, str]:
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
    initial_named = list(model.vision_proj.named_parameters())
    initial_hash = projector_initialization_sha256(initial_named)
    projector_manifest = json.loads(
        (ROOT / "configs/projector_init_manifest.json").read_text(encoding="utf-8")
    )
    if initial_hash != projector_manifest["canonical_state_sha256"]:
        raise RuntimeError("Projector initialization does not match the frozen manifest")
    wrapper = prepare_intrinsic_projector(model, intrinsic_dimension=4096, seed=137)
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(coordinate))
    materialized = list(wrapper.materialized_parameters().items())
    return initial_hash, canonical_named_tensor_sha256(materialized)


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_directory = args.output_directory.resolve()
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    registry_path = ROOT / "configs/experiment_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_final_checkpoint(checkpoint, registry, args.checkpoint)
    coordinate = checkpoint["subspace_params"].detach().cpu().numpy()
    levels = registry["quantization"]["levels"]
    quantized = quantize_equal_width_bin_means(coordinate, levels=levels)
    length = description_length_upper_bound(
        quantized.counts,
        dimension=len(coordinate),
        codebook_bits_per_center=registry["description_length"][
            "codebook_bits_per_center"
        ],
        structure_choice_bits=registry["description_length"]["structure_choice_bits"],
    )
    descriptor = {
        "schema_version": 1,
        "configuration_id": registry["model"]["configuration_id"],
        "parameterization": registry["model"]["parameterization"],
        "intrinsic_dimension": registry["model"]["intrinsic_dimension"],
        "subspace_seed": registry["model"]["structured_projection"]["seed"],
        "levels": levels,
        "quantization_algorithm_version": QUANTIZATION_ALGORITHM_VERSION,
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "projector_init_manifest_sha256": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "implementation_contract_sha256": sha256_file(
            ROOT / "configs/implementation_contract.json"
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
    initial_projector_hash, materialized_projector_hash = build_quantized_projector_hash(
        quantized.reconstructed
    )
    hypothesis_descriptor = {
        "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "materialized_projector_parameters_sha256": materialized_projector_hash,
    }
    hypothesis_hash = sha256_json(hypothesis_descriptor)

    parent = args.output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".quantized_q11.", dir=parent))
    try:
        file_hashes = write_quantized_arrays(temporary, quantized)
        manifest = {
            "schema_version": 1,
            "status": "quantized_model_frozen_before_certificate_sampling",
            "source": {
                "formal_checkpoint": str(args.checkpoint.relative_to(ROOT)),
                "formal_checkpoint_sha256": sha256_file(args.checkpoint),
                "formal_coordinate_sha256": hashlib.sha256(
                    coordinate.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
                "experiment_registry_sha256": sha256_file(registry_path),
                "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
                "implementation_contract_sha256": descriptor[
                    "implementation_contract_sha256"
                ],
            },
            "quantization": {
                "algorithm_version": QUANTIZATION_ALGORITHM_VERSION,
                "dimension": len(coordinate),
                "levels": levels,
                "center_dtype": "float16",
                "assignment_dtype": "uint16",
                "tie_break": "lowest_center_index",
                "recompute_assignments_after_fp16_cast": True,
                "qat_enabled": False,
                "centers": [float(value) for value in quantized.centers],
                "counts": [int(value) for value in quantized.counts],
                "assignments_changed_after_fp16_cast": int(
                    np.count_nonzero(quantized.assignments != quantized.initial_assignments)
                ),
                "coordinate_mse": float(
                    np.mean((coordinate.astype(np.float64) - quantized.reconstructed) ** 2)
                ),
                "coordinate_max_abs_error": float(
                    np.max(np.abs(coordinate.astype(np.float64) - quantized.reconstructed))
                ),
            },
            "description_length": {
                "version": DESCRIPTION_LENGTH_VERSION,
                **length,
            },
            "hashes": {
                "projector_initialization_sha256": initial_projector_hash,
                "reconstructed_coordinate_sha256": coordinate_hash,
                "materialized_projector_parameters_sha256": materialized_projector_hash,
                "compressed_model_sha256": compressed_hash,
                "quantized_hypothesis_sha256": hypothesis_hash,
            },
            "compressed_model_descriptor": descriptor,
            "files": file_hashes,
            "certificate_sample_seed": None,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.rename(temporary, args.output_directory)
        for artifact in args.output_directory.iterdir():
            artifact.chmod(0o444)
        args.output_directory.chmod(0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "K_VLM_upper_bits": length["K_VLM_upper_bits"],
        "compressed_model_sha256": compressed_hash,
        "counts": length["counts"],
        "output_directory": str(args.output_directory),
        "quantized_hypothesis_sha256": hypothesis_hash,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
