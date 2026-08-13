from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "upstream/minimind-v"))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minimind_v_bound.compression.artifact import (  # noqa: E402
    canonical_compressed_model_sha256,
    canonical_named_tensor_sha256,
    canonical_reconstructed_coordinate_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.compression.description_length import (  # noqa: E402
    description_length_upper_bound,
)
from minimind_v_bound.data.caption_dataset import (  # noqa: E402
    LockedTrainCaptionDataset,
    locked_caption_collate,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently reconstruct and smoke-test the frozen S-4K model"
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/quantized_q11",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs/s4k_formal_v1/final.pt"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/quantized_q11_verification.json",
    )
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_directory = args.artifact_directory.resolve()
    checkpoint_path = args.checkpoint.resolve()
    report_path = args.report.resolve()
    if "locked_test" in artifact_directory.parts or "locked_test" in report_path.parts:
        raise ValueError("verification must not access or write the locked-test area")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    manifest, quantized = load_and_verify_quantized_arrays(artifact_directory)
    if manifest["status"] != "quantized_model_frozen_before_certificate_sampling":
        raise RuntimeError("artifact is not in the expected frozen state")
    if manifest["certificate_sample_seed"] is not None:
        raise RuntimeError("certificate sampling unexpectedly occurred already")
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=manifest["compressed_model_descriptor"],
    )
    coordinate_hash = canonical_reconstructed_coordinate_sha256(
        quantized.reconstructed
    )
    if compressed_hash != manifest["hashes"]["compressed_model_sha256"]:
        raise RuntimeError("independent compressed-model hash mismatch")
    if coordinate_hash != manifest["hashes"]["reconstructed_coordinate_sha256"]:
        raise RuntimeError("independent reconstructed-coordinate hash mismatch")
    length = description_length_upper_bound(
        quantized.counts, dimension=4096, codebook_bits_per_center=16
    )
    if length["K_VLM_upper_bits"] != manifest["description_length"]["K_VLM_upper_bits"]:
        raise RuntimeError("independent description-length result mismatch")

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
    weights = torch.load(
        ROOT / "model/llm_768.pth", map_location="cpu", weights_only=True
    )
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError("pinned LLM checkpoint does not match the architecture")
    wrapper = prepare_intrinsic_projector(model, intrinsic_dimension=4096, seed=137)

    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(quantized.reconstructed))
    projector_hash = canonical_named_tensor_sha256(
        list(wrapper.materialized_parameters().items())
    )
    if projector_hash != manifest["hashes"]["materialized_projector_parameters_sha256"]:
        raise RuntimeError("independent materialized-projector hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    dataset = LockedTrainCaptionDataset(
        parquet_path=ROOT / "dataset/raw/pretrain_i2t.parquet",
        cluster_database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        train_manifest_path=ROOT / "dataset/manifests/train_clusters.jsonl",
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt="Describe this image.",
        image_token_length=64,
        max_sequence_length=450,
        index_directory=ROOT / "dataset/manifests/train_index",
        smoke_limit=1,
    )
    batch = locked_caption_collate([dataset[0]])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model.to(device).eval()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = {key: value.to(device) for key, value in batch["pixel_values"].items()}
    formal_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def evaluate_coordinate(coordinate: torch.Tensor) -> float:
        with torch.no_grad():
            wrapper.subspace_params.copy_(coordinate.to(device=device, dtype=torch.float32))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values=pixel_values,
                )
                loss = result.loss + result.aux_loss
        value = float(loss.float().cpu())
        if not math.isfinite(value):
            raise RuntimeError("model smoke-test loss is non-finite")
        return value

    prequantized_loss = evaluate_coordinate(formal_checkpoint["subspace_params"])
    quantized_loss = evaluate_coordinate(torch.from_numpy(quantized.reconstructed))
    report = {
        "schema_version": 1,
        "status": "verified_without_certificate_or_locked_test_access",
        "artifact_manifest_sha256": sha256_file(artifact_directory / "manifest.json"),
        "formal_checkpoint_sha256": sha256_file(checkpoint_path),
        "compressed_model_sha256": compressed_hash,
        "reconstructed_coordinate_sha256": coordinate_hash,
        "materialized_projector_parameters_sha256": projector_hash,
        "K_VLM_upper_bits": length["K_VLM_upper_bits"],
        "training_smoke_sample": {
            "row_index": int(batch["row_index"][0]),
            "caption_index": int(batch["caption_index"][0]),
            "valid_label_count": int(batch["valid_label_count"][0]),
            "prequantized_loss_nats_per_token": prequantized_loss,
            "quantized_loss_nats_per_token": quantized_loss,
            "quantized_minus_prequantized_loss": quantized_loss - prequantized_loss,
        },
        "certificate_sample_seed": None,
        "locked_test_accessed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    report_path.chmod(0o444)
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(json.dumps({**report, "report_sha256": report_sha256}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
