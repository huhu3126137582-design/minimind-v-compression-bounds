from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "upstream/minimind-v"))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from evaluate_certificate_s4k import (  # noqa: E402
    atomic_json,
    load_chunk,
    records_sha256,
    sha256_named_files,
    write_chunk,
)
from minimind_v_bound.certificate.dataset import (  # noqa: E402
    CERTIFICATE_DATASET_VERSION,
    CertificateCaptionDataset,
    build_certificate_caption_records,
    certificate_caption_collate,
)
from minimind_v_bound.certificate.risk import (  # noqa: E402
    ALPHA_GRID_VERSION,
    RISK_IMPLEMENTATION_VERSION,
    hierarchical_token_weights,
    smoothed_token_losses_bits,
)
from minimind_v_bound.certificate.sampling import read_cluster_hashes  # noqa: E402
from minimind_v_bound.certificate.search import (  # noqa: E402
    ALPHA_SEARCH_ALGORITHM_VERSION,
    search_reduced_dyadic_alpha,
)
from minimind_v_bound.compression.artifact import (  # noqa: E402
    canonical_compressed_model_sha256,
    canonical_named_tensor_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.configuration_v6_compressibility import (  # noqa: E402
    load_v6_registry,
    verify_v6_protocol_freeze,
)
from minimind_v_bound.models.qat_projector_v6 import (  # noqa: E402
    prepare_qat_projector_v6,
)


EVALUATOR_VERSION = "v6-nine-q11-shared-M9-bf16-v1"
CANDIDATE_ORDER = (
    "S-D256-Q11", "S-D512-Q11", "S-D1024-Q11",
    "SL-R1-D256-Q11", "SL-R1-D512-Q11", "SL-R1-D1024-Q11",
    "SL-R4-D256-Q11", "SL-R4-D512-Q11", "SL-R4-D1024-Q11",
)
OUTPUT_SLUGS = {
    item: item.lower().replace("-", "_") for item in CANDIDATE_ORDER
}
FAMILY_MANIFEST = ROOT / "runs/v6_compressibility/quantized_family_manifest.json"
FAMILY_VERIFICATION = ROOT / "runs/v6_compressibility/quantized_family_verification.json"
SAMPLE_DIRECTORY = ROOT / "runs/v6_compressibility/shared_certification_sample_n10000"
SAMPLE_VERIFICATION = FAMILY_VERIFICATION
EVALUATION_FREEZE = ROOT / "configs/v6_q11_only_protocol_revision_manifest.json"
RESULT_ROOT = ROOT / "runs/v6_compressibility/certificate_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen v6 Q=11 model on the new shared M=9 sample"
    )
    parser.add_argument("--configuration-id", choices=CANDIDATE_ORDER, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def output_directory_for(configuration_id: str) -> Path:
    return RESULT_ROOT / OUTPUT_SLUGS[configuration_id]


def load_family_and_sample() -> tuple[dict, dict, dict[str, dict]]:
    family = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
    if family.get("status") != "v6_nine_q11_quantized_hypotheses_frozen_before_shared_sampling":
        raise RuntimeError("the v6 Q=11 family is not frozen")
    if tuple(family.get("candidate_order", ())) != CANDIDATE_ORDER:
        raise RuntimeError("the frozen candidate order changed")
    hypotheses = {item["configuration_id"]: item for item in family["hypotheses"]}
    if tuple(hypotheses) != CANDIDATE_ORDER or family.get("model_count") != 9:
        raise RuntimeError("the frozen hypothesis family is not exactly M=9")

    family_verification = json.loads(FAMILY_VERIFICATION.read_text(encoding="utf-8"))
    if family_verification.get("status") != (
        "v6_nine_q11_quantized_hypotheses_independently_verified_before_sampling"
    ):
        raise RuntimeError("the hypothesis family lacks independent verification")

    sample = json.loads((SAMPLE_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    if sample.get("status") != "v6_shared_certificate_sample_frozen_before_loss_evaluation":
        raise RuntimeError("the v3 shared sample is not frozen before evaluation")
    if sample.get("certificate_losses_evaluated") or sample.get("alpha_selected"):
        raise RuntimeError("the v3 sample timing flags are invalid")
    if sample.get("heldout_validation_accessed") or sample.get("existing_certificate_sample_accessed"):
        raise RuntimeError("the v3 sample used prohibited prior data")
    descriptor = sample["sampling_descriptor"]
    if descriptor.get("model_count") != 9 or descriptor.get("sample_size") != 10000:
        raise RuntimeError("the v3 sample does not use the frozen M=9 contract")
    if descriptor.get("existing_certificate_samples_reused") is not False:
        raise RuntimeError("the v3 sample reused an existing certificate sample")
    if descriptor.get("family_manifest_sha256") != sha256_file(FAMILY_MANIFEST):
        raise RuntimeError("the sample is not bound to the current hypothesis family")
    sample_models = {}
    for item in descriptor["models"]:
        binding = dict(item)
        binding.update(item.get("hashes", {}))
        binding.update(item.get("description_length", {}))
        sample_models[item["configuration_id"]] = binding
    if tuple(sample_models) != CANDIDATE_ORDER:
        raise RuntimeError("the sample model order changed")
    for configuration_id in CANDIDATE_ORDER:
        hypothesis = hypotheses[configuration_id]
        binding = sample_models[configuration_id]
        expected = {
            "artifact_directory": hypothesis["artifact_directory"],
            "artifact_manifest_sha256": hypothesis["artifact_manifest_sha256"],
            "candidate_id_code": hypothesis["candidate_id_code"],
            "parameterization": hypothesis["parameterization"],
            "intrinsic_dimension": hypothesis["intrinsic_dimension"],
            "lora_rank": hypothesis["lora_rank"],
            "compressed_model_sha256": hypothesis["hashes"]["compressed_model_sha256"],
            "quantized_hypothesis_sha256": hypothesis["hashes"]["quantized_hypothesis_sha256"],
            "materialized_projector_parameters_sha256": hypothesis["hashes"]["materialized_projector_parameters_sha256"],
            "K_VLM_upper_bits": hypothesis["description_length"]["K_VLM_upper_bits"],
        }
        for key, value in expected.items():
            if key in {"compressed_model_sha256", "quantized_hypothesis_sha256", "materialized_projector_parameters_sha256"}:
                actual = binding.get("hashes", {}).get(key)
            elif key == "K_VLM_upper_bits":
                actual = binding.get("description_length", {}).get(key)
            else:
                actual = binding.get(key)
            if actual != value:
                raise RuntimeError(f"sample/family binding mismatch for {configuration_id}: {key}")
    return family, sample, sample_models


def verify_evaluation_freeze() -> dict:
    freeze = json.loads(EVALUATION_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "q11_only_scope_revision_frozen_before_resume":
        raise RuntimeError("v3 certificate evaluation protocol is not frozen")
    if freeze.get("model_count") != 9:
        raise RuntimeError("evaluation freeze does not use M=9")
    for relative, expected in freeze.get("artifact_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen evaluation artifact changed: {relative}")
    return freeze


def construct_quantized_model(
    *, registry: dict, hypothesis: dict, quantized: object, vision_path: Path, llm_path: Path
) -> tuple[torch.nn.Module, str]:
    torch.manual_seed(registry["training"]["projector_constructor_seed"])
    config = VLMConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=registry["evaluation"]["max_sequence_length"],
        use_moe=False,
        dropout=0.0,
    )
    model = MiniMindVLM(config, vision_model_path=str(vision_path))
    weights = torch.load(llm_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError("pinned LLM checkpoint does not match the architecture")

    dimension = int(hypothesis["intrinsic_dimension"])
    common = registry["candidate_family"]["common_side_information"]
    if hypothesis["parameterization"] in {"subspace", "projector_sublora"}:
        common = registry["candidate_family"]["common_side_information"]
        wrapper = prepare_qat_projector_v6(
            model,
            parameterization="sublora" if hypothesis["parameterization"] == "projector_sublora" else "subspace",
            intrinsic_dimension=dimension,
            quantization_levels=11,
            rank=int(hypothesis["lora_rank"]) if hypothesis["lora_rank"] is not None else None,
            lora_scale=float(common["lora_scale"]),
            subspace_seed=int(common["subspace_seed"]),
            factor_init_seed=int(common["factor_init_seed"]),
        )
    else:
        raise RuntimeError("unknown v6 parameterization")
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(quantized.reconstructed))
        wrapper.quantization_centers.copy_(torch.from_numpy(quantized.centers).float())
        wrapper.qat_initialized.fill_(True)
    wrapper.set_qat_forward(True)
    if hypothesis["parameterization"] == "subspace":
        materialized = wrapper.materialized_parameters()
    else:
        materialized = wrapper.materialized_projector_parameters()
    materialized_hash = canonical_named_tensor_sha256(list(materialized.items()))
    if materialized_hash != hypothesis["hashes"]["materialized_projector_parameters_sha256"]:
        raise RuntimeError("materialized quantized Projector hash mismatch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, materialized_hash


def main() -> None:
    args = parse_args()
    if args.batch_size != 32 or args.num_workers != 4:
        raise ValueError("formal v3 evaluation fixes batch_size=32 and num_workers=4")
    result_directory = output_directory_for(args.configuration_id)
    work_directory = result_directory.parent / f"{result_directory.name}_work"
    if result_directory.exists():
        raise FileExistsError(f"refusing to overwrite {result_directory}")
    forbidden_paths = [SAMPLE_DIRECTORY, result_directory, work_directory]
    if any("locked_test" in path.resolve().parts for path in forbidden_paths):
        raise RuntimeError("v3 certificate evaluation must not access the old test")

    registry_path = ROOT / "configs/experiment_registry_v6_compressibility.yaml"
    verify_v6_protocol_freeze(root=ROOT, registry_path=registry_path.resolve())
    registry = load_v6_registry(registry_path)
    evaluation_freeze = verify_evaluation_freeze()
    family, sample, sample_models = load_family_and_sample()
    hypothesis = {item["configuration_id"]: item for item in family["hypotheses"]}[
        args.configuration_id
    ]
    binding = sample_models[args.configuration_id]

    base_registry = yaml.safe_load((ROOT / registry["frozen_inputs"]["base_registry"]).read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / registry["frozen_inputs"]["frozen_manifest"]).read_text(encoding="utf-8"))
    raw_dataset_path = ROOT / registry["frozen_inputs"]["raw_dataset"]
    if sha256_file(raw_dataset_path) != frozen["dataset_sha256"]:
        raise RuntimeError("raw dataset changed")
    llm_path = ROOT / base_registry["frozen_base"]["language"]["local_path"]
    if sha256_file(llm_path) != frozen["llm_checkpoint_sha256"]:
        raise RuntimeError("LLM checkpoint changed")
    vision_path = ROOT / base_registry["frozen_base"]["vision"]["local_path"]
    for relative, key in {
        "model.safetensors": "vision_checkpoint_sha256",
        "config.json": "vision_config_sha256",
        "preprocessor_config.json": "vision_processor_config_sha256",
    }.items():
        if sha256_file(vision_path / relative) != frozen[key]:
            raise RuntimeError(f"vision artifact changed: {relative}")
    tokenizer_path = ROOT / base_registry["frozen_base"]["tokenizer"]["local_path"]
    if sha256_named_files(tokenizer_path, ["tokenizer.json", "tokenizer_config.json"]) != frozen["tokenizer_sha256"]:
        raise RuntimeError("tokenizer changed")

    artifact_directory = ROOT / hypothesis["artifact_directory"]
    artifact_manifest_path = artifact_directory / "manifest.json"
    if sha256_file(artifact_manifest_path) != hypothesis["artifact_manifest_sha256"]:
        raise RuntimeError("quantized artifact manifest changed")
    _, quantized = load_and_verify_quantized_arrays(artifact_directory)
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=hypothesis["compressed_model_descriptor"],
    )
    if compressed_hash != binding["compressed_model_sha256"]:
        raise RuntimeError("compressed-model binding failed")

    sample_manifest_path = SAMPLE_DIRECTORY / "manifest.json"
    sample_verification = json.loads(SAMPLE_VERIFICATION.read_text(encoding="utf-8"))
    if sample.get("status") != "v6_shared_certificate_sample_frozen_before_loss_evaluation":
        raise RuntimeError("shared sample is not frozen before evaluation")
    indices_path = SAMPLE_DIRECTORY / "sample_indices.npy"
    if sha256_file(indices_path) != sample["hashes"]["sample_indices_sha256"]:
        raise RuntimeError("shared sample indices changed")
    sample_indices = np.load(indices_path, allow_pickle=False)
    if sample_indices.dtype != np.uint32 or sample_indices.shape != (10000,):
        raise RuntimeError("shared sample index array violates its contract")
    selected_indices, multiplicities = np.unique(sample_indices, return_counts=True)
    selected_indices = selected_indices.astype(np.uint32)
    multiplicities = multiplicities.astype(np.uint32)

    descriptor = sample["sampling_descriptor"]
    train_manifest_path = ROOT / registry["frozen_inputs"]["train_manifest"]
    if sha256_file(train_manifest_path) != descriptor["train_manifest_sha256"]:
        raise RuntimeError("training manifest changed")
    train_hashes = read_cluster_hashes(train_manifest_path)
    if len(train_hashes) != descriptor["population_size"]:
        raise RuntimeError("training population size changed")
    records = build_certificate_caption_records(
        database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        selected_cluster_hashes=[train_hashes[int(index)] for index in selected_indices],
    )
    model, materialized_hash = construct_quantized_model(
        registry=registry,
        hypothesis=hypothesis,
        quantized=quantized,
        vision_path=vision_path,
        llm_path=llm_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    vocabulary_size = int(model.config.vocab_size)
    if tokenizer.vocab_size != vocabulary_size or len(tokenizer) != vocabulary_size:
        raise RuntimeError("tokenizer and model vocabulary sizes disagree")
    dataset = CertificateCaptionDataset(
        parquet_path=raw_dataset_path,
        records=records,
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt=registry["evaluation"]["canonical_prompt"],
        image_token_length=registry["evaluation"]["image_token_length"],
        max_sequence_length=registry["evaluation"]["max_sequence_length"],
    )

    contract = {
        "schema_version": 1,
        "status": "v6_fixed_before_certificate_forward_evaluation",
        "configuration_id": args.configuration_id,
        "candidate_id_code": hypothesis["candidate_id_code"],
        "parameterization": hypothesis["parameterization"],
        "intrinsic_dimension": hypothesis["intrinsic_dimension"],
        "lora_rank": hypothesis["lora_rank"],
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_version": CERTIFICATE_DATASET_VERSION,
        "risk_version": RISK_IMPLEMENTATION_VERSION,
        "alpha_grid_version": ALPHA_GRID_VERSION,
        "alpha_search_algorithm_version": ALPHA_SEARCH_ALGORITHM_VERSION,
        "tie_break": "minimum_bound_then_lower_alpha",
        "forward_dtype": "bfloat16_autocast",
        "correct_token_log_probability": "torch_log_softmax_fp32_then_gather_store_fp32",
        "smoothing_and_certificate_dtype": "numpy_float64_log_domain",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "caption_count": len(records),
        "unique_cluster_count": len(selected_indices),
        "sample_draw_count": len(sample_indices),
        "vocabulary_size": vocabulary_size,
        "caption_records_sha256": records_sha256(records),
        "raw_dataset_sha256": sha256_file(raw_dataset_path),
        "cluster_database_sha256": sha256_file(ROOT / "dataset/clusters/image_clusters.sqlite3"),
        "train_manifest_sha256": descriptor["train_manifest_sha256"],
        "family_manifest_sha256": sha256_file(FAMILY_MANIFEST),
        "family_verification_sha256": sha256_file(FAMILY_VERIFICATION),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_verification_sha256": sha256_file(SAMPLE_VERIFICATION),
        "artifact_manifest_sha256": hypothesis["artifact_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "quantized_hypothesis_sha256": binding["quantized_hypothesis_sha256"],
        "materialized_projector_parameters_sha256": materialized_hash,
        "evaluation_protocol_freeze_sha256": sha256_file(EVALUATION_FREEZE),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "shared_helper_source_sha256": sha256_file(ROOT / "evaluate_certificate_s4k.py"),
        "dataset_source_sha256": sha256_file(ROOT / "minimind_v_bound/certificate/dataset.py"),
        "risk_source_sha256": sha256_file(ROOT / "minimind_v_bound/certificate/risk.py"),
        "alpha_search_source_sha256": sha256_file(ROOT / "minimind_v_bound/certificate/search.py"),
        "K_VLM_upper_bits": binding["K_VLM_upper_bits"],
        "population_size": descriptor["population_size"],
        "delta_model": descriptor["delta_model"],
        "delta_subsample": descriptor["delta_subsample_family_total"],
        "model_count": descriptor["model_count"],
        "per_model_subsample_correction": descriptor["per_model_subsample_correction"],
        "max_denominator_bits": 16,
        "old_v1_test_accessed": False,
        "existing_certificate_sample_accessed": False,
    }
    if contract["model_count"] != 9 or contract["per_model_subsample_correction"] != "ln_9_over_delta_subsample":
        raise RuntimeError("formal v3 evaluation does not use the M=9 correction")

    work_directory.mkdir(parents=True, exist_ok=True)
    chunks_directory = work_directory / "chunks"
    chunks_directory.mkdir(exist_ok=True)
    contract_path = work_directory / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("existing work contract does not match")
    else:
        atomic_json(contract_path, contract)

    random.seed(20260816)
    np.random.seed(20260816)
    torch.manual_seed(20260816)
    torch.cuda.manual_seed_all(20260816)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=certificate_caption_collate,
        generator=torch.Generator().manual_seed(20260816),
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal v3 evaluation requires CUDA bfloat16 execution")
    torch.cuda.set_device(device)
    model.to(device).eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("not every model module is in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("certificate model still has a trainable parameter")

    started = time.perf_counter()
    completed_captions = 0
    for batch_index, batch in enumerate(loader):
        positions = batch["caption_position"].numpy()
        chunk_path = chunks_directory / f"batch_{batch_index:06d}.npz"
        if chunk_path.exists():
            chunk = load_chunk(chunk_path, positions)
            expected_clusters = batch["cluster_position"].numpy().astype(np.uint32)
            if not np.array_equal(chunk["cluster_positions"], expected_clusters):
                raise RuntimeError("resumed chunk cluster mapping mismatch")
        else:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = {key: value.to(device, non_blocking=True) for key, value in batch["pixel_values"].items()}
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, labels=None, pixel_values=pixel_values)
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            token_counts = valid.sum(dim=1).cpu().numpy().astype(np.uint16)
            if not np.array_equal(token_counts.astype(np.int64), batch["valid_label_count"].numpy()):
                raise RuntimeError("shifted label counts disagree with encoder")
            token_logits = output.logits[:, :-1, :][valid].float()
            targets = shifted_labels[valid]
            logp = torch.log_softmax(token_logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float32, copy=False)
            write_chunk(
                chunk_path,
                caption_positions=positions.astype(np.int64),
                cluster_positions=batch["cluster_position"].numpy().astype(np.uint32),
                token_counts=token_counts,
                correct_token_log_probabilities=logp,
            )
            chunk = load_chunk(chunk_path, positions)
        completed_captions += len(positions)
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
            print(json.dumps({
                "configuration_id": args.configuration_id,
                "batch": batch_index + 1,
                "batches": len(loader),
                "captions": completed_captions,
                "caption_total": len(dataset),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }, sort_keys=True), flush=True)

    positions_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    logp_parts: list[np.ndarray] = []
    for batch_index in range(len(loader)):
        start = batch_index * args.batch_size
        stop = min(start + args.batch_size, len(records))
        chunk = load_chunk(chunks_directory / f"batch_{batch_index:06d}.npz", np.arange(start, stop, dtype=np.int64))
        positions_parts.append(chunk["caption_positions"])
        cluster_parts.append(chunk["cluster_positions"])
        count_parts.append(chunk["token_counts"])
        logp_parts.append(chunk["correct_token_log_probabilities"])
    caption_positions = np.concatenate(positions_parts)
    caption_clusters = np.concatenate(cluster_parts)
    token_counts = np.concatenate(count_parts)
    logp = np.concatenate(logp_parts)
    if not np.array_equal(caption_positions, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("caption positions are not contiguous")
    expected_clusters = np.asarray([record.cluster_position for record in records], dtype=np.uint32)
    if not np.array_equal(caption_clusters, expected_clusters):
        raise RuntimeError("caption cluster mapping mismatch")
    caption_offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(token_counts, dtype=np.int64)])
    token_weights = hierarchical_token_weights(
        caption_offsets=caption_offsets,
        caption_cluster_positions=caption_clusters,
        cluster_multiplicities=multiplicities,
    )
    search = search_reduced_dyadic_alpha(
        log_probabilities=logp,
        token_weights=token_weights,
        vocabulary_size=vocabulary_size,
        max_denominator_bits=contract["max_denominator_bits"],
        model_description_bits=contract["K_VLM_upper_bits"],
        population_size=contract["population_size"],
        sample_size=contract["sample_draw_count"],
        delta_model=contract["delta_model"],
        delta_subsample=contract["delta_subsample"],
        model_count=contract["model_count"],
    )
    selected_losses = smoothed_token_losses_bits(logp, alpha=search.alpha, vocabulary_size=vocabulary_size)
    caption_losses = np.add.reduceat(selected_losses, caption_offsets[:-1]) / token_counts
    captions_per_cluster = np.bincount(caption_clusters.astype(np.int64), minlength=len(selected_indices))
    if np.any(captions_per_cluster == 0):
        raise RuntimeError("a selected cluster has no caption")
    cluster_losses = np.bincount(caption_clusters.astype(np.int64), weights=caption_losses, minlength=len(selected_indices)) / captions_per_cluster
    hierarchical_empirical = float(np.dot(multiplicities.astype(np.float64), cluster_losses) / len(sample_indices))
    if not math.isclose(hierarchical_empirical, search.empirical_risk_bits_per_token, rel_tol=0.0, abs_tol=2e-12):
        raise RuntimeError("hierarchical empirical risk mismatch")
    lower = float(search.certificate["loss_lower_bits_per_token"])
    upper = float(search.certificate["loss_upper_bits_per_token"])
    if np.any(cluster_losses < lower - 2e-12) or np.any(cluster_losses > upper + 2e-12):
        raise RuntimeError("a cluster loss lies outside its analytic interval")

    result_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{result_directory.name}.", dir=result_directory.parent))
    try:
        arrays = {
            "correct_token_log_probabilities": logp.astype(np.float32, copy=False),
            "caption_offsets": caption_offsets,
            "caption_cluster_positions": caption_clusters,
            "selected_train_manifest_indices": selected_indices,
            "cluster_multiplicities": multiplicities,
            "selected_alpha_caption_losses": caption_losses.astype(np.float64),
            "selected_alpha_cluster_losses": cluster_losses.astype(np.float64),
        }
        file_hashes: dict[str, str] = {}
        for name, array in arrays.items():
            path = temporary / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            file_hashes[f"{name}_sha256"] = sha256_file(path)
        records_path = temporary / "caption_records.jsonl"
        with records_path.open("x", encoding="utf-8", newline="\n") as output:
            for record, count in zip(records, token_counts, strict=True):
                output.write(json.dumps({
                    "caption_position": record.caption_position,
                    "cluster_position": record.cluster_position,
                    "cluster_sha256": record.cluster_sha256,
                    "representative_row_index": record.representative_row_index,
                    "source_row_index": record.source_row_index,
                    "source_caption_index": record.source_caption_index,
                    "caption_sha256": hashlib.sha256(record.caption.encode("utf-8")).hexdigest(),
                    "valid_token_count": int(count),
                }, sort_keys=True, separators=(",", ":")) + "\n")
        file_hashes["caption_records_sha256"] = sha256_file(records_path)
        result_manifest = {
            "schema_version": 1,
            "status": "v6_certificate_final_alpha_selected_without_old_test",
            "evaluation_contract": contract,
            "evaluation_summary": {
                "caption_count": len(records),
                "correct_token_count": len(logp),
                "unique_cluster_count": len(selected_indices),
                "sample_draw_count": len(sample_indices),
                "minimum_cluster_loss_bits_per_token": float(cluster_losses.min()),
                "maximum_cluster_loss_bits_per_token": float(cluster_losses.max()),
            },
            "selected_alpha": {
                "numerator": search.numerator,
                "denominator_bits": search.denominator_bits,
                "denominator": 1 << search.denominator_bits,
                "value": search.alpha,
                "code_length_bits": search.alpha_code_length_bits,
                "tie_break": "minimum_bound_then_lower_alpha",
            },
            "certificate": search.certificate,
            "search_audit": {
                "grid_candidate_count": (1 << contract["max_denominator_bits"]) - 1,
                "exact_objective_evaluations": search.exact_objective_evaluations,
                "derivative_evaluations": search.derivative_evaluations,
                "branch_lower_bound_evaluations": search.branch_lower_bound_evaluations,
                "pruned_grid_candidates": search.pruned_grid_candidates,
                "hierarchical_empirical_recalculation": hierarchical_empirical,
            },
            "non_vacuity": {
                "uniform_predictor_bits_per_token": math.log2(vocabulary_size),
                "certificate_strictly_below_uniform": float(search.certificate["certificate_bound_bits_per_token"]) < math.log2(vocabulary_size),
            },
            "files": file_hashes,
            "old_v1_test_accessed": False,
            "existing_certificate_sample_accessed": False,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(result_manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.rename(temporary, result_directory)
        for artifact in result_directory.iterdir():
            artifact.chmod(0o444)
        result_directory.chmod(0o555)
        for chunk in chunks_directory.iterdir():
            chunk.chmod(0o444)
        contract_path.chmod(0o444)
        chunks_directory.chmod(0o555)
        work_directory.chmod(0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "configuration_id": args.configuration_id,
        "output_directory": str(result_directory),
        "alpha_fraction": f"{search.numerator}/{1 << search.denominator_bits}",
        "empirical_risk_bits_per_token": search.empirical_risk_bits_per_token,
        "model_complexity_term_bits_per_token": search.certificate["model_complexity_term_bits_per_token"],
        "subsample_complexity_term_bits_per_token": search.certificate["subsample_complexity_term_bits_per_token"],
        "certificate_bound_bits_per_token": search.certificate["certificate_bound_bits_per_token"],
        "uniform_predictor_bits_per_token": math.log2(vocabulary_size),
        "non_vacuous": result_manifest["non_vacuity"]["certificate_strictly_below_uniform"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
