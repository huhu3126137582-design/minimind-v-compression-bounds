from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from evaluate_certificate_s4k import atomic_json, load_chunk, records_sha256, write_chunk
from evaluate_certificate_v6 import (
    CANDIDATE_ORDER,
    FAMILY_MANIFEST,
    FAMILY_VERIFICATION,
    OUTPUT_SLUGS,
    construct_quantized_model,
)
from minimind_v_bound.certificate.dataset import (
    CERTIFICATE_DATASET_VERSION,
    CertificateCaptionDataset,
    build_certificate_caption_records,
    certificate_caption_collate,
)
from minimind_v_bound.certificate.sampling import read_cluster_hashes
from minimind_v_bound.compression.artifact import (
    canonical_compressed_model_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.configuration_v6_compressibility import (
    load_v6_registry,
    verify_v6_protocol_freeze,
)
from minimind_v_bound.evaluation.heldout_validation_v5 import (
    HELDOUT_VALIDATION_METRICS_VERSION,
    RANK_ALPHA,
    caption_and_cluster_losses,
)


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EVALUATOR_VERSION = "v6-nine-q11-heldout-validation-bf16-v1"
PROTOCOL_FREEZE = ROOT / "configs/v6_q11_only_protocol_revision_manifest.json"
RESULT_ROOT = ROOT / "runs/v6_compressibility/heldout_validation"
FORMAL_BATCH_SIZE = 32
FORMAL_NUM_WORKERS = 4
FORMAL_SEED = 20260818


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen model on the fixed MiniMind-V held-out clusters"
    )
    parser.add_argument("--configuration-id", choices=CANDIDATE_ORDER, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    return parser.parse_args()


def result_directory_for(configuration_id: str) -> Path:
    return RESULT_ROOT / OUTPUT_SLUGS[configuration_id]


def certificate_directory_for(configuration_id: str) -> Path:
    return ROOT / "runs/v6_compressibility/certificate_results" / OUTPUT_SLUGS[configuration_id]


def _load_protocol_freeze() -> dict:
    freeze = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "q11_only_scope_revision_frozen_before_resume":
        raise RuntimeError("the v6 held-out validation protocol is not frozen")
    if freeze.get("model_count") != 9 or RANK_ALPHA != 1 / 8:
        raise RuntimeError("the frozen v6 ranking contract changed")
    for relative, expected in freeze["artifact_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen v6 source changed: {relative}")
    return freeze


def _load_frozen_inputs(configuration_id: str) -> tuple[dict, dict, dict, dict]:
    family = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
    if family.get("status") != "v6_nine_q11_quantized_hypotheses_frozen_before_shared_sampling":
        raise RuntimeError("the nine-model family is not frozen")
    family_verification = json.loads(FAMILY_VERIFICATION.read_text(encoding="utf-8"))
    if family_verification.get("status") != (
        "v6_nine_q11_quantized_hypotheses_independently_verified_before_sampling"
    ):
        raise RuntimeError("the nine-model family lacks independent verification")
    hypotheses = {item["configuration_id"]: item for item in family["hypotheses"]}
    if tuple(hypotheses) != CANDIDATE_ORDER:
        raise RuntimeError("the nine-model family order changed")

    certificate_path = certificate_directory_for(configuration_id) / "manifest.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("status") != "v6_certificate_final_alpha_selected_without_old_test":
        raise RuntimeError("the final certificate is not frozen")
    if certificate.get("old_v1_test_accessed") or certificate.get(
        "existing_certificate_sample_accessed"
    ):
        raise RuntimeError("the v3 certificate reports prohibited data access")

    entry = {"ranking_alpha": {"value": RANK_ALPHA}, "ranking_certificate": certificate["certificate"]}
    return hypotheses[configuration_id], certificate, entry, family


def _write_caption_records(path: Path, records: list, token_counts: np.ndarray) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for record, count in zip(records, token_counts, strict=True):
            output.write(
                json.dumps(
                    {
                        "caption_position": record.caption_position,
                        "cluster_position": record.cluster_position,
                        "cluster_sha256": record.cluster_sha256,
                        "representative_row_index": record.representative_row_index,
                        "source_row_index": record.source_row_index,
                        "source_caption_index": record.source_caption_index,
                        "caption_sha256": hashlib.sha256(
                            record.caption.encode("utf-8")
                        ).hexdigest(),
                        "valid_token_count": int(count),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    if args.batch_size != FORMAL_BATCH_SIZE or args.num_workers != FORMAL_NUM_WORKERS:
        raise ValueError("formal v6 evaluation fixes batch_size=32 and num_workers=4")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal v6 evaluation requires CUDA bfloat16 execution")
    result_directory = result_directory_for(args.configuration_id)
    work_directory = result_directory.parent / f"{result_directory.name}_work"
    if result_directory.exists():
        raise FileExistsError(f"refusing to overwrite {result_directory}")

    freeze = _load_protocol_freeze()
    hypothesis, certificate, ranking_entry, family = _load_frozen_inputs(
        args.configuration_id
    )
    binding = {
        "quantized_hypothesis_sha256": hypothesis["hashes"]["quantized_hypothesis_sha256"],
        "final_alpha": certificate["selected_alpha"]["value"],
        "ranking_certificate_bits_per_token": certificate["certificate"]["certificate_bound_bits_per_token"],
    }

    registry_path = ROOT / "configs/experiment_registry_v6_compressibility.yaml"
    verify_v6_protocol_freeze(root=ROOT, registry_path=registry_path.resolve())
    registry = load_v6_registry(registry_path)
    base_registry = yaml.safe_load(
        (ROOT / registry["frozen_inputs"]["base_registry"]).read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (ROOT / registry["frozen_inputs"]["frozen_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    raw_dataset_path = ROOT / registry["frozen_inputs"]["raw_dataset"]
    if sha256_file(raw_dataset_path) != frozen["dataset_sha256"]:
        raise RuntimeError("the raw MiniMind-V parquet changed")
    llm_path = ROOT / base_registry["frozen_base"]["language"]["local_path"]
    vision_path = ROOT / base_registry["frozen_base"]["vision"]["local_path"]
    if sha256_file(llm_path) != frozen["llm_checkpoint_sha256"]:
        raise RuntimeError("the frozen language model changed")
    for relative, key in {
        "model.safetensors": "vision_checkpoint_sha256",
        "config.json": "vision_config_sha256",
        "preprocessor_config.json": "vision_processor_config_sha256",
    }.items():
        if sha256_file(vision_path / relative) != frozen[key]:
            raise RuntimeError(f"the frozen vision input changed: {relative}")

    split_path = ROOT / "dataset/manifests/split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation_manifest_path = ROOT / split["test_manifest"]
    train_manifest_path = ROOT / split["train_manifest"]
    if sha256_file(validation_manifest_path) != split["test_manifest_sha256"]:
        raise RuntimeError("the held-out validation manifest changed")
    if sha256_file(train_manifest_path) != split["train_manifest_sha256"]:
        raise RuntimeError("the training manifest changed")
    validation_hashes = read_cluster_hashes(validation_manifest_path)
    train_hashes = read_cluster_hashes(train_manifest_path)
    if len(validation_hashes) != 62374 or set(validation_hashes).intersection(train_hashes):
        raise RuntimeError("the held-out validation split violates its frozen contract")
    records = build_certificate_caption_records(
        database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        selected_cluster_hashes=validation_hashes,
    )

    artifact_directory = ROOT / hypothesis["artifact_directory"]
    artifact_manifest_path = artifact_directory / "manifest.json"
    if sha256_file(artifact_manifest_path) != hypothesis["artifact_manifest_sha256"]:
        raise RuntimeError("the frozen quantized artifact changed")
    _, quantized = load_and_verify_quantized_arrays(artifact_directory)
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=hypothesis["compressed_model_descriptor"],
    )
    if compressed_hash != hypothesis["hashes"]["compressed_model_sha256"]:
        raise RuntimeError("the compressed model hash changed")
    model, materialized_hash = construct_quantized_model(
        registry=registry,
        hypothesis=hypothesis,
        quantized=quantized,
        vision_path=vision_path,
        llm_path=llm_path,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    vocabulary_size = int(model.config.vocab_size)
    if tokenizer.vocab_size != vocabulary_size or len(tokenizer) != vocabulary_size:
        raise RuntimeError("the tokenizer and model vocabulary sizes disagree")
    dataset = CertificateCaptionDataset(
        parquet_path=raw_dataset_path,
        records=records,
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt=registry["evaluation"]["canonical_prompt"],
        image_token_length=registry["evaluation"]["image_token_length"],
        max_sequence_length=registry["evaluation"]["max_sequence_length"],
    )

    final_alpha = float(certificate["selected_alpha"]["value"])
    contract = {
        "schema_version": 1,
        "status": "v6_fixed_before_model_validation_forward_pass",
        "configuration_id": args.configuration_id,
        "candidate_order": list(CANDIDATE_ORDER),
        "model_count": 9,
        "evaluator_version": EVALUATOR_VERSION,
        "metrics_version": HELDOUT_VALIDATION_METRICS_VERSION,
        "dataset_version": CERTIFICATE_DATASET_VERSION,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "forward_dtype": "bfloat16_autocast",
        "correct_token_log_probability": "torch_log_softmax_fp32_then_gather_store_fp32",
        "statistics_dtype": "numpy_float64_log_domain",
        "validation_cluster_count": len(validation_hashes),
        "caption_count": len(records),
        "caption_records_sha256": records_sha256(records),
        "vocabulary_size": vocabulary_size,
        "rank_alpha": RANK_ALPHA,
        "final_alpha": final_alpha,
        "raw_bpd_prediction_smoothing": False,
        "alpha_search_performed_on_validation": False,
        "model_updated_from_validation": False,
        "validation_used_for_certificate": False,
        "raw_dataset_sha256": sha256_file(raw_dataset_path),
        "split_manifest_sha256": sha256_file(split_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "validation_manifest_sha256": sha256_file(validation_manifest_path),
        "cluster_database_sha256": sha256_file(
            ROOT / "dataset/clusters/image_clusters.sqlite3"
        ),
        "family_manifest_sha256": sha256_file(FAMILY_MANIFEST),
        "family_verification_sha256": sha256_file(FAMILY_VERIFICATION),
        "artifact_manifest_sha256": hypothesis["artifact_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "quantized_hypothesis_sha256": hypothesis["hashes"][
            "quantized_hypothesis_sha256"
        ],
        "materialized_projector_parameters_sha256": materialized_hash,
        "final_certificate_manifest_sha256": sha256_file(
            certificate_directory_for(args.configuration_id) / "manifest.json"
        ),
        "ranking_manifest_sha256": sha256_file(FAMILY_MANIFEST),
        "ranking_certificate_bits_per_token": ranking_entry[
            "ranking_certificate"
        ]["certificate_bound_bits_per_token"],
        "certification_empirical_rank_risk_bits_per_token": ranking_entry[
            "ranking_certificate"
        ]["empirical_risk_bits_per_token"],
        "protocol_freeze_sha256": sha256_file(PROTOCOL_FREEZE),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
    }
    if contract["final_alpha"] != binding["final_alpha"]:
        raise RuntimeError("the final alpha differs from the frozen v6 binding")
    if contract["ranking_certificate_bits_per_token"] != binding[
        "ranking_certificate_bits_per_token"
    ]:
        raise RuntimeError("the ranking certificate differs from the v6 binding")

    work_directory.mkdir(parents=True, exist_ok=True)
    chunks_directory = work_directory / "chunks"
    chunks_directory.mkdir(exist_ok=True)
    contract_path = work_directory / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("the existing v6 work contract differs")
    else:
        atomic_json(contract_path, contract)

    random.seed(FORMAL_SEED)
    np.random.seed(FORMAL_SEED)
    torch.manual_seed(FORMAL_SEED)
    torch.cuda.manual_seed_all(FORMAL_SEED)
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
        generator=torch.Generator().manual_seed(FORMAL_SEED),
    )
    torch.cuda.set_device(device)
    model.to(device).eval()
    if any(module.training for module in model.modules()) or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("the held-out model is not completely frozen in eval mode")

    started = time.perf_counter()
    completed_captions = 0
    for batch_index, batch in enumerate(loader):
        positions = batch["caption_position"].numpy()
        chunk_path = chunks_directory / f"batch_{batch_index:06d}.npz"
        if chunk_path.exists():
            chunk = load_chunk(chunk_path, positions)
            if not np.array_equal(
                chunk["cluster_positions"],
                batch["cluster_position"].numpy().astype(np.uint32),
            ):
                raise RuntimeError("a resumed v6 chunk has the wrong cluster mapping")
        else:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["pixel_values"].items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = model(input_ids=input_ids, labels=None, pixel_values=pixel_values)
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            token_counts = valid.sum(dim=1).cpu().numpy().astype(np.uint16)
            if not np.array_equal(
                token_counts.astype(np.int64), batch["valid_label_count"].numpy()
            ):
                raise RuntimeError("the v6 shifted-label counts changed")
            token_logits = output.logits[:, :-1, :][valid].float()
            targets = shifted_labels[valid]
            logp = (
                torch.log_softmax(token_logits, dim=-1)
                .gather(1, targets.unsqueeze(1))
                .squeeze(1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            write_chunk(
                chunk_path,
                caption_positions=positions.astype(np.int64),
                cluster_positions=batch["cluster_position"].numpy().astype(np.uint32),
                token_counts=token_counts,
                correct_token_log_probabilities=logp,
            )
        completed_captions += len(positions)
        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            print(
                json.dumps(
                    {
                        "configuration_id": args.configuration_id,
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "captions": completed_captions,
                        "caption_total": len(dataset),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    positions_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    logp_parts: list[np.ndarray] = []
    for batch_index in range(len(loader)):
        start = batch_index * args.batch_size
        stop = min(start + args.batch_size, len(records))
        chunk = load_chunk(
            chunks_directory / f"batch_{batch_index:06d}.npz",
            np.arange(start, stop, dtype=np.int64),
        )
        positions_parts.append(chunk["caption_positions"])
        cluster_parts.append(chunk["cluster_positions"])
        count_parts.append(chunk["token_counts"])
        logp_parts.append(chunk["correct_token_log_probabilities"])
    caption_positions = np.concatenate(positions_parts)
    caption_clusters = np.concatenate(cluster_parts)
    token_counts = np.concatenate(count_parts)
    logp = np.concatenate(logp_parts)
    if not np.array_equal(caption_positions, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("the v6 caption positions are not contiguous")
    expected_clusters = np.asarray(
        [record.cluster_position for record in records], dtype=np.uint32
    )
    if not np.array_equal(caption_clusters, expected_clusters):
        raise RuntimeError("the v6 caption-to-cluster mapping changed")
    offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(token_counts, dtype=np.int64)]
    )

    raw = caption_and_cluster_losses(
        correct_token_log_probabilities=logp,
        caption_offsets=offsets,
        caption_cluster_positions=caption_clusters,
        cluster_count=len(validation_hashes),
        vocabulary_size=vocabulary_size,
        alpha=None,
    )
    rank = caption_and_cluster_losses(
        correct_token_log_probabilities=logp,
        caption_offsets=offsets,
        caption_cluster_positions=caption_clusters,
        cluster_count=len(validation_hashes),
        vocabulary_size=vocabulary_size,
        alpha=RANK_ALPHA,
    )
    final = caption_and_cluster_losses(
        correct_token_log_probabilities=logp,
        caption_offsets=offsets,
        caption_cluster_positions=caption_clusters,
        cluster_count=len(validation_hashes),
        vocabulary_size=vocabulary_size,
        alpha=final_alpha,
    )

    result_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result_directory.name}.", dir=result_directory.parent)
    )
    try:
        arrays = {
            "correct_token_log_probabilities": logp.astype(np.float32, copy=False),
            "caption_offsets": offsets,
            "caption_cluster_positions": caption_clusters,
            "raw_caption_losses": raw["caption_losses_bits_per_token"],
            "raw_cluster_losses": raw["cluster_losses_bits_per_token"],
            "rank_alpha_caption_losses": rank["caption_losses_bits_per_token"],
            "rank_alpha_cluster_losses": rank["cluster_losses_bits_per_token"],
            "final_alpha_caption_losses": final["caption_losses_bits_per_token"],
            "final_alpha_cluster_losses": final["cluster_losses_bits_per_token"],
        }
        file_hashes: dict[str, str] = {}
        for name, array in arrays.items():
            path = temporary / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            file_hashes[f"{name}_sha256"] = sha256_file(path)
        records_path = temporary / "caption_records.jsonl"
        _write_caption_records(records_path, records, token_counts)
        file_hashes["caption_records_sha256"] = sha256_file(records_path)
        manifest = {
            "schema_version": 1,
            "status": "v6_heldout_validation_model_evaluation_final",
            "evaluation_contract": contract,
            "metrics": {
                "raw_validation_bpd_bits_per_token": raw["risk_bits_per_token"],
                "rank_alpha_validation_risk_bits_per_token": rank[
                    "risk_bits_per_token"
                ],
                "final_alpha_validation_risk_bits_per_token": final[
                    "risk_bits_per_token"
                ],
            },
            "summary": {
                "validation_cluster_count": len(validation_hashes),
                "caption_count": len(records),
                "correct_token_count": len(logp),
            },
            "files": file_hashes,
            "alpha_search_performed_on_validation": False,
            "model_updated_from_validation": False,
            "validation_used_for_certificate": False,
        }
        with (temporary / "manifest.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
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

    print(
        json.dumps(
            {
                "configuration_id": args.configuration_id,
                "output_directory": str(result_directory),
                "raw_validation_bpd_bits_per_token": raw["risk_bits_per_token"],
                "rank_alpha_validation_risk_bits_per_token": rank[
                    "risk_bits_per_token"
                ],
                "final_alpha_validation_risk_bits_per_token": final[
                    "risk_bits_per_token"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
