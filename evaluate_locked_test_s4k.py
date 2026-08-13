from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import stat
import subprocess
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

from minimind_v_bound.certificate.dataset import (  # noqa: E402
    CERTIFICATE_DATASET_VERSION,
    CertificateCaptionDataset,
    build_certificate_caption_records,
    certificate_caption_collate,
)
from minimind_v_bound.certificate.risk import (  # noqa: E402
    RISK_IMPLEMENTATION_VERSION,
    loss_interval_bits,
    smoothed_token_losses_bits,
)
from minimind_v_bound.certificate.sampling import read_cluster_hashes  # noqa: E402
from minimind_v_bound.compression.artifact import (  # noqa: E402
    canonical_named_tensor_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.evaluation.test_statistics import (  # noqa: E402
    TEST_INTERVAL_VERSION,
    test_risk_interval_and_classification,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
)


LOCKED_TEST_EVALUATOR_VERSION = "s4k-locked-test-fixed-alpha-bf16-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform the one-time locked-test evaluation of frozen S-4K"
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--work-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/locked_test_evaluation_work",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/locked_test_evaluation",
    )
    parser.add_argument(
        "--unlock-receipt",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/locked_test_unlock_receipt.json",
    )
    return parser.parse_args()


def require_read_only_tree(directory: Path) -> None:
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if directory.stat().st_mode & write_bits:
        raise RuntimeError(f"frozen directory is writable by mode: {directory}")
    for path in directory.iterdir():
        if path.stat().st_mode & write_bits:
            raise RuntimeError(f"frozen artifact is writable by mode: {path}")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def records_sha256(records) -> str:
    digest = hashlib.sha256(b"minimind-v-bound-locked-test-caption-records-v1")
    for record in records:
        payload = canonical_json(
            {
                "caption_position": record.caption_position,
                "cluster_position": record.cluster_position,
                "cluster_sha256": record.cluster_sha256,
                "representative_row_index": record.representative_row_index,
                "source_row_index": record.source_row_index,
                "source_caption_index": record.source_caption_index,
                "caption_sha256": hashlib.sha256(record.caption.encode("utf-8")).hexdigest(),
            }
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sha256_named_files(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        encoded_name = relative_path.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        path = root / relative_path
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.rename(temporary, path)


def write_chunk(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as output:
        np.savez(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    os.rename(temporary, path)


def load_chunk(path: Path, expected_positions: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        chunk = {name: archive[name] for name in archive.files}
    if set(chunk) != {
        "caption_positions",
        "cluster_positions",
        "token_counts",
        "correct_token_log_probabilities",
    }:
        raise RuntimeError(f"test chunk has unexpected fields: {path}")
    if not np.array_equal(chunk["caption_positions"], expected_positions):
        raise RuntimeError(f"test chunk caption ordering mismatch: {path}")
    counts = chunk["token_counts"]
    logp = chunk["correct_token_log_probabilities"]
    if counts.dtype != np.uint16 or np.any(counts == 0):
        raise RuntimeError(f"test chunk has invalid token counts: {path}")
    if logp.dtype != np.float32 or len(logp) != int(counts.astype(np.int64).sum()):
        raise RuntimeError(f"test chunk token data mismatch: {path}")
    if not np.isfinite(logp).all() or np.any(logp > 0.0):
        raise RuntimeError(f"test chunk has invalid log probabilities: {path}")
    return chunk


def verify_certificate_files(certificate_directory: Path, manifest: dict) -> None:
    for key, expected_hash in manifest["files"].items():
        filename = key.removesuffix("_sha256")
        suffix = ".jsonl" if filename == "caption_records" else ".npy"
        if sha256_file(certificate_directory / f"{filename}{suffix}") != expected_hash:
            raise RuntimeError(f"certificate artifact hash mismatch: {filename}")


def main() -> None:
    args = parse_args()
    if args.batch_size != 32 or args.num_workers != 4:
        raise ValueError("formal locked-test evaluation requires batch_size=32, workers=4")
    work_directory = args.work_directory.resolve()
    output_directory = args.output_directory.resolve()
    unlock_receipt_path = args.unlock_receipt.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    registry_path = ROOT / "configs/experiment_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    frozen_manifest_path = ROOT / "configs/frozen_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    for repository, expected_commit in (
        (ROOT / "upstream/minimind-v", frozen_manifest["minimind_v_git_commit"]),
        (
            ROOT / "upstream/SubLoRA-bounds-for-LLMs",
            frozen_manifest["sublora_bounds_git_commit"],
        ),
    ):
        actual_commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        )
        if actual_commit != expected_commit or dirty:
            raise RuntimeError(f"frozen upstream repository changed: {repository}")
    raw_dataset_path = ROOT / registry["data"]["local_path"]
    if sha256_file(raw_dataset_path) != frozen_manifest["dataset_sha256"]:
        raise RuntimeError("raw dataset changed before locked-test evaluation")
    llm_path = ROOT / registry["frozen_base"]["language"]["local_path"]
    if sha256_file(llm_path) != frozen_manifest["llm_checkpoint_sha256"]:
        raise RuntimeError("LLM checkpoint changed before locked-test evaluation")
    vision_path = ROOT / registry["frozen_base"]["vision"]["local_path"]
    for relative_path, manifest_key in {
        "model.safetensors": "vision_checkpoint_sha256",
        "config.json": "vision_config_sha256",
        "preprocessor_config.json": "vision_processor_config_sha256",
    }.items():
        if sha256_file(vision_path / relative_path) != frozen_manifest[manifest_key]:
            raise RuntimeError(f"vision artifact changed: {relative_path}")
    tokenizer_path = ROOT / registry["frozen_base"]["tokenizer"]["local_path"]
    if sha256_named_files(
        tokenizer_path, ["tokenizer.json", "tokenizer_config.json"]
    ) != frozen_manifest["tokenizer_sha256"]:
        raise RuntimeError("tokenizer changed before locked-test evaluation")
    split_manifest_path = ROOT / "dataset/manifests/split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    test_manifest_path = ROOT / split_manifest["test_manifest"]
    if "locked_test" not in test_manifest_path.parts:
        raise RuntimeError("registered test manifest is not in the locked-test area")
    if stat.S_IMODE(test_manifest_path.stat().st_mode) != 0o400:
        raise RuntimeError("locked test manifest does not retain mode 0400")

    certificate_directory = ROOT / "runs/s4k_formal_v1/certificate_evaluation"
    certificate_verification_path = (
        ROOT / "runs/s4k_formal_v1/certificate_evaluation_verification.json"
    )
    quantized_directory = ROOT / "runs/s4k_formal_v1/quantized_q11"
    sample_directory = ROOT / "runs/s4k_formal_v1/certificate_sample_n10000"
    for directory in (certificate_directory, quantized_directory, sample_directory):
        require_read_only_tree(directory)
    certificate_manifest_path = certificate_directory / "manifest.json"
    certificate_manifest = json.loads(
        certificate_manifest_path.read_text(encoding="utf-8")
    )
    if certificate_manifest["status"] != "certificate_final_alpha_selected_before_locked_test":
        raise RuntimeError("certificate was not finalized before test unlock")
    verify_certificate_files(certificate_directory, certificate_manifest)
    certificate_verification = json.loads(
        certificate_verification_path.read_text(encoding="utf-8")
    )
    if certificate_verification[
        "status"
    ] != "certificate_independently_recomputed_and_verified":
        raise RuntimeError("certificate independent verification is missing")
    if certificate_verification["certificate_manifest_sha256"] != sha256_file(
        certificate_manifest_path
    ):
        raise RuntimeError("certificate verification is not bound to current certificate")

    selected_alpha = certificate_manifest["selected_alpha"]
    if (selected_alpha["numerator"], selected_alpha["denominator_bits"]) != (1, 3):
        raise RuntimeError("final certificate alpha is not the frozen value 1/8")
    alpha = selected_alpha["value"]
    certificate_bound = certificate_manifest["certificate"][
        "certificate_bound_bits_per_token"
    ]
    eta = registry["test"]["eta"]
    model_count = registry["certificate"]["model_count"]
    if eta != 0.05 or model_count != 1:
        raise RuntimeError("test inference contract differs from preregistration")

    evaluator_path = Path(__file__).resolve()
    statistics_source_path = ROOT / "minimind_v_bound/evaluation/test_statistics.py"
    pre_unlock_receipt = {
        "schema_version": 1,
        "status": "test_unlock_preconditions_frozen_before_first_manifest_read",
        "authorization": "user_requested_next_registered_step_after_final_certificate",
        "test_manifest": {
            "path": str(test_manifest_path.relative_to(ROOT)),
            "expected_sha256_from_preexisting_split_manifest": split_manifest[
                "test_manifest_sha256"
            ],
            "expected_cluster_count": split_manifest["test_clusters"],
            "pre_unlock_mode": "0400",
            "content_read_before_this_receipt": False,
        },
        "frozen_inputs": {
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
            "raw_dataset_sha256": frozen_manifest["dataset_sha256"],
            "llm_checkpoint_sha256": frozen_manifest["llm_checkpoint_sha256"],
            "vision_checkpoint_sha256": frozen_manifest["vision_checkpoint_sha256"],
            "tokenizer_sha256": frozen_manifest["tokenizer_sha256"],
            "quantized_manifest_sha256": sha256_file(
                quantized_directory / "manifest.json"
            ),
            "certificate_sample_manifest_sha256": sha256_file(
                sample_directory / "manifest.json"
            ),
            "certificate_manifest_sha256": sha256_file(certificate_manifest_path),
            "certificate_verification_sha256": sha256_file(
                certificate_verification_path
            ),
            "quantized_hypothesis_sha256": certificate_manifest[
                "evaluation_contract"
            ]["quantized_hypothesis_sha256"],
            "K_VLM_upper_bits": certificate_manifest["evaluation_contract"][
                "K_VLM_upper_bits"
            ],
            "alpha_fraction": "1/8",
            "alpha_code_length_bits": selected_alpha["code_length_bits"],
            "certificate_bound_bits_per_token": certificate_bound,
        },
        "fixed_test_rules": {
            "test_eta": eta,
            "model_count": model_count,
            "interval_version": TEST_INTERVAL_VERSION,
            "radius_formula": "Delta_alpha*sqrt(ln(2*M/eta)/(2*N_test))",
            "interval_clipping": "[max(a_alpha,R-e),min(b_alpha,R+e)]",
            "classification": [
                "strong_support_if_U_le_B",
                "compatible_but_uncertain_if_L_le_B_lt_U",
                "statistically_significant_violation_signal_if_L_gt_B",
            ],
            "alpha_reselection_after_unlock": False,
            "model_or_quantizer_update_after_unlock": False,
            "single_model_ranking_statistics": "not_applicable",
        },
        "implementation": {
            "evaluator_version": LOCKED_TEST_EVALUATOR_VERSION,
            "evaluator_source_sha256": sha256_file(evaluator_path),
            "statistics_source_sha256": sha256_file(statistics_source_path),
            "dataset_source_sha256": sha256_file(
                ROOT / "minimind_v_bound/certificate/dataset.py"
            ),
            "risk_source_sha256": sha256_file(
                ROOT / "minimind_v_bound/certificate/risk.py"
            ),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "forward_dtype": "bfloat16_autocast",
            "smoothing_and_statistics_dtype": "numpy_float64_log_domain",
        },
    }
    unlock_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if unlock_receipt_path.exists():
        existing_receipt = json.loads(unlock_receipt_path.read_text(encoding="utf-8"))
        if existing_receipt != pre_unlock_receipt:
            raise RuntimeError("existing unlock receipt differs from fixed preconditions")
    else:
        atomic_json(unlock_receipt_path, pre_unlock_receipt)
        unlock_receipt_path.chmod(0o444)

    # The first content read of the locked test manifest occurs only after the
    # immutable receipt above exists and fixes every statistical decision.
    actual_test_manifest_hash = sha256_file(test_manifest_path)
    if actual_test_manifest_hash != split_manifest["test_manifest_sha256"]:
        raise RuntimeError("unlocked test manifest hash differs from frozen split")
    test_hashes = read_cluster_hashes(test_manifest_path)
    if len(test_hashes) != split_manifest["test_clusters"]:
        raise RuntimeError("unlocked test cluster count differs from frozen split")
    train_manifest_path = ROOT / split_manifest["train_manifest"]
    train_hashes = read_cluster_hashes(train_manifest_path)
    if set(test_hashes).intersection(train_hashes):
        raise RuntimeError("train and unlocked test clusters overlap")

    records = build_certificate_caption_records(
        database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        selected_cluster_hashes=test_hashes,
    )
    record_hash = records_sha256(records)

    quantized_manifest, quantized = load_and_verify_quantized_arrays(
        quantized_directory
    )
    if quantized_manifest["hashes"]["quantized_hypothesis_sha256"] != certificate_manifest[
        "evaluation_contract"
    ]["quantized_hypothesis_sha256"]:
        raise RuntimeError("test model differs from the certified quantized hypothesis")
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
    weights = torch.load(llm_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError("pinned LLM checkpoint does not match architecture")
    wrapper = prepare_intrinsic_projector(model, intrinsic_dimension=4096, seed=137)
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(quantized.reconstructed))
    projector_hash = canonical_named_tensor_sha256(
        list(wrapper.materialized_parameters().items())
    )
    if projector_hash != quantized_manifest["hashes"]["materialized_projector_parameters_sha256"]:
        raise RuntimeError("test Projector differs from certified Projector")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    vocabulary_size = int(model.config.vocab_size)
    if tokenizer.vocab_size != vocabulary_size or len(tokenizer) != vocabulary_size:
        raise RuntimeError("tokenizer vocabulary differs from certified model")
    dataset = CertificateCaptionDataset(
        parquet_path=ROOT / "dataset/raw/pretrain_i2t.parquet",
        records=records,
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt=registry["data"]["prompt"],
        image_token_length=registry["data"]["image_token_length"],
        max_sequence_length=registry["data"]["max_sequence_length"],
    )
    contract = {
        "schema_version": 1,
        "status": "fixed_locked_test_evaluation_after_audited_unlock",
        "unlock_receipt_sha256": sha256_file(unlock_receipt_path),
        "test_manifest_sha256": actual_test_manifest_hash,
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "test_cluster_count": len(test_hashes),
        "caption_count": len(records),
        "caption_records_sha256": record_hash,
        "quantized_manifest_sha256": sha256_file(
            quantized_directory / "manifest.json"
        ),
        "quantized_hypothesis_sha256": quantized_manifest["hashes"][
            "quantized_hypothesis_sha256"
        ],
        "materialized_projector_parameters_sha256": projector_hash,
        "certificate_manifest_sha256": sha256_file(certificate_manifest_path),
        "certificate_bound_bits_per_token": certificate_bound,
        "fixed_alpha_numerator": selected_alpha["numerator"],
        "fixed_alpha_denominator_bits": selected_alpha["denominator_bits"],
        "fixed_alpha": alpha,
        "vocabulary_size": vocabulary_size,
        "eta": eta,
        "model_count": model_count,
        "evaluator_version": LOCKED_TEST_EVALUATOR_VERSION,
        "dataset_version": CERTIFICATE_DATASET_VERSION,
        "risk_version": RISK_IMPLEMENTATION_VERSION,
        "interval_version": TEST_INTERVAL_VERSION,
        "evaluator_source_sha256": sha256_file(evaluator_path),
        "statistics_source_sha256": sha256_file(statistics_source_path),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "forward_dtype": "bfloat16_autocast",
        "smoothing_and_statistics_dtype": "numpy_float64_log_domain",
        "alpha_search_performed_on_test": False,
    }
    work_directory.mkdir(parents=True, exist_ok=True)
    chunks_directory = work_directory / "chunks"
    chunks_directory.mkdir(exist_ok=True)
    contract_path = work_directory / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("existing locked-test work contract mismatch")
    else:
        atomic_json(contract_path, contract)

    random.seed(20260817)
    np.random.seed(20260817)
    torch.manual_seed(20260817)
    torch.cuda.manual_seed_all(20260817)
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
        generator=torch.Generator().manual_seed(20260817),
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model.to(device).eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("test model is not fully in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("test model contains trainable parameters")

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
                raise RuntimeError("resumed test chunk cluster mapping mismatch")
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
                raise RuntimeError("test shifted labels disagree with canonical encoder")
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
            chunk = load_chunk(chunk_path, positions)
        completed_captions += len(positions)
        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            print(
                json.dumps(
                    {
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

    all_positions: list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_logp: list[np.ndarray] = []
    for batch_index in range(len(loader)):
        start = batch_index * args.batch_size
        stop = min(start + args.batch_size, len(records))
        chunk = load_chunk(
            chunks_directory / f"batch_{batch_index:06d}.npz",
            np.arange(start, stop, dtype=np.int64),
        )
        all_positions.append(chunk["caption_positions"])
        all_clusters.append(chunk["cluster_positions"])
        all_counts.append(chunk["token_counts"])
        all_logp.append(chunk["correct_token_log_probabilities"])
    caption_positions = np.concatenate(all_positions)
    caption_clusters = np.concatenate(all_clusters)
    token_counts = np.concatenate(all_counts)
    logp = np.concatenate(all_logp)
    if not np.array_equal(caption_positions, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("test caption positions are not contiguous")
    if not np.array_equal(
        caption_clusters,
        np.asarray([record.cluster_position for record in records], dtype=np.uint32),
    ):
        raise RuntimeError("test caption cluster mapping mismatch")
    offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(token_counts, dtype=np.int64)]
    )
    if offsets[-1] != len(logp):
        raise RuntimeError("test token offsets mismatch log probabilities")

    token_losses = smoothed_token_losses_bits(
        logp, alpha=alpha, vocabulary_size=vocabulary_size
    )
    caption_losses = np.add.reduceat(token_losses, offsets[:-1]) / token_counts
    captions_per_cluster = np.bincount(
        caption_clusters.astype(np.int64), minlength=len(test_hashes)
    )
    if np.any(captions_per_cluster == 0):
        raise RuntimeError("a test cluster has no caption")
    cluster_losses = np.bincount(
        caption_clusters.astype(np.int64),
        weights=caption_losses,
        minlength=len(test_hashes),
    ) / captions_per_cluster
    empirical_test_risk = float(cluster_losses.mean())
    loss_lower, loss_upper, loss_width = loss_interval_bits(
        alpha=alpha, vocabulary_size=vocabulary_size
    )
    if np.any(cluster_losses < loss_lower - 2e-12) or np.any(
        cluster_losses > loss_upper + 2e-12
    ):
        raise RuntimeError("a test cluster loss lies outside the analytic range")
    inference = test_risk_interval_and_classification(
        empirical_test_risk=empirical_test_risk,
        loss_lower=loss_lower,
        loss_upper=loss_upper,
        test_cluster_count=len(test_hashes),
        model_count=model_count,
        eta=eta,
        certificate_bound=certificate_bound,
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".locked_test_evaluation.", dir=output_directory.parent)
    )
    try:
        arrays = {
            "correct_token_log_probabilities": logp.astype(np.float32, copy=False),
            "caption_offsets": offsets,
            "caption_cluster_positions": caption_clusters,
            "fixed_alpha_caption_losses": caption_losses.astype(np.float64),
            "fixed_alpha_cluster_losses": cluster_losses.astype(np.float64),
        }
        file_hashes: dict[str, str] = {}
        for name, array in arrays.items():
            path = temporary / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            file_hashes[f"{name}_sha256"] = sha256_file(path)
        records_path = temporary / "caption_records.jsonl"
        with records_path.open("x", encoding="utf-8", newline="\n") as output:
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
        file_hashes["caption_records_sha256"] = sha256_file(records_path)
        result_manifest = {
            "schema_version": 1,
            "status": "locked_test_evaluation_final",
            "evaluation_contract": contract,
            "evaluation_summary": {
                "test_cluster_count": len(test_hashes),
                "caption_count": len(records),
                "correct_token_count": len(logp),
                "minimum_cluster_loss_bits_per_token": float(cluster_losses.min()),
                "maximum_cluster_loss_bits_per_token": float(cluster_losses.max()),
            },
            "fixed_alpha": {
                "numerator": selected_alpha["numerator"],
                "denominator_bits": selected_alpha["denominator_bits"],
                "denominator": 1 << selected_alpha["denominator_bits"],
                "value": alpha,
                "selected_before_test_unlock": True,
            },
            "test_result": {
                "empirical_test_risk_bits_per_token": empirical_test_risk,
                "loss_lower_bits_per_token": loss_lower,
                "loss_upper_bits_per_token": loss_upper,
                "loss_width_bits_per_token": loss_width,
                "certificate_bound_bits_per_token": certificate_bound,
                **inference,
            },
            "joint_failure_probability_upper_bound": certificate_manifest[
                "evaluation_contract"
            ]["delta_model"]
            + certificate_manifest["evaluation_contract"]["delta_subsample"]
            + eta,
            "files": file_hashes,
            "test_manifest_first_read_after_unlock_receipt": True,
            "alpha_search_performed_on_test": False,
            "model_updated_after_test_unlock": False,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(result_manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.rename(temporary, output_directory)
        for artifact in output_directory.iterdir():
            artifact.chmod(0o444)
        output_directory.chmod(0o555)
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
                "output_directory": str(output_directory),
                "test_clusters": len(test_hashes),
                "test_risk_bits_per_token": empirical_test_risk,
                "confidence_interval": [
                    inference["confidence_interval_lower_bits_per_token"],
                    inference["confidence_interval_upper_bits_per_token"],
                ],
                "certificate_bound_bits_per_token": certificate_bound,
                "coverage_classification": inference["coverage_classification"],
                "alpha": alpha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
