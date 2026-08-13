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
from minimind_v_bound.certificate.search import (  # noqa: E402
    ALPHA_SEARCH_ALGORITHM_VERSION,
    search_reduced_dyadic_alpha,
)
from minimind_v_bound.certificate.sampling import read_cluster_hashes  # noqa: E402
from minimind_v_bound.compression.artifact import (  # noqa: E402
    canonical_compressed_model_sha256,
    canonical_named_tensor_sha256,
    load_and_verify_quantized_arrays,
    sha256_file,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
)


EVALUATOR_VERSION = "s4k-certificate-correct-token-logp-bf16-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the formal frozen S-4K certificate")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_sample_n10000",
    )
    parser.add_argument(
        "--quantized-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/quantized_q11",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_evaluation_work",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_evaluation",
    )
    return parser.parse_args()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def records_sha256(records) -> str:
    digest = hashlib.sha256()
    digest.update(b"minimind-v-bound-certificate-caption-records-v1")
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
    required = {
        "caption_positions",
        "cluster_positions",
        "token_counts",
        "correct_token_log_probabilities",
    }
    if set(chunk) != required:
        raise RuntimeError(f"chunk has unexpected fields: {path}")
    if not np.array_equal(chunk["caption_positions"], expected_positions):
        raise RuntimeError(f"chunk caption positions mismatch: {path}")
    counts = chunk["token_counts"]
    logp = chunk["correct_token_log_probabilities"]
    if counts.dtype != np.uint16 or np.any(counts == 0):
        raise RuntimeError(f"chunk contains invalid token counts: {path}")
    if logp.dtype != np.float32 or len(logp) != int(counts.astype(np.int64).sum()):
        raise RuntimeError(f"chunk log probabilities mismatch token counts: {path}")
    if not np.isfinite(logp).all() or np.any(logp > 0.0):
        raise RuntimeError(f"chunk contains invalid log probabilities: {path}")
    return chunk


def main() -> None:
    args = parse_args()
    if args.batch_size != 32:
        raise ValueError("formal certificate evaluation batch size is fixed at 32")
    if args.num_workers != 4:
        raise ValueError("formal certificate evaluation worker count is fixed at 4")
    paths = [
        args.sample_directory.resolve(),
        args.quantized_directory.resolve(),
        args.work_directory.resolve(),
        args.output_directory.resolve(),
    ]
    sample_directory, quantized_directory, work_directory, output_directory = paths
    if any("locked_test" in path.parts for path in paths):
        raise ValueError("certificate evaluation must not access the locked-test area")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    registry_path = ROOT / "configs/experiment_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    frozen_manifest = json.loads(
        (ROOT / "configs/frozen_manifest.json").read_text(encoding="utf-8")
    )
    raw_dataset_path = ROOT / registry["data"]["local_path"]
    raw_dataset_hash = sha256_file(raw_dataset_path)
    if raw_dataset_hash != frozen_manifest["dataset_sha256"]:
        raise RuntimeError("raw dataset no longer matches the frozen manifest")
    llm_path = ROOT / registry["frozen_base"]["language"]["local_path"]
    if sha256_file(llm_path) != frozen_manifest["llm_checkpoint_sha256"]:
        raise RuntimeError("LLM checkpoint no longer matches the frozen manifest")
    vision_path = ROOT / registry["frozen_base"]["vision"]["local_path"]
    frozen_vision_files = {
        "model.safetensors": "vision_checkpoint_sha256",
        "config.json": "vision_config_sha256",
        "preprocessor_config.json": "vision_processor_config_sha256",
    }
    for relative_path, manifest_key in frozen_vision_files.items():
        if sha256_file(vision_path / relative_path) != frozen_manifest[manifest_key]:
            raise RuntimeError(f"vision artifact no longer matches: {relative_path}")
    tokenizer_path = ROOT / registry["frozen_base"]["tokenizer"]["local_path"]
    if sha256_named_files(
        tokenizer_path, ["tokenizer.json", "tokenizer_config.json"]
    ) != frozen_manifest["tokenizer_sha256"]:
        raise RuntimeError("tokenizer no longer matches the frozen manifest")
    sample_manifest = json.loads(
        (sample_directory / "manifest.json").read_text(encoding="utf-8")
    )
    if sample_manifest["status"] != "certificate_sample_frozen_before_loss_evaluation":
        raise RuntimeError("certificate sample is not frozen for evaluation")
    indices_path = sample_directory / "sample_indices.npy"
    if sha256_file(indices_path) != sample_manifest["hashes"]["sample_indices_sha256"]:
        raise RuntimeError("certificate sample index hash mismatch")
    sample_indices = np.load(indices_path, allow_pickle=False)
    if sample_indices.dtype != np.uint32 or sample_indices.shape != (10000,):
        raise RuntimeError("certificate sample has the wrong dtype or shape")

    quantized_manifest, quantized = load_and_verify_quantized_arrays(quantized_directory)
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=quantized_manifest["compressed_model_descriptor"],
    )
    descriptor = sample_manifest["sampling_descriptor"]
    if compressed_hash != descriptor["compressed_model_sha256"]:
        raise RuntimeError("certificate sample is not bound to this compressed model")
    if sha256_file(quantized_directory / "manifest.json") != descriptor[
        "quantized_manifest_sha256"
    ]:
        raise RuntimeError("certificate sample quantized-manifest binding failed")

    train_manifest_path = ROOT / "dataset/manifests/train_clusters.jsonl"
    train_hashes = read_cluster_hashes(train_manifest_path)
    if len(train_hashes) != descriptor["population_size"]:
        raise RuntimeError("training population size mismatch")
    if sha256_file(train_manifest_path) != descriptor["train_manifest_sha256"]:
        raise RuntimeError("training manifest hash mismatch")
    selected_indices, multiplicities = np.unique(sample_indices, return_counts=True)
    selected_indices = selected_indices.astype(np.uint32)
    multiplicities = multiplicities.astype(np.uint32)
    selected_hashes = [train_hashes[int(index)] for index in selected_indices]
    records = build_certificate_caption_records(
        database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        selected_cluster_hashes=selected_hashes,
    )
    record_hash = records_sha256(records)

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
        raise RuntimeError("pinned LLM checkpoint does not match the architecture")
    wrapper = prepare_intrinsic_projector(model, intrinsic_dimension=4096, seed=137)
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.from_numpy(quantized.reconstructed))
    projector_hash = canonical_named_tensor_sha256(
        list(wrapper.materialized_parameters().items())
    )
    if projector_hash != quantized_manifest["hashes"]["materialized_projector_parameters_sha256"]:
        raise RuntimeError("materialized quantized Projector hash mismatch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    vocabulary_size = int(model.config.vocab_size)
    if tokenizer.vocab_size != vocabulary_size or len(tokenizer) != vocabulary_size:
        raise RuntimeError("tokenizer and model vocabulary sizes disagree")
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
        "status": "fixed_before_certificate_forward_evaluation",
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
        "caption_records_sha256": record_hash,
        "raw_dataset_sha256": raw_dataset_hash,
        "cluster_database_sha256": sha256_file(
            ROOT / "dataset/clusters/image_clusters.sqlite3"
        ),
        "train_manifest_sha256": descriptor["train_manifest_sha256"],
        "sample_manifest_sha256": sha256_file(sample_directory / "manifest.json"),
        "quantized_manifest_sha256": descriptor["quantized_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "quantized_hypothesis_sha256": descriptor["quantized_hypothesis_sha256"],
        "materialized_projector_parameters_sha256": projector_hash,
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "dataset_source_sha256": sha256_file(
            ROOT / "minimind_v_bound/certificate/dataset.py"
        ),
        "risk_source_sha256": sha256_file(
            ROOT / "minimind_v_bound/certificate/risk.py"
        ),
        "alpha_search_source_sha256": sha256_file(
            ROOT / "minimind_v_bound/certificate/search.py"
        ),
        "K_VLM_upper_bits": descriptor["K_VLM_upper_bits"],
        "population_size": descriptor["population_size"],
        "delta_model": descriptor["delta_model"],
        "delta_subsample": descriptor["delta_subsample"],
        "model_count": descriptor["model_count"],
        "max_denominator_bits": registry["certificate"]["alpha_search"][
            "max_denominator_bits"
        ],
        "locked_test_accessed": False,
    }
    work_directory.mkdir(parents=True, exist_ok=True)
    chunks_directory = work_directory / "chunks"
    chunks_directory.mkdir(exist_ok=True)
    contract_path = work_directory / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("existing evaluation work contract does not match")
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
            pixel_values = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["pixel_values"].items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = model(
                    input_ids=input_ids,
                    labels=None,
                    pixel_values=pixel_values,
                )
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            token_counts = valid.sum(dim=1).cpu().numpy().astype(np.uint16)
            if not np.array_equal(
                token_counts.astype(np.int64), batch["valid_label_count"].numpy()
            ):
                raise RuntimeError("shifted label counts disagree with canonical encoder")
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
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "captions": completed_captions,
                        "caption_total": len(dataset),
                        "elapsed_seconds": round(elapsed, 3),
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
        raise RuntimeError("final caption positions are not contiguous")
    expected_clusters = np.asarray(
        [record.cluster_position for record in records], dtype=np.uint32
    )
    if not np.array_equal(caption_clusters, expected_clusters):
        raise RuntimeError("final caption cluster mapping mismatch")
    caption_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(token_counts, dtype=np.int64)]
    )
    if int(caption_offsets[-1]) != len(logp):
        raise RuntimeError("final token offsets disagree with log probabilities")
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
        model_description_bits=descriptor["K_VLM_upper_bits"],
        population_size=descriptor["population_size"],
        sample_size=len(sample_indices),
        delta_model=descriptor["delta_model"],
        delta_subsample=descriptor["delta_subsample"],
        model_count=descriptor["model_count"],
    )

    selected_token_losses = smoothed_token_losses_bits(
        logp, alpha=search.alpha, vocabulary_size=vocabulary_size
    )
    caption_losses = np.add.reduceat(selected_token_losses, caption_offsets[:-1]) / token_counts
    caption_counts_per_cluster = np.bincount(
        caption_clusters.astype(np.int64), minlength=len(selected_indices)
    )
    cluster_loss_sums = np.bincount(
        caption_clusters.astype(np.int64),
        weights=caption_losses,
        minlength=len(selected_indices),
    )
    cluster_losses = cluster_loss_sums / caption_counts_per_cluster
    hierarchical_empirical = float(
        np.dot(multiplicities.astype(np.float64), cluster_losses) / len(sample_indices)
    )
    if not math.isclose(
        hierarchical_empirical,
        search.empirical_risk_bits_per_token,
        rel_tol=0.0,
        abs_tol=2e-12,
    ):
        raise RuntimeError("hierarchical and token-weight empirical risks disagree")
    lower = float(search.certificate["loss_lower_bits_per_token"])
    upper = float(search.certificate["loss_upper_bits_per_token"])
    if np.any(cluster_losses < lower - 2e-12) or np.any(cluster_losses > upper + 2e-12):
        raise RuntimeError("a certificate cluster loss lies outside its analytic range")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".certificate_evaluation.", dir=output_directory.parent)
    )
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
            "status": "certificate_final_alpha_selected_before_locked_test",
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
                "certificate_strictly_below_uniform": float(
                    search.certificate["certificate_bound_bits_per_token"]
                )
                < math.log2(vocabulary_size),
            },
            "files": file_hashes,
            "locked_test_accessed": False,
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
                "alpha": search.alpha,
                "alpha_fraction": f"{search.numerator}/{1 << search.denominator_bits}",
                "empirical_risk_bits_per_token": search.empirical_risk_bits_per_token,
                "model_complexity_term_bits_per_token": search.certificate[
                    "model_complexity_term_bits_per_token"
                ],
                "subsample_complexity_term_bits_per_token": search.certificate[
                    "subsample_complexity_term_bits_per_token"
                ],
                "certificate_bound_bits_per_token": search.certificate[
                    "certificate_bound_bits_per_token"
                ],
                "uniform_predictor_bits_per_token": math.log2(vocabulary_size),
                "non_vacuous": result_manifest["non_vacuity"][
                    "certificate_strictly_below_uniform"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
