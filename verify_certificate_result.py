from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from minimind_v_bound.certificate.search import search_reduced_dyadic_alpha
from minimind_v_bound.compression.artifact import sha256_file


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently recompute the frozen certificate result"
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_evaluation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/certificate_evaluation_verification.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_directory = args.result_directory.resolve()
    report_path = args.report.resolve()
    if "locked_test" in result_directory.parts or "locked_test" in report_path.parts:
        raise ValueError("certificate verification must not access the locked-test area")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    manifest_path = result_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "certificate_final_alpha_selected_before_locked_test":
        raise RuntimeError("certificate is not in its expected final pre-test state")
    if manifest["locked_test_accessed"]:
        raise RuntimeError("certificate manifest claims locked-test access")
    file_paths = {
        "correct_token_log_probabilities": result_directory
        / "correct_token_log_probabilities.npy",
        "caption_offsets": result_directory / "caption_offsets.npy",
        "caption_cluster_positions": result_directory
        / "caption_cluster_positions.npy",
        "selected_train_manifest_indices": result_directory
        / "selected_train_manifest_indices.npy",
        "cluster_multiplicities": result_directory / "cluster_multiplicities.npy",
        "selected_alpha_caption_losses": result_directory
        / "selected_alpha_caption_losses.npy",
        "selected_alpha_cluster_losses": result_directory
        / "selected_alpha_cluster_losses.npy",
        "caption_records": result_directory / "caption_records.jsonl",
    }
    for name, path in file_paths.items():
        if sha256_file(path) != manifest["files"][f"{name}_sha256"]:
            raise RuntimeError(f"certificate file hash mismatch: {name}")

    contract = manifest["evaluation_contract"]
    source_bindings = {
        "evaluator_source_sha256": ROOT / "evaluate_certificate_s4k.py",
        "dataset_source_sha256": ROOT / "minimind_v_bound/certificate/dataset.py",
        "risk_source_sha256": ROOT / "minimind_v_bound/certificate/risk.py",
        "alpha_search_source_sha256": ROOT / "minimind_v_bound/certificate/search.py",
        "sample_manifest_sha256": ROOT
        / "runs/s4k_formal_v1/certificate_sample_n10000/manifest.json",
        "quantized_manifest_sha256": ROOT
        / "runs/s4k_formal_v1/quantized_q11/manifest.json",
        "train_manifest_sha256": ROOT / "dataset/manifests/train_clusters.jsonl",
    }
    for key, path in source_bindings.items():
        if sha256_file(path) != contract[key]:
            raise RuntimeError(f"certificate contract source mismatch: {key}")

    logp = np.load(file_paths["correct_token_log_probabilities"], allow_pickle=False)
    offsets = np.load(file_paths["caption_offsets"], allow_pickle=False)
    caption_clusters = np.load(file_paths["caption_cluster_positions"], allow_pickle=False)
    selected_indices = np.load(
        file_paths["selected_train_manifest_indices"], allow_pickle=False
    )
    multiplicities = np.load(file_paths["cluster_multiplicities"], allow_pickle=False)
    stored_caption_losses = np.load(
        file_paths["selected_alpha_caption_losses"], allow_pickle=False
    )
    stored_cluster_losses = np.load(
        file_paths["selected_alpha_cluster_losses"], allow_pickle=False
    )
    caption_count = contract["caption_count"]
    cluster_count = contract["unique_cluster_count"]
    if logp.dtype != np.float32 or logp.ndim != 1 or np.any(logp > 0.0):
        raise RuntimeError("invalid stored correct-token log probabilities")
    if offsets.dtype != np.int64 or offsets.shape != (caption_count + 1,):
        raise RuntimeError("invalid caption offsets")
    if offsets[0] != 0 or offsets[-1] != len(logp) or np.any(np.diff(offsets) <= 0):
        raise RuntimeError("caption offsets are not a strict complete partition")
    if caption_clusters.dtype != np.uint32 or caption_clusters.shape != (caption_count,):
        raise RuntimeError("invalid caption-to-cluster mapping")
    if selected_indices.dtype != np.uint32 or selected_indices.shape != (cluster_count,):
        raise RuntimeError("invalid selected training indices")
    if not np.all(selected_indices[1:] > selected_indices[:-1]):
        raise RuntimeError("selected training indices are not canonical and unique")
    if multiplicities.dtype != np.uint32 or multiplicities.shape != (cluster_count,):
        raise RuntimeError("invalid cluster multiplicities")
    if int(multiplicities.astype(np.int64).sum()) != contract["sample_draw_count"]:
        raise RuntimeError("cluster multiplicities do not sum to the sample size")

    token_counts = np.diff(offsets)
    captions_per_cluster = np.bincount(
        caption_clusters.astype(np.int64), minlength=cluster_count
    )
    if np.any(captions_per_cluster == 0):
        raise RuntimeError("a selected cluster has no caption")
    caption_weights = (
        multiplicities[caption_clusters].astype(np.float64)
        / contract["sample_draw_count"]
        / captions_per_cluster[caption_clusters]
    )
    token_weights = np.repeat(caption_weights / token_counts, token_counts)
    if not math.isclose(float(token_weights.sum()), 1.0, abs_tol=2e-15):
        raise RuntimeError("independent hierarchical weights do not sum to one")

    selected = manifest["selected_alpha"]
    alpha = selected["numerator"] / float(1 << selected["denominator_bits"])
    if alpha != selected["value"] or selected["numerator"] % 2 != 1:
        raise RuntimeError("selected alpha is not the stored reduced dyadic fraction")
    vocabulary_size = contract["vocabulary_size"]
    logp64 = logp.astype(np.float64)
    token_losses = -np.logaddexp(
        math.log1p(-alpha) + logp64,
        math.log(alpha) - math.log(vocabulary_size),
    ) / math.log(2.0)
    caption_losses = np.add.reduceat(token_losses, offsets[:-1]) / token_counts
    cluster_losses = np.bincount(
        caption_clusters.astype(np.int64),
        weights=caption_losses,
        minlength=cluster_count,
    ) / captions_per_cluster
    if not np.allclose(caption_losses, stored_caption_losses, rtol=0.0, atol=2e-14):
        raise RuntimeError("independently recomputed caption losses mismatch")
    if not np.allclose(cluster_losses, stored_cluster_losses, rtol=0.0, atol=2e-14):
        raise RuntimeError("independently recomputed cluster losses mismatch")
    empirical = float(
        np.dot(multiplicities.astype(np.float64), cluster_losses)
        / contract["sample_draw_count"]
    )

    alpha_length = selected["denominator_bits"] + 2 * math.floor(
        math.log2(selected["denominator_bits"])
    )
    lower = -math.log2(1.0 - alpha + alpha / vocabulary_size)
    upper = math.log2(vocabulary_size / alpha)
    width = math.log2(1.0 + (1.0 - alpha) * vocabulary_size / alpha)
    model_term = width * math.sqrt(
        (
            (contract["K_VLM_upper_bits"] + alpha_length) * math.log(2.0)
            + math.log(1.0 / contract["delta_model"])
        )
        / (2.0 * contract["population_size"])
    )
    subsample_term = width * math.sqrt(
        (
            alpha_length * math.log(2.0)
            + math.log(contract["model_count"] / contract["delta_subsample"])
        )
        / (2.0 * contract["sample_draw_count"])
    )
    bound = empirical + model_term + subsample_term
    recomputed = {
        "empirical_risk_bits_per_token": empirical,
        "loss_lower_bits_per_token": lower,
        "loss_upper_bits_per_token": upper,
        "loss_width_bits_per_token": width,
        "alpha_code_length_bits": alpha_length,
        "joint_description_length_bits": contract["K_VLM_upper_bits"] + alpha_length,
        "model_complexity_term_bits_per_token": model_term,
        "subsample_complexity_term_bits_per_token": subsample_term,
        "certificate_bound_bits_per_token": bound,
    }
    for key, value in recomputed.items():
        stored = manifest["certificate"][key]
        if isinstance(value, int):
            if value != stored:
                raise RuntimeError(f"certificate integer component mismatch: {key}")
        elif not math.isclose(value, stored, rel_tol=0.0, abs_tol=2e-12):
            raise RuntimeError(f"certificate floating component mismatch: {key}")
    if np.any(cluster_losses < lower - 2e-12) or np.any(cluster_losses > upper + 2e-12):
        raise RuntimeError("a cluster loss lies outside the analytic loss interval")

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
    if (search.numerator, search.denominator_bits) != (
        selected["numerator"],
        selected["denominator_bits"],
    ):
        raise RuntimeError("replayed complete dyadic search selected a different alpha")
    if not math.isclose(
        float(search.certificate["certificate_bound_bits_per_token"]),
        bound,
        rel_tol=0.0,
        abs_tol=2e-12,
    ):
        raise RuntimeError("replayed alpha-search objective mismatch")

    uniform = math.log2(vocabulary_size)
    report = {
        "schema_version": 1,
        "status": "certificate_independently_recomputed_and_verified",
        "certificate_manifest_sha256": sha256_file(manifest_path),
        "selected_alpha_fraction": f"{selected['numerator']}/{1 << selected['denominator_bits']}",
        "empirical_risk_bits_per_token": empirical,
        "model_complexity_term_bits_per_token": model_term,
        "subsample_complexity_term_bits_per_token": subsample_term,
        "certificate_bound_bits_per_token": bound,
        "uniform_predictor_bits_per_token": uniform,
        "certificate_strictly_below_uniform": bound < uniform,
        "all_cluster_losses_in_analytic_interval": True,
        "complete_dyadic_search_replayed": True,
        "locked_test_accessed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    report_path.chmod(0o444)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
