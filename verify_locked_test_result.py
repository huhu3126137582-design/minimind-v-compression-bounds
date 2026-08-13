from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from minimind_v_bound.certificate.sampling import read_cluster_hashes
from minimind_v_bound.compression.artifact import sha256_file


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently recompute and verify the final locked-test result"
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/locked_test_evaluation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs/s4k_formal_v1/locked_test_evaluation_verification_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_directory = args.result_directory.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    manifest_path = result_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "locked_test_evaluation_final":
        raise RuntimeError("locked-test result is not final")
    if (
        manifest["alpha_search_performed_on_test"]
        or manifest["model_updated_after_test_unlock"]
    ):
        raise RuntimeError("locked-test result reports a prohibited post-unlock action")

    paths = {
        "correct_token_log_probabilities": result_directory
        / "correct_token_log_probabilities.npy",
        "caption_offsets": result_directory / "caption_offsets.npy",
        "caption_cluster_positions": result_directory
        / "caption_cluster_positions.npy",
        "fixed_alpha_caption_losses": result_directory
        / "fixed_alpha_caption_losses.npy",
        "fixed_alpha_cluster_losses": result_directory
        / "fixed_alpha_cluster_losses.npy",
        "caption_records": result_directory / "caption_records.jsonl",
    }
    for name, path in paths.items():
        if sha256_file(path) != manifest["files"][f"{name}_sha256"]:
            raise RuntimeError(f"locked-test artifact hash mismatch: {name}")

    contract = manifest["evaluation_contract"]
    bound_inputs = {
        "unlock_receipt_sha256": ROOT
        / "runs/s4k_formal_v1/locked_test_unlock_receipt.json",
        "certificate_manifest_sha256": ROOT
        / "runs/s4k_formal_v1/certificate_evaluation/manifest.json",
        "quantized_manifest_sha256": ROOT
        / "runs/s4k_formal_v1/quantized_q11/manifest.json",
        "train_manifest_sha256": ROOT / "dataset/manifests/train_clusters.jsonl",
        "test_manifest_sha256": ROOT / "dataset/locked_test/test_clusters.jsonl",
        "evaluator_source_sha256": ROOT / "evaluate_locked_test_s4k.py",
        "statistics_source_sha256": ROOT
        / "minimind_v_bound/evaluation/test_statistics.py",
    }
    for key, path in bound_inputs.items():
        if sha256_file(path) != contract[key]:
            raise RuntimeError(f"locked-test contract binding mismatch: {key}")

    test_hashes = read_cluster_hashes(
        ROOT / "dataset/locked_test/test_clusters.jsonl"
    )
    train_hashes = read_cluster_hashes(ROOT / "dataset/manifests/train_clusters.jsonl")
    if len(test_hashes) != contract["test_cluster_count"]:
        raise RuntimeError("test manifest count mismatch")
    if set(test_hashes).intersection(train_hashes):
        raise RuntimeError("train/test cluster overlap found during final verification")

    logp = np.load(paths["correct_token_log_probabilities"], allow_pickle=False)
    offsets = np.load(paths["caption_offsets"], allow_pickle=False)
    caption_clusters = np.load(paths["caption_cluster_positions"], allow_pickle=False)
    stored_caption_losses = np.load(paths["fixed_alpha_caption_losses"], allow_pickle=False)
    stored_cluster_losses = np.load(paths["fixed_alpha_cluster_losses"], allow_pickle=False)
    caption_count = contract["caption_count"]
    cluster_count = contract["test_cluster_count"]
    if logp.dtype != np.float32 or logp.ndim != 1:
        raise RuntimeError("stored test log probabilities have invalid dtype or rank")
    if not np.isfinite(logp).all() or np.any(logp > 0.0):
        raise RuntimeError("stored test log probabilities are invalid")
    if offsets.dtype != np.int64 or offsets.shape != (caption_count + 1,):
        raise RuntimeError("stored test caption offsets have invalid dtype or shape")
    token_counts = np.diff(offsets)
    if offsets[0] != 0 or offsets[-1] != len(logp) or np.any(token_counts <= 0):
        raise RuntimeError("stored test caption offsets are not a strict partition")
    if caption_clusters.dtype != np.uint32 or caption_clusters.shape != (caption_count,):
        raise RuntimeError("stored test caption-cluster mapping is invalid")
    if np.any(caption_clusters >= cluster_count):
        raise RuntimeError("stored test caption cluster position is out of range")

    record_count = 0
    with paths["caption_records"].open("r", encoding="utf-8") as source:
        for caption_position, line in enumerate(source):
            if caption_position >= caption_count:
                raise RuntimeError("too many locked-test caption records")
            record = json.loads(line)
            cluster_position = int(caption_clusters[caption_position])
            if record["caption_position"] != caption_position:
                raise RuntimeError("locked-test caption record position mismatch")
            if record["cluster_position"] != cluster_position:
                raise RuntimeError("locked-test caption record cluster position mismatch")
            if record["cluster_sha256"] != test_hashes[cluster_position]:
                raise RuntimeError("locked-test caption record cluster hash mismatch")
            if record["valid_token_count"] != int(token_counts[caption_position]):
                raise RuntimeError("locked-test caption record token count mismatch")
            record_count += 1
    if record_count != caption_count:
        raise RuntimeError("too few locked-test caption records")

    numerator = contract["fixed_alpha_numerator"]
    denominator_bits = contract["fixed_alpha_denominator_bits"]
    alpha = numerator / float(1 << denominator_bits)
    if alpha != contract["fixed_alpha"] or (numerator, denominator_bits) != (1, 3):
        raise RuntimeError("locked-test alpha is not the preselected 1/8")
    vocabulary_size = contract["vocabulary_size"]
    token_losses = -np.logaddexp(
        math.log1p(-alpha) + logp.astype(np.float64),
        math.log(alpha) - math.log(vocabulary_size),
    ) / math.log(2.0)
    caption_losses = np.add.reduceat(token_losses, offsets[:-1]) / token_counts
    captions_per_cluster = np.bincount(
        caption_clusters.astype(np.int64), minlength=cluster_count
    )
    if np.any(captions_per_cluster == 0):
        raise RuntimeError("a locked-test cluster has no caption")
    cluster_losses = np.bincount(
        caption_clusters.astype(np.int64),
        weights=caption_losses,
        minlength=cluster_count,
    ) / captions_per_cluster
    if not np.allclose(caption_losses, stored_caption_losses, rtol=0.0, atol=2e-14):
        raise RuntimeError("independently recomputed test caption losses mismatch")
    if not np.allclose(cluster_losses, stored_cluster_losses, rtol=0.0, atol=2e-14):
        raise RuntimeError("independently recomputed test cluster losses mismatch")
    empirical = float(cluster_losses.mean())

    lower_loss = -math.log2(1.0 - alpha + alpha / vocabulary_size)
    upper_loss = math.log2(vocabulary_size / alpha)
    width = math.log2(1.0 + (1.0 - alpha) * vocabulary_size / alpha)
    if np.any(cluster_losses < lower_loss - 2e-12) or np.any(
        cluster_losses > upper_loss + 2e-12
    ):
        raise RuntimeError("a test cluster loss is outside the analytic interval")
    radius = width * math.sqrt(
        math.log(2.0 * contract["model_count"] / contract["eta"])
        / (2.0 * cluster_count)
    )
    interval_lower = max(lower_loss, empirical - radius)
    interval_upper = min(upper_loss, empirical + radius)
    certificate_bound = contract["certificate_bound_bits_per_token"]
    if interval_upper <= certificate_bound:
        classification = "strong_support"
    elif interval_lower <= certificate_bound < interval_upper:
        classification = "compatible_but_uncertain"
    else:
        classification = "statistically_significant_violation_signal"

    recomputed = {
        "empirical_test_risk_bits_per_token": empirical,
        "loss_lower_bits_per_token": lower_loss,
        "loss_upper_bits_per_token": upper_loss,
        "loss_width_bits_per_token": width,
        "certificate_bound_bits_per_token": certificate_bound,
        "hoeffding_radius_bits_per_token": radius,
        "confidence_interval_lower_bits_per_token": interval_lower,
        "confidence_interval_upper_bits_per_token": interval_upper,
    }
    for key, value in recomputed.items():
        if not math.isclose(
            value, manifest["test_result"][key], rel_tol=0.0, abs_tol=2e-12
        ):
            raise RuntimeError(f"locked-test statistic mismatch: {key}")
    if classification != manifest["test_result"]["coverage_classification"]:
        raise RuntimeError("locked-test coverage classification mismatch")

    certificate_manifest = json.loads(
        (ROOT / "runs/s4k_formal_v1/certificate_evaluation/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    certificate_empirical = certificate_manifest["certificate"][
        "empirical_risk_bits_per_token"
    ]
    report = {
        "schema_version": 1,
        "status": "locked_test_result_independently_recomputed_and_verified",
        "locked_test_result_manifest_sha256": sha256_file(manifest_path),
        "test_cluster_manifest_sha256": sha256_file(
            ROOT / "dataset/locked_test/test_clusters.jsonl"
        ),
        "test_cluster_count": cluster_count,
        "caption_count": caption_count,
        "correct_token_count": len(logp),
        "fixed_alpha_fraction": "1/8",
        "empirical_test_risk_bits_per_token": empirical,
        "test_minus_certificate_empirical_bits_per_token": empirical
        - certificate_empirical,
        "hoeffding_radius_bits_per_token": radius,
        "confidence_interval_lower_bits_per_token": interval_lower,
        "confidence_interval_upper_bits_per_token": interval_upper,
        "certificate_bound_bits_per_token": certificate_bound,
        "certificate_margin_above_test_interval_upper": certificate_bound
        - interval_upper,
        "coverage_classification": classification,
        "all_cluster_losses_in_analytic_interval": True,
        "train_test_cluster_intersection": 0,
        "alpha_search_performed_on_test": False,
        "model_updated_after_test_unlock": False,
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
