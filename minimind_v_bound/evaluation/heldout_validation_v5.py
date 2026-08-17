from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from minimind_v_bound.certificate.risk import smoothed_token_losses_bits


HELDOUT_VALIDATION_METRICS_VERSION = (
    "v5-cluster-caption-token-raw-and-fixed-alpha-ranking-v1"
)
RANK_ALPHA = 1.0 / 8.0
EXPECTED_MODEL_COUNT = 7


def _validate_hierarchy(
    *,
    correct_token_log_probabilities: np.ndarray,
    caption_offsets: np.ndarray,
    caption_cluster_positions: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logp = np.asarray(correct_token_log_probabilities)
    offsets = np.asarray(caption_offsets)
    clusters = np.asarray(caption_cluster_positions)
    if logp.dtype != np.float32 or logp.ndim != 1:
        raise ValueError("correct-token log probabilities must be a float32 vector")
    if not np.isfinite(logp).all() or np.any(logp > 0.0):
        raise ValueError("correct-token log probabilities must be finite and <= 0")
    if offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("caption offsets must be a non-empty int64 vector")
    if offsets[0] != 0 or offsets[-1] != len(logp):
        raise ValueError("caption offsets do not partition the token probabilities")
    token_counts = np.diff(offsets)
    if np.any(token_counts <= 0):
        raise ValueError("every caption must contain an evaluated token")
    if clusters.dtype != np.uint32 or clusters.shape != (len(token_counts),):
        raise ValueError("caption-to-cluster mapping has the wrong dtype or shape")
    if cluster_count <= 0 or np.any(clusters >= cluster_count):
        raise ValueError("caption-to-cluster mapping contains an invalid position")
    captions_per_cluster = np.bincount(
        clusters.astype(np.int64), minlength=cluster_count
    )
    if np.any(captions_per_cluster == 0):
        raise ValueError("every held-out image cluster must contain a caption")
    return token_counts, clusters, captions_per_cluster


def caption_and_cluster_losses(
    *,
    correct_token_log_probabilities: np.ndarray,
    caption_offsets: np.ndarray,
    caption_cluster_positions: np.ndarray,
    cluster_count: int,
    vocabulary_size: int,
    alpha: float | None,
) -> dict[str, Any]:
    """Compute caption-equal, then cluster-equal BPD from one forward pass.

    ``alpha=None`` is the ordinary unsmoothed held-out BPD. A numeric alpha
    uses the same prediction smoothing implementation as the certificates.
    """

    logp = np.asarray(correct_token_log_probabilities)
    offsets = np.asarray(caption_offsets)
    token_counts, clusters, captions_per_cluster = _validate_hierarchy(
        correct_token_log_probabilities=logp,
        caption_offsets=offsets,
        caption_cluster_positions=caption_cluster_positions,
        cluster_count=cluster_count,
    )
    if vocabulary_size <= 1:
        raise ValueError("vocabulary size must exceed one")
    if alpha is None:
        token_losses = -logp.astype(np.float64) / math.log(2.0)
        metric = "raw_unsmoothed_bpd"
    else:
        token_losses = smoothed_token_losses_bits(
            logp, alpha=float(alpha), vocabulary_size=vocabulary_size
        )
        metric = "prediction_smoothed_bpd"
    caption_losses = np.add.reduceat(token_losses, offsets[:-1]) / token_counts
    cluster_losses = np.bincount(
        clusters.astype(np.int64),
        weights=caption_losses,
        minlength=cluster_count,
    ) / captions_per_cluster
    if not np.isfinite(caption_losses).all() or not np.isfinite(cluster_losses).all():
        raise RuntimeError("held-out losses must be finite")
    return {
        "metric": metric,
        "alpha": None if alpha is None else float(alpha),
        "caption_losses_bits_per_token": caption_losses,
        "cluster_losses_bits_per_token": cluster_losses,
        "risk_bits_per_token": float(cluster_losses.mean(dtype=np.float64)),
    }


def pairwise_concordance(
    predictor_values: np.ndarray, validation_values: np.ndarray
) -> dict[str, int | float]:
    predictors = np.asarray(predictor_values, dtype=np.float64)
    validation = np.asarray(validation_values, dtype=np.float64)
    if predictors.ndim != 1 or predictors.shape != validation.shape:
        raise ValueError("concordance inputs must be same-length vectors")
    concordant = discordant = tied = 0
    for left in range(len(predictors)):
        for right in range(left + 1, len(predictors)):
            predictor_sign = np.sign(predictors[left] - predictors[right])
            validation_sign = np.sign(validation[left] - validation[right])
            if predictor_sign == 0 or validation_sign == 0:
                tied += 1
            elif predictor_sign == validation_sign:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "comparable_pairs": comparable,
        "concordance_fraction": (
            concordant / comparable if comparable else math.nan
        ),
    }


def ranking_summary(
    predictor_values: np.ndarray, validation_values: np.ndarray
) -> dict[str, Any]:
    predictors = np.asarray(predictor_values, dtype=np.float64)
    validation = np.asarray(validation_values, dtype=np.float64)
    if predictors.shape != (EXPECTED_MODEL_COUNT,) or validation.shape != (
        EXPECTED_MODEL_COUNT,
    ):
        raise ValueError("the frozen trend analysis requires exactly seven models")
    if not np.isfinite(predictors).all() or not np.isfinite(validation).all():
        raise ValueError("ranking inputs must be finite")
    validation_ranks = rankdata(validation, method="average")
    minimum_index = int(np.argmin(predictors))
    return {
        "spearman_rho": float(spearmanr(predictors, validation).statistic),
        "kendall_tau_b": float(
            kendalltau(predictors, validation, variant="b").statistic
        ),
        "minimum_predictor_model_index": minimum_index,
        "minimum_predictor_validation_rank": float(validation_ranks[minimum_index]),
        "minimum_predictor_validation_rank_at_most_2": bool(
            validation_ranks[minimum_index] <= 2.0
        ),
        "pairwise": pairwise_concordance(predictors, validation),
    }


def analyze_fixed_seven_model_results(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(entries) != EXPECTED_MODEL_COUNT:
        raise ValueError("analysis requires exactly seven ordered entries")
    certificate = np.asarray(
        [entry["ranking_certificate_bits_per_token"] for entry in entries],
        dtype=np.float64,
    )
    certification_empirical = np.asarray(
        [entry["certification_empirical_risk_bits_per_token"] for entry in entries],
        dtype=np.float64,
    )
    validation = np.asarray(
        [entry["validation_rank_risk_bits_per_token"] for entry in entries],
        dtype=np.float64,
    )
    certificate_summary = ranking_summary(certificate, validation)
    empirical_summary = ranking_summary(certification_empirical, validation)
    pairwise_rows: list[dict[str, Any]] = []
    for left in range(EXPECTED_MODEL_COUNT):
        for right in range(left + 1, EXPECTED_MODEL_COUNT):
            certificate_sign = int(np.sign(certificate[left] - certificate[right]))
            validation_sign = int(np.sign(validation[left] - validation[right]))
            if certificate_sign == 0 or validation_sign == 0:
                classification = "tied"
            elif certificate_sign == validation_sign:
                classification = "concordant"
            else:
                classification = "discordant"
            pairwise_rows.append(
                {
                    "left_configuration_id": entries[left]["configuration_id"],
                    "right_configuration_id": entries[right]["configuration_id"],
                    "ranking_certificate_difference_bits_per_token": float(
                        certificate[left] - certificate[right]
                    ),
                    "validation_rank_risk_difference_bits_per_token": float(
                        validation[left] - validation[right]
                    ),
                    "classification": classification,
                }
            )
    if len(pairwise_rows) != 21:
        raise RuntimeError("the seven-model family must produce exactly 21 pairs")
    return {
        "schema_version": 1,
        "metrics_version": HELDOUT_VALIDATION_METRICS_VERSION,
        "model_count": EXPECTED_MODEL_COUNT,
        "rank_alpha": RANK_ALPHA,
        "candidate_order": [entry["configuration_id"] for entry in entries],
        "certificate_vs_validation": certificate_summary,
        "certification_empirical_vs_validation": empirical_summary,
        "certificate_spearman_minus_empirical_spearman": float(
            certificate_summary["spearman_rho"] - empirical_summary["spearman_rho"]
        ),
        "validation_certificate_risk_gap_bits_per_token": [
            float(validation[index] - certification_empirical[index])
            for index in range(EXPECTED_MODEL_COUNT)
        ],
        "ascending_ranking_certificate_order": [
            entries[index]["configuration_id"] for index in np.argsort(certificate)
        ],
        "ascending_validation_rank_risk_order": [
            entries[index]["configuration_id"] for index in np.argsort(validation)
        ],
        "pairwise_model_comparisons": pairwise_rows,
    }
