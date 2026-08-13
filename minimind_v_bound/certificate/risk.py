from __future__ import annotations

import math

import numpy as np


RISK_IMPLEMENTATION_VERSION = "cluster-caption-token-fp64-logdomain-v1"
ALPHA_GRID_VERSION = "all-reduced-dyadic-b1-through-b16-v1"


def alpha_code_length_bits(denominator_bits: int) -> int:
    if denominator_bits < 1:
        raise ValueError("denominator_bits must be positive")
    return denominator_bits + 2 * math.floor(math.log2(denominator_bits))


def reduced_dyadic_grid(max_denominator_bits: int = 16) -> dict[str, np.ndarray]:
    if max_denominator_bits < 1 or max_denominator_bits > 30:
        raise ValueError("max_denominator_bits must lie in [1, 30]")
    numerators: list[int] = []
    denominator_bits: list[int] = []
    code_lengths: list[int] = []
    for bits in range(1, max_denominator_bits + 1):
        values = range(1, 1 << bits, 2)
        numerators.extend(values)
        denominator_bits.extend([bits] * (1 << (bits - 1)))
        code_lengths.extend(
            [alpha_code_length_bits(bits)] * (1 << (bits - 1))
        )
    numerator_array = np.asarray(numerators, dtype=np.uint32)
    bits_array = np.asarray(denominator_bits, dtype=np.uint8)
    alpha = numerator_array.astype(np.float64) / np.exp2(bits_array.astype(np.float64))
    return {
        "numerator": numerator_array,
        "denominator_bits": bits_array,
        "alpha": alpha,
        "code_length_bits": np.asarray(code_lengths, dtype=np.uint16),
    }


def smoothed_token_losses_bits(
    correct_token_log_probabilities: np.ndarray,
    *,
    alpha: float,
    vocabulary_size: int,
) -> np.ndarray:
    log_probabilities = np.asarray(correct_token_log_probabilities, dtype=np.float64)
    if log_probabilities.ndim != 1 or np.any(log_probabilities > 0.0):
        raise ValueError("correct-token log probabilities must be a 1D array <= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must exceed one")
    smoothed_log_probability = np.logaddexp(
        math.log1p(-alpha) + log_probabilities,
        math.log(alpha) - math.log(vocabulary_size),
    )
    return -smoothed_log_probability / math.log(2.0)


def hierarchical_token_weights(
    *,
    caption_offsets: np.ndarray,
    caption_cluster_positions: np.ndarray,
    cluster_multiplicities: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(caption_offsets)
    cluster_positions = np.asarray(caption_cluster_positions)
    multiplicities = np.asarray(cluster_multiplicities)
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError("caption_offsets must start at zero and contain an endpoint")
    token_counts = np.diff(offsets.astype(np.int64))
    caption_count = len(token_counts)
    if np.any(token_counts <= 0):
        raise ValueError("every caption must contain at least one evaluated token")
    if cluster_positions.shape != (caption_count,):
        raise ValueError("caption cluster mapping has the wrong shape")
    if multiplicities.ndim != 1 or np.any(multiplicities <= 0):
        raise ValueError("every represented cluster must have positive multiplicity")
    if np.any(cluster_positions < 0) or np.any(cluster_positions >= len(multiplicities)):
        raise ValueError("caption cluster position is out of range")
    captions_per_cluster = np.bincount(
        cluster_positions.astype(np.int64), minlength=len(multiplicities)
    )
    if np.any(captions_per_cluster == 0):
        raise ValueError("every represented cluster must contain a caption")
    draw_count = int(multiplicities.astype(np.int64).sum())
    caption_weights = (
        multiplicities[cluster_positions].astype(np.float64)
        / float(draw_count)
        / captions_per_cluster[cluster_positions].astype(np.float64)
    )
    token_weights = np.repeat(caption_weights / token_counts, token_counts)
    if token_weights.shape != (int(offsets[-1]),):
        raise RuntimeError("constructed token weights have the wrong shape")
    if not math.isclose(float(token_weights.sum()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise RuntimeError("hierarchical token weights do not sum to one")
    return token_weights


def loss_interval_bits(*, alpha: float, vocabulary_size: int) -> tuple[float, float, float]:
    if not 0.0 < alpha < 1.0 or vocabulary_size <= 1:
        raise ValueError("invalid alpha or vocabulary size")
    lower = -math.log2(1.0 - alpha + alpha / vocabulary_size)
    upper = math.log2(vocabulary_size / alpha)
    width = math.log2(1.0 + (1.0 - alpha) * vocabulary_size / alpha)
    if not math.isclose(upper - lower, width, rel_tol=2e-15, abs_tol=2e-15):
        raise RuntimeError("loss interval width identity failed")
    return lower, upper, width


def certificate_components_with_vocabulary(
    *,
    empirical_risk_bits: float,
    alpha: float,
    alpha_code_length_bits_value: int,
    model_description_bits: int,
    population_size: int,
    certificate_sample_size: int,
    delta_model: float,
    delta_subsample: float,
    vocabulary_size: int,
    model_count: int = 1,
) -> dict[str, float | int]:
    if not math.isfinite(empirical_risk_bits):
        raise ValueError("empirical risk must be finite")
    if min(
        alpha_code_length_bits_value,
        model_description_bits,
        population_size,
        certificate_sample_size,
        model_count,
    ) <= 0:
        raise ValueError("lengths, sizes, and model count must be positive")
    if not 0.0 < delta_model < 1.0 or not 0.0 < delta_subsample < 1.0:
        raise ValueError("failure probabilities must lie in (0, 1)")
    lower, upper, width = loss_interval_bits(
        alpha=alpha, vocabulary_size=vocabulary_size
    )
    model_term = width * math.sqrt(
        (
            (model_description_bits + alpha_code_length_bits_value) * math.log(2.0)
            + math.log(1.0 / delta_model)
        )
        / (2.0 * population_size)
    )
    subsample_term = width * math.sqrt(
        (
            alpha_code_length_bits_value * math.log(2.0)
            + math.log(model_count / delta_subsample)
        )
        / (2.0 * certificate_sample_size)
    )
    return {
        "empirical_risk_bits_per_token": empirical_risk_bits,
        "loss_lower_bits_per_token": lower,
        "loss_upper_bits_per_token": upper,
        "loss_width_bits_per_token": width,
        "alpha_code_length_bits": alpha_code_length_bits_value,
        "joint_description_length_bits": model_description_bits
        + alpha_code_length_bits_value,
        "model_complexity_term_bits_per_token": model_term,
        "subsample_complexity_term_bits_per_token": subsample_term,
        "certificate_bound_bits_per_token": empirical_risk_bits
        + model_term
        + subsample_term,
    }
