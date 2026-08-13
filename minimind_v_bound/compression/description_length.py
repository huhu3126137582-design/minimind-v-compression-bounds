from __future__ import annotations

import math

import numpy as np


DESCRIPTION_LENGTH_VERSION = "empirical-arithmetic-upper-elias-gamma-v1"


def description_length_upper_bound(
    counts: np.ndarray,
    *,
    dimension: int,
    codebook_bits_per_center: int = 16,
    structure_choice_bits: int = 0,
) -> dict[str, int | float | list[float] | list[int]]:
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("counts must be a one-dimensional Q-vector")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("counts must be non-negative integers")
    counts64 = counts.astype(np.int64)
    if int(counts64.sum()) != dimension or dimension <= 0:
        raise ValueError("counts must sum to the positive intrinsic dimension")
    if codebook_bits_per_center <= 0 or structure_choice_bits < 0:
        raise ValueError("bit costs must be non-negative and codebook cost positive")

    probabilities = counts64.astype(np.float64) / float(dimension)
    positive = probabilities > 0
    entropy = -math.fsum(
        float(probability) * math.log2(float(probability))
        for probability in probabilities[positive]
    )
    entropy_product = dimension * entropy
    symbol_bits = math.ceil(entropy_product) + 1
    levels = len(counts64)
    codebook_bits = codebook_bits_per_center * levels
    count_bits_per_symbol = math.ceil(math.log2(dimension + 1))
    count_bits = levels * count_bits_per_symbol
    raw_bits = symbol_bits + codebook_bits + count_bits + structure_choice_bits
    if raw_bits <= 0:
        raise RuntimeError("raw description length must be positive")
    prefix_overhead = 2 * math.floor(math.log2(raw_bits)) + 1
    prefix_bits = raw_bits + prefix_overhead
    return {
        "dimension": dimension,
        "levels": levels,
        "counts": [int(value) for value in counts64],
        "probabilities": [float(value) for value in probabilities],
        "empirical_symbol_entropy_bits_per_symbol": entropy,
        "d_times_empirical_entropy_bits": entropy_product,
        "symbol_entropy_code_upper_bits": symbol_bits,
        "codebook_bits": codebook_bits,
        "count_bits_per_symbol": count_bits_per_symbol,
        "count_bits": count_bits,
        "structure_choice_bits": structure_choice_bits,
        "C_VLM_raw_upper_bits": raw_bits,
        "prefix_overhead_upper_bits": prefix_overhead,
        "K_VLM_upper_bits": prefix_bits,
    }

