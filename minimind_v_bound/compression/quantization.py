from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


QUANTIZATION_ALGORITHM_VERSION = "equal-width-bin-means-fp16-nearest-v1"


@dataclass(frozen=True)
class QuantizedVector:
    centers: np.ndarray
    assignments: np.ndarray
    counts: np.ndarray
    reconstructed: np.ndarray
    initial_assignments: np.ndarray

    def validate(self, *, dimension: int, levels: int) -> None:
        if self.centers.shape != (levels,) or self.centers.dtype != np.float16:
            raise ValueError("centers must contain exactly Q FP16 values")
        if self.assignments.shape != (dimension,) or self.assignments.dtype != np.uint16:
            raise ValueError("assignments must contain exactly d uint16 symbols")
        if self.counts.shape != (levels,) or self.counts.dtype != np.uint32:
            raise ValueError("counts must contain exactly Q uint32 values")
        if self.reconstructed.shape != (dimension,) or self.reconstructed.dtype != np.float32:
            raise ValueError("reconstruction must contain exactly d FP32 values")
        if not np.isfinite(self.centers).all() or not np.isfinite(self.reconstructed).all():
            raise ValueError("quantization produced non-finite values")
        if np.any(self.assignments >= levels):
            raise ValueError("assignment lies outside the codebook")
        expected_counts = np.bincount(self.assignments, minlength=levels).astype(np.uint32)
        if not np.array_equal(self.counts, expected_counts):
            raise ValueError("counts disagree with assignments")
        if int(self.counts.sum()) != dimension:
            raise ValueError("counts do not sum to d")
        expected = self.centers[self.assignments].astype(np.float32)
        if not np.array_equal(self.reconstructed, expected):
            raise ValueError("reconstruction is not exactly centers[assignments]")


def _to_numpy_fp32(vector: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(vector):
        vector = vector.detach().cpu().numpy()
    result = np.asarray(vector)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("quantized vector must be a non-empty one-dimensional array")
    if result.dtype != np.float32:
        raise ValueError("the preregistered input coordinate must be FP32")
    if not np.isfinite(result).all():
        raise ValueError("cannot quantize a non-finite coordinate")
    return np.ascontiguousarray(result)


def quantize_equal_width_bin_means(
    vector: torch.Tensor | np.ndarray,
    *,
    levels: int,
    edge_epsilon: float = 1e-6,
) -> QuantizedVector:
    """Apply the preregistered deterministic scalar quantizer.

    The initial Q equal-width bins cover ``[-max_abs-eps, max_abs+eps]``.
    Non-empty centers are the FP64-computed mean of their FP32 members; an
    empty-bin center is its left edge.  Centers are cast to FP16, after which
    every coordinate is reassigned to its nearest stored center. ``argmin``
    supplies the fixed lowest-index tie break.
    """
    values = _to_numpy_fp32(vector)
    if levels < 2 or levels > np.iinfo(np.uint16).max:
        raise ValueError("levels must lie in [2, 65535]")
    if edge_epsilon <= 0:
        raise ValueError("edge_epsilon must be positive")

    values64 = values.astype(np.float64)
    max_abs = max(float(values64.max()), abs(float(values64.min())))
    edges = np.linspace(
        -max_abs - edge_epsilon,
        max_abs + edge_epsilon,
        levels + 1,
        dtype=np.float64,
    )
    initial = np.searchsorted(edges, values64, side="right") - 1
    initial = np.clip(initial, 0, levels - 1).astype(np.uint16)
    centers64 = np.empty(levels, dtype=np.float64)
    for symbol in range(levels):
        members = values64[initial == symbol]
        centers64[symbol] = members.mean() if members.size else edges[symbol]
    centers = centers64.astype(np.float16)
    if not np.isfinite(centers).all():
        raise ValueError("FP16 center conversion overflowed")

    # Computing distance in FP64 makes the tie rule and assignments independent
    # of vectorized FP16 arithmetic choices. np.argmin selects the first index.
    distances = np.abs(values64[:, None] - centers.astype(np.float64)[None, :])
    assignments = np.argmin(distances, axis=1).astype(np.uint16)
    counts = np.bincount(assignments, minlength=levels).astype(np.uint32)
    reconstructed = centers[assignments].astype(np.float32)
    result = QuantizedVector(
        centers=np.ascontiguousarray(centers),
        assignments=np.ascontiguousarray(assignments),
        counts=np.ascontiguousarray(counts),
        reconstructed=np.ascontiguousarray(reconstructed),
        initial_assignments=np.ascontiguousarray(initial),
    )
    result.validate(dimension=len(values), levels=levels)
    return result

