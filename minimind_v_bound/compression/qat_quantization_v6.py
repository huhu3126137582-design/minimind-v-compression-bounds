from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .quantization import QuantizedVector, quantize_equal_width_bin_means


QAT_QUANTIZATION_VERSION = "coordinate-ste-plus-learned-centers-v1"


def initial_qat_centers(vector: Tensor, *, levels: int) -> Tensor:
    if vector.ndim != 1 or vector.dtype != torch.float32:
        raise ValueError("QAT coordinate must be one-dimensional FP32")
    quantized = quantize_equal_width_bin_means(vector, levels=levels)
    return torch.from_numpy(quantized.centers.astype(np.float32)).to(vector.device)


def nearest_center_assignments(vector: Tensor, centers: Tensor) -> Tensor:
    if vector.ndim != 1 or centers.ndim != 1 or centers.numel() < 2:
        raise ValueError("QAT vector and centers must be non-empty one-dimensional tensors")
    if vector.dtype != torch.float32 or centers.dtype != torch.float32:
        raise ValueError("QAT assignments require FP32 vector and centers")
    if not torch.isfinite(vector).all() or not torch.isfinite(centers).all():
        raise ValueError("QAT vector and centers must be finite")
    return torch.argmin(torch.abs(vector[:, None] - centers[None, :]), dim=1)


def qat_ste_coordinate(vector: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    assignments = nearest_center_assignments(vector, centers)
    selected = centers[assignments]
    # Forward equals selected centers. Backward is identity for the coordinate
    # and gathered-gradient accumulation for the selected learned centers.
    quantized = selected + (vector - vector.detach())
    return quantized, assignments


def finalize_qat_quantization(
    vector: Tensor | np.ndarray, learned_centers: Tensor | np.ndarray
) -> QuantizedVector:
    if torch.is_tensor(vector):
        vector = vector.detach().cpu().numpy()
    if torch.is_tensor(learned_centers):
        learned_centers = learned_centers.detach().cpu().numpy()
    values = np.asarray(vector)
    centers32 = np.asarray(learned_centers)
    if values.ndim != 1 or values.dtype != np.float32 or values.size == 0:
        raise ValueError("final QAT coordinate must be non-empty FP32")
    if centers32.ndim != 1 or centers32.dtype != np.float32 or centers32.size < 2:
        raise ValueError("final learned centers must be one-dimensional FP32")
    if not np.isfinite(values).all() or not np.isfinite(centers32).all():
        raise ValueError("cannot finalize non-finite QAT state")
    centers = centers32.astype(np.float16)
    if not np.isfinite(centers).all():
        raise ValueError("FP16 QAT center conversion overflowed")
    values64 = values.astype(np.float64)
    distances = np.abs(values64[:, None] - centers.astype(np.float64)[None, :])
    assignments = np.argmin(distances, axis=1).astype(np.uint16)
    counts = np.bincount(assignments, minlength=len(centers)).astype(np.uint32)
    reconstructed = centers[assignments].astype(np.float32)
    result = QuantizedVector(
        centers=np.ascontiguousarray(centers),
        assignments=np.ascontiguousarray(assignments),
        counts=np.ascontiguousarray(counts),
        reconstructed=np.ascontiguousarray(reconstructed),
        initial_assignments=np.ascontiguousarray(assignments.copy()),
    )
    result.validate(dimension=len(values), levels=len(centers))
    return result
