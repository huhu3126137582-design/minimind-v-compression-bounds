from __future__ import annotations

import numpy as np
import pytest
import torch

from minimind_v_bound.compression.quantization import (
    quantize_equal_width_bin_means,
)


def test_quantization_is_deterministic_and_reconstructs_from_stored_centers() -> None:
    coordinate = np.array(
        [-1.0, -0.51, -0.49, -0.01, 0.0, 0.01, 0.49, 0.51, 1.0],
        dtype=np.float32,
    )
    before = coordinate.copy()
    first = quantize_equal_width_bin_means(coordinate, levels=5)
    second = quantize_equal_width_bin_means(torch.from_numpy(coordinate), levels=5)

    assert np.array_equal(coordinate, before)
    assert first.centers.dtype == np.float16
    assert first.assignments.dtype == np.uint16
    assert first.counts.dtype == np.uint32
    assert first.reconstructed.dtype == np.float32
    assert np.array_equal(first.centers, second.centers)
    assert np.array_equal(first.assignments, second.assignments)
    assert np.array_equal(first.reconstructed, first.centers[first.assignments])
    assert int(first.counts.sum()) == coordinate.size


def test_reassignment_uses_lowest_center_index_for_exact_ties() -> None:
    # With all-zero input, every initial bin except the middle one is empty.
    # The first two stored FP16 centers coincide, so the nearest-center tie must
    # resolve to symbol zero via np.argmin's first-index rule.
    coordinate = np.zeros(4, dtype=np.float32)
    result = quantize_equal_width_bin_means(coordinate, levels=3, edge_epsilon=1e-9)

    assert result.centers[0] == result.centers[1]
    assert np.array_equal(result.assignments, np.zeros(4, dtype=np.uint16))
    assert result.counts.tolist() == [4, 0, 0]


@pytest.mark.parametrize(
    "coordinate",
    [np.array([1.0], dtype=np.float64), np.array([[1.0]], dtype=np.float32)],
)
def test_quantizer_rejects_non_preregistered_input_shape_or_dtype(
    coordinate: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        quantize_equal_width_bin_means(coordinate, levels=11)
