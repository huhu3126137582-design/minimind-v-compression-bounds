from __future__ import annotations

import math

import numpy as np
import pytest

from minimind_v_bound.compression.description_length import (
    description_length_upper_bound,
)


def test_description_length_charges_every_preregistered_component() -> None:
    result = description_length_upper_bound(
        np.array([1, 1], dtype=np.uint32), dimension=2
    )

    assert result["empirical_symbol_entropy_bits_per_symbol"] == 1.0
    assert result["symbol_entropy_code_upper_bits"] == 3
    assert result["codebook_bits"] == 32
    assert result["count_bits_per_symbol"] == 2
    assert result["count_bits"] == 4
    assert result["structure_choice_bits"] == 0
    assert result["C_VLM_raw_upper_bits"] == 39
    assert result["prefix_overhead_upper_bits"] == 11
    assert result["K_VLM_upper_bits"] == 50
    assert math.isclose(sum(result["probabilities"]), 1.0)


def test_zero_probability_symbols_do_not_create_nan_entropy() -> None:
    result = description_length_upper_bound(
        np.array([4, 0], dtype=np.uint32), dimension=4
    )

    assert result["empirical_symbol_entropy_bits_per_symbol"] == 0.0
    assert result["symbol_entropy_code_upper_bits"] == 1
    assert result["K_VLM_upper_bits"] == 50


def test_description_length_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="sum"):
        description_length_upper_bound(
            np.array([2, 1], dtype=np.uint32), dimension=4
        )

