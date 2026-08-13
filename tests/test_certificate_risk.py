from __future__ import annotations

import math

import numpy as np

from minimind_v_bound.certificate.risk import (
    alpha_code_length_bits,
    certificate_components_with_vocabulary,
    hierarchical_token_weights,
    loss_interval_bits,
    reduced_dyadic_grid,
    smoothed_token_losses_bits,
)


def test_smoothed_loss_matches_p_zero_one_and_uniform_analytics() -> None:
    vocabulary_size = 6400
    alpha = 0.125
    values = smoothed_token_losses_bits(
        np.array([-np.inf, 0.0, -math.log(vocabulary_size)], dtype=np.float64),
        alpha=alpha,
        vocabulary_size=vocabulary_size,
    )
    lower, upper, width = loss_interval_bits(
        alpha=alpha, vocabulary_size=vocabulary_size
    )

    assert math.isclose(values[0], upper, rel_tol=0.0, abs_tol=2e-14)
    assert math.isclose(values[1], lower, rel_tol=0.0, abs_tol=2e-14)
    assert math.isclose(values[2], math.log2(vocabulary_size), abs_tol=2e-14)
    assert math.isclose(upper - lower, width, abs_tol=2e-14)


def test_hierarchical_weights_encode_token_caption_cluster_order() -> None:
    # Cluster 0 has two captions with 2 and 1 tokens and is drawn twice.
    # Cluster 1 has one 1-token caption and is drawn once.
    weights = hierarchical_token_weights(
        caption_offsets=np.array([0, 2, 3, 4], dtype=np.int64),
        caption_cluster_positions=np.array([0, 0, 1], dtype=np.uint32),
        cluster_multiplicities=np.array([2, 1], dtype=np.uint32),
    )
    token_losses = np.array([1.0, 3.0, 5.0, 7.0])
    expected = (2.0 * ((1.0 + 3.0) / 2.0 + 5.0) / 2.0 + 7.0) / 3.0

    assert math.isclose(float(weights.sum()), 1.0, abs_tol=2e-15)
    assert math.isclose(float(np.dot(weights, token_losses)), expected, abs_tol=2e-15)


def test_reduced_dyadic_grid_is_complete_unique_and_charged() -> None:
    grid = reduced_dyadic_grid(16)

    assert len(grid["alpha"]) == 65535
    assert len(np.unique(grid["alpha"])) == 65535
    assert np.all(grid["numerator"] % 2 == 1)
    assert float(grid["alpha"].min()) == 1.0 / 65536.0
    assert float(grid["alpha"].max()) == 65535.0 / 65536.0
    assert alpha_code_length_bits(1) == 1
    assert alpha_code_length_bits(16) == 24
    assert grid["code_length_bits"][0] == 1
    assert grid["code_length_bits"][-1] == 24


def test_certificate_components_sum_exactly() -> None:
    result = certificate_components_with_vocabulary(
        empirical_risk_bits=4.0,
        alpha=1.0 / 16.0,
        alpha_code_length_bits_value=8,
        model_description_bits=11089,
        population_size=561359,
        certificate_sample_size=10000,
        delta_model=0.025,
        delta_subsample=0.025,
        vocabulary_size=6400,
        model_count=1,
    )

    assert result["joint_description_length_bits"] == 11097
    assert math.isclose(
        result["certificate_bound_bits_per_token"],
        result["empirical_risk_bits_per_token"]
        + result["model_complexity_term_bits_per_token"]
        + result["subsample_complexity_term_bits_per_token"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
