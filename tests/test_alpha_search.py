from __future__ import annotations

import math

import numpy as np

from minimind_v_bound.certificate.risk import (
    alpha_code_length_bits,
    certificate_components_with_vocabulary,
    reduced_dyadic_grid,
    smoothed_token_losses_bits,
)
from minimind_v_bound.certificate.search import search_reduced_dyadic_alpha


def test_exact_search_matches_exhaustive_grid() -> None:
    rng = np.random.default_rng(1234)
    probabilities = rng.beta(0.7, 5.0, size=80)
    logp = np.log(probabilities)
    weights = rng.random(len(logp))
    weights /= weights.sum()
    arguments = {
        "vocabulary_size": 97,
        "model_description_bits": 130,
        "population_size": 2000,
        "sample_size": 400,
        "delta_model": 0.025,
        "delta_subsample": 0.025,
        "model_count": 1,
    }
    result = search_reduced_dyadic_alpha(
        log_probabilities=logp,
        token_weights=weights,
        max_denominator_bits=8,
        **arguments,
    )
    grid = reduced_dyadic_grid(8)
    exhaustive: list[tuple[float, float, int, int]] = []
    for numerator, bits, alpha in zip(
        grid["numerator"], grid["denominator_bits"], grid["alpha"], strict=True
    ):
        empirical = float(
            np.dot(
                weights,
                smoothed_token_losses_bits(
                    logp, alpha=float(alpha), vocabulary_size=arguments["vocabulary_size"]
                ),
            )
        )
        certificate = certificate_components_with_vocabulary(
            empirical_risk_bits=empirical,
            alpha=float(alpha),
            alpha_code_length_bits_value=alpha_code_length_bits(int(bits)),
            model_description_bits=arguments["model_description_bits"],
            population_size=arguments["population_size"],
            certificate_sample_size=arguments["sample_size"],
            delta_model=arguments["delta_model"],
            delta_subsample=arguments["delta_subsample"],
            vocabulary_size=arguments["vocabulary_size"],
            model_count=arguments["model_count"],
        )
        exhaustive.append(
            (
                float(certificate["certificate_bound_bits_per_token"]),
                float(alpha),
                int(bits),
                int(numerator),
            )
        )
    expected = min(exhaustive)

    assert result.denominator_bits == expected[2]
    assert result.numerator == expected[3]
    assert math.isclose(
        result.certificate["certificate_bound_bits_per_token"], expected[0], abs_tol=1e-14
    )
    assert result.exact_objective_evaluations + result.pruned_grid_candidates == 255
