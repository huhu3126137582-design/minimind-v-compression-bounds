from __future__ import annotations

import math

import pytest

from minimind_v_bound.evaluation.test_statistics import (
    test_risk_interval_and_classification as compute_test_result,
)


def test_two_sided_hoeffding_radius_and_clipping() -> None:
    result = compute_test_result(
        empirical_test_risk=1.0,
        loss_lower=0.0,
        loss_upper=2.0,
        test_cluster_count=100,
        model_count=1,
        eta=0.05,
        certificate_bound=2.0,
    )
    expected_radius = 2.0 * math.sqrt(math.log(40.0) / 200.0)

    assert math.isclose(result["hoeffding_radius_bits_per_token"], expected_radius)
    assert result["confidence_interval_lower_bits_per_token"] == max(
        0.0, 1.0 - expected_radius
    )
    assert result["confidence_interval_upper_bits_per_token"] == min(
        2.0, 1.0 + expected_radius
    )
    assert result["coverage_classification"] == "strong_support"


@pytest.mark.parametrize(
    ("certificate", "classification"),
    [
        (2.0, "strong_support"),
        (1.0, "compatible_but_uncertain"),
        (-1.0, "statistically_significant_violation_signal"),
    ],
)
def test_coverage_classification_is_fixed(
    certificate: float, classification: str
) -> None:
    result = compute_test_result(
        empirical_test_risk=1.0,
        loss_lower=0.0,
        loss_upper=2.0,
        test_cluster_count=100,
        model_count=1,
        eta=0.05,
        certificate_bound=certificate,
    )
    assert result["coverage_classification"] == classification


def test_test_interval_rejects_risk_outside_loss_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        compute_test_result(
            empirical_test_risk=3.0,
            loss_lower=0.0,
            loss_upper=2.0,
            test_cluster_count=100,
            model_count=1,
            eta=0.05,
            certificate_bound=2.0,
        )
