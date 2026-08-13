from __future__ import annotations

import math


TEST_INTERVAL_VERSION = "two-sided-hoeffding-bonferroni-clipped-v1"


def test_risk_interval_and_classification(
    *,
    empirical_test_risk: float,
    loss_lower: float,
    loss_upper: float,
    test_cluster_count: int,
    model_count: int,
    eta: float,
    certificate_bound: float,
) -> dict[str, float | str]:
    values = (empirical_test_risk, loss_lower, loss_upper, certificate_bound)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("risks and bounds must be finite")
    if loss_lower > empirical_test_risk or empirical_test_risk > loss_upper:
        raise ValueError("empirical test risk lies outside its analytic range")
    if test_cluster_count <= 0 or model_count <= 0:
        raise ValueError("cluster and model counts must be positive")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0, 1)")
    width = loss_upper - loss_lower
    radius = width * math.sqrt(
        math.log(2.0 * model_count / eta) / (2.0 * test_cluster_count)
    )
    interval_lower = max(loss_lower, empirical_test_risk - radius)
    interval_upper = min(loss_upper, empirical_test_risk + radius)
    if interval_upper <= certificate_bound:
        classification = "strong_support"
    elif interval_lower <= certificate_bound < interval_upper:
        classification = "compatible_but_uncertain"
    else:
        classification = "statistically_significant_violation_signal"
    return {
        "hoeffding_radius_bits_per_token": radius,
        "confidence_interval_lower_bits_per_token": interval_lower,
        "confidence_interval_upper_bits_per_token": interval_upper,
        "coverage_classification": classification,
    }

