from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .risk import (
    alpha_code_length_bits,
    certificate_components_with_vocabulary,
    loss_interval_bits,
    smoothed_token_losses_bits,
)


ALPHA_SEARCH_ALGORITHM_VERSION = "convex-half-plus-monotone-branch-bound-v1"


@dataclass(frozen=True)
class AlphaSearchResult:
    numerator: int
    denominator_bits: int
    alpha: float
    alpha_code_length_bits: int
    empirical_risk_bits_per_token: float
    certificate: dict[str, float | int]
    exact_objective_evaluations: int
    derivative_evaluations: int
    branch_lower_bound_evaluations: int
    pruned_grid_candidates: int


class WeightedRiskEvaluator:
    def __init__(
        self,
        log_probabilities: np.ndarray,
        token_weights: np.ndarray,
        *,
        vocabulary_size: int,
    ) -> None:
        self.logp = np.asarray(log_probabilities, dtype=np.float64)
        self.weights = np.asarray(token_weights, dtype=np.float64)
        if self.logp.ndim != 1 or self.weights.shape != self.logp.shape:
            raise ValueError("log probabilities and weights must be aligned vectors")
        if np.any(self.logp > 0.0) or np.any(self.weights < 0.0):
            raise ValueError("invalid log probabilities or weights")
        if not math.isclose(float(self.weights.sum()), 1.0, abs_tol=2e-15):
            raise ValueError("token weights must sum to one")
        if vocabulary_size <= 1:
            raise ValueError("vocabulary size must exceed one")
        self.vocabulary_size = vocabulary_size
        self.probabilities = np.exp(self.logp)
        self.high_probability = self.logp >= -math.log(vocabulary_size)
        self._risk_cache: dict[float, float] = {}
        self._parts_cache: dict[float, tuple[float, float]] = {}
        self.exact_evaluations = 0
        self.derivative_evaluations = 0
        self.lower_bound_evaluations = 0

    def risk(self, alpha: float) -> float:
        key = float(alpha)
        if key not in self._risk_cache:
            losses = smoothed_token_losses_bits(
                self.logp, alpha=key, vocabulary_size=self.vocabulary_size
            )
            self._risk_cache[key] = float(np.dot(self.weights, losses))
            self.exact_evaluations += 1
        return self._risk_cache[key]

    def risk_parts(self, alpha: float) -> tuple[float, float]:
        key = float(alpha)
        if key not in self._parts_cache:
            losses = smoothed_token_losses_bits(
                self.logp, alpha=key, vocabulary_size=self.vocabulary_size
            )
            high = float(
                np.dot(self.weights[self.high_probability], losses[self.high_probability])
            )
            low = float(
                np.dot(self.weights[~self.high_probability], losses[~self.high_probability])
            )
            self._parts_cache[key] = (high, low)
            self.lower_bound_evaluations += 1
        return self._parts_cache[key]

    def derivative(self, alpha: float, complexity_coefficient: float) -> float:
        probability = self.probabilities
        slope = 1.0 / self.vocabulary_size - probability
        mixture = (1.0 - alpha) * probability + alpha / self.vocabulary_size
        risk_derivative = -float(np.dot(self.weights, slope / mixture)) / math.log(2.0)
        vocabulary_size = self.vocabulary_size
        delta_derivative = (
            -(vocabulary_size - 1.0)
            / (vocabulary_size - (vocabulary_size - 1.0) * alpha)
            - 1.0 / alpha
        ) / math.log(2.0)
        self.derivative_evaluations += 1
        return risk_derivative + complexity_coefficient * delta_derivative


def _complexity_coefficient(
    *,
    alpha_length: int,
    model_description_bits: int,
    population_size: int,
    sample_size: int,
    delta_model: float,
    delta_subsample: float,
    model_count: int,
) -> float:
    return math.sqrt(
        (
            (model_description_bits + alpha_length) * math.log(2.0)
            + math.log(1.0 / delta_model)
        )
        / (2.0 * population_size)
    ) + math.sqrt(
        (alpha_length * math.log(2.0) + math.log(model_count / delta_subsample))
        / (2.0 * sample_size)
    )


def search_reduced_dyadic_alpha(
    *,
    log_probabilities: np.ndarray,
    token_weights: np.ndarray,
    vocabulary_size: int,
    max_denominator_bits: int,
    model_description_bits: int,
    population_size: int,
    sample_size: int,
    delta_model: float,
    delta_subsample: float,
    model_count: int,
) -> AlphaSearchResult:
    if max_denominator_bits < 1 or max_denominator_bits > 20:
        raise ValueError("unsupported dyadic search depth")
    evaluator = WeightedRiskEvaluator(
        log_probabilities, token_weights, vocabulary_size=vocabulary_size
    )
    best_key: tuple[float, float, int, int] | None = None
    best_payload: tuple[int, int, float, int, dict[str, float | int]] | None = None
    evaluated_candidates: set[tuple[int, int]] = set()

    def evaluate_candidate(bits: int, numerator: int) -> None:
        nonlocal best_key, best_payload
        if numerator <= 0 or numerator >= (1 << bits) or numerator % 2 != 1:
            return
        candidate_id = (bits, numerator)
        if candidate_id in evaluated_candidates:
            return
        evaluated_candidates.add(candidate_id)
        alpha = numerator / float(1 << bits)
        alpha_length = alpha_code_length_bits(bits)
        empirical = evaluator.risk(alpha)
        certificate = certificate_components_with_vocabulary(
            empirical_risk_bits=empirical,
            alpha=alpha,
            alpha_code_length_bits_value=alpha_length,
            model_description_bits=model_description_bits,
            population_size=population_size,
            certificate_sample_size=sample_size,
            delta_model=delta_model,
            delta_subsample=delta_subsample,
            vocabulary_size=vocabulary_size,
            model_count=model_count,
        )
        key = (
            float(certificate["certificate_bound_bits_per_token"]),
            alpha,
            bits,
            numerator,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_payload = (bits, numerator, alpha, alpha_length, certificate)

    # On alpha <= 1/2, both the empirical smoothed loss and Delta_alpha are
    # convex. Locate the continuous minimum for each code-length class, then
    # check the adjacent odd dyadic numerators exactly.
    convex_pruned_candidates = 0
    for bits in range(1, max_denominator_bits + 1):
        denominator = 1 << bits
        maximum_numerator = 1 if bits == 1 else denominator // 2 - 1
        low = 1.0 / denominator
        high = maximum_numerator / denominator
        coefficient = _complexity_coefficient(
            alpha_length=alpha_code_length_bits(bits),
            model_description_bits=model_description_bits,
            population_size=population_size,
            sample_size=sample_size,
            delta_model=delta_model,
            delta_subsample=delta_subsample,
            model_count=model_count,
        )
        if low == high:
            optimum = low
        elif evaluator.derivative(low, coefficient) >= 0.0:
            optimum = low
        elif evaluator.derivative(high, coefficient) <= 0.0:
            optimum = high
        else:
            left, right = low, high
            for _ in range(60):
                middle = (left + right) / 2.0
                if evaluator.derivative(middle, coefficient) < 0.0:
                    left = middle
                else:
                    right = middle
            optimum = (left + right) / 2.0
        central = int(math.floor(optimum * denominator))
        candidates = {1, maximum_numerator}
        for value in range(central - 4, central + 6):
            if value % 2 == 1:
                candidates.add(value)
        evaluated_before = len(evaluated_candidates)
        for numerator in sorted(candidates):
            if 1 <= numerator <= maximum_numerator:
                evaluate_candidate(bits, numerator)
        lower_half_count = (maximum_numerator + 1) // 2
        convex_pruned_candidates += lower_half_count - (
            len(evaluated_candidates) - evaluated_before
        )

    if best_key is None:
        raise RuntimeError("alpha search produced no candidates")

    # Above 1/2 Delta is no longer convex. Branch-and-bound remains exact:
    # p>=1/V losses increase with alpha, p<1/V losses decrease, and Delta
    # decreases. Those three endpoint values give a valid interval lower bound.
    pruned_candidates = convex_pruned_candidates
    for bits in range(2, max_denominator_bits + 1):
        denominator = 1 << bits
        first = denominator // 2 + 1
        last = denominator - 1
        coefficient = _complexity_coefficient(
            alpha_length=alpha_code_length_bits(bits),
            model_description_bits=model_description_bits,
            population_size=population_size,
            sample_size=sample_size,
            delta_model=delta_model,
            delta_subsample=delta_subsample,
            model_count=model_count,
        )
        queue: list[tuple[float, int, int]] = []

        def lower_bound(low_numerator: int, high_numerator: int) -> float:
            low_alpha = low_numerator / denominator
            high_alpha = high_numerator / denominator
            high_part, _ = evaluator.risk_parts(low_alpha)
            _, low_part = evaluator.risk_parts(high_alpha)
            width = loss_interval_bits(
                alpha=high_alpha, vocabulary_size=vocabulary_size
            )[2]
            return high_part + low_part + coefficient * width

        heapq.heappush(queue, (lower_bound(first, last), first, last))
        while queue:
            bound, low_numerator, high_numerator = heapq.heappop(queue)
            count = (high_numerator - low_numerator) // 2 + 1
            assert best_key is not None
            # A small FP64 safety margin prevents a rounding-level overestimate
            # of the analytic lower bound from pruning a genuine near tie.
            if bound > best_key[0] + 1e-12:
                pruned_candidates += count
                continue
            if count <= 8:
                for numerator in range(low_numerator, high_numerator + 1, 2):
                    evaluate_candidate(bits, numerator)
                continue
            left_count = count // 2
            left_high = low_numerator + 2 * (left_count - 1)
            right_low = left_high + 2
            heapq.heappush(
                queue,
                (lower_bound(low_numerator, left_high), low_numerator, left_high),
            )
            heapq.heappush(
                queue,
                (lower_bound(right_low, high_numerator), right_low, high_numerator),
            )

    assert best_payload is not None
    bits, numerator, alpha, alpha_length, certificate = best_payload
    total_grid_size = (1 << max_denominator_bits) - 1
    if len(evaluated_candidates) + pruned_candidates != total_grid_size:
        raise RuntimeError(
            "alpha search accounting failed: "
            f"evaluated={len(evaluated_candidates)}, pruned={pruned_candidates}, "
            f"grid={total_grid_size}"
        )
    return AlphaSearchResult(
        numerator=numerator,
        denominator_bits=bits,
        alpha=alpha,
        alpha_code_length_bits=alpha_length,
        empirical_risk_bits_per_token=float(
            certificate["empirical_risk_bits_per_token"]
        ),
        certificate=certificate,
        exact_objective_evaluations=evaluator.exact_evaluations,
        derivative_evaluations=evaluator.derivative_evaluations,
        branch_lower_bound_evaluations=evaluator.lower_bound_evaluations,
        pruned_grid_candidates=pruned_candidates,
    )
