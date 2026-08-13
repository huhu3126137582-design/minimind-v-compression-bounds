from __future__ import annotations

import math


def minimind_cosine_learning_rate(
    step: int, total_steps: int, initial_learning_rate: float
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step <= total_steps:
        raise ValueError("step must lie in [0, total_steps]")
    return initial_learning_rate * (
        0.1 + 0.45 * (1.0 + math.cos(math.pi * step / total_steps))
    )
