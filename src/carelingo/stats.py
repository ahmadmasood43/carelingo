"""Confidence intervals for small-sample proportions."""

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; unlike the normal approximation it stays inside
    [0, 1] and gives a useful upper bound when 0 of n events are observed."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def proportion(successes: int, n: int) -> str:
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} ({successes / n:.0%}, CI {low:.0%}–{high:.0%})" if n else "0/0"
