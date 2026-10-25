"""Uncertainty and significance helpers. Every reported rate should come with an interval."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Paired comparison of two systems on the same items (exact binomial McNemar test)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"n_pairs": len(a), "only_a": only_a, "only_b": only_b, "p_value": float(p)}


def paired_bootstrap_diff(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    if len(d) == 0:
        return {"mean_diff": math.nan, "lo": math.nan, "hi": math.nan}
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"mean_diff": float(d.mean()), "lo": lo, "hi": hi}
