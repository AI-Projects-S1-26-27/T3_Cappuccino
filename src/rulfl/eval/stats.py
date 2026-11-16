"""E7: paired bootstrap intervals and exact one-sided signed-rank tests.

All computations use NumPy only. Differences are (comparison - FedAvg),
so a negative difference indicates an improvement in MAE.
"""
from __future__ import annotations

import numpy as np

# Differences are measured in RUL cycles. Before ranking, normalise harmless
# binary floating-point noise; differences under ~5e-11 become zero and values
# equal to 10 decimal places receive the same rank. This is an implementation
# convention, not an additional requirement written into E7.
_WILCOXON_DECIMALS = 10
_MAX_EXACT_PAIRS = 20  # Computational safeguard: enumeration requires 2**n cases.


def _vector(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Expected a nonempty finite 1-D vector")
    return arr


def paired_diff_bootstrap_ci(a, b, n_boot=10000, seed=0):
    """Return (mean(a-b), 2.5th, 97.5th percentile) from paired resampling.

    An entire (a_i, b_i) pair is selected together for each bootstrap draw.
    This is a two-sided 95% percentile interval, not a BCa or normal interval.
    """
    a, b = _vector(a), _vector(b)
    if a.shape != b.shape or isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("Arrays must match in length; n_boot must be positive integer")
    d = a - b
    if not np.all(np.isfinite(d)):
        raise ValueError("Differences must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        means[k] = np.mean(d[rng.integers(d.size, size=d.size)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(d)), float(lo), float(hi)


def wilcoxon_signed_rank_exact_one_sided(diffs, alternative="less"):
    """Exact sign-flip Wilcoxon p-value, zero_method='wilcox'.

    Rounds RUL-cycle differences to 10 decimal places *before* dropping zero
    differences and assigning midranks to tied absolute differences. Uses all
    2**n sign patterns. 'less' means comparison MAE is lower than FedAvg.
    A maximum of 20 input pairs bounds cost; the project uses 5 or 15 pairs.
    """
    d = _vector(diffs)
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be 'less' or 'greater'")
    if d.size > _MAX_EXACT_PAIRS:
        raise ValueError("Exact enumeration supports at most 20 input pairs")

    d = np.round(d, decimals=_WILCOXON_DECIMALS)
    d = d[d != 0]
    if d.size == 0:
        return 1.0

    magnitudes = np.abs(d)
    order = np.argsort(magnitudes, kind="stable")
    # Doubled ranks are exact integers even with averaged ties (.5 values),
    # preventing floating-point noise in the tail probability comparison.
    twice_ranks = np.empty(d.size, dtype=np.int64)
    i = 0
    while i < d.size:
        j = i + 1
        while j < d.size and magnitudes[order[j]] == magnitudes[order[i]]:
            j += 1
        twice_ranks[order[i:j]] = (i + 1 + j)
        i = j

    observed = int(twice_ranks[d > 0].sum())
    distribution = np.array([0], dtype=np.int64)
    for twice_rank in twice_ranks:
        distribution = np.concatenate((distribution, distribution + twice_rank))
    if alternative == "less":
        return float(np.mean(distribution <= observed))
    return float(np.mean(distribution >= observed))
