"""Label-skew partitioner: which client owns which engine, and how skewed the result is (DECISIONS PT1-PT11).

Rule (proposal): sort engines by lifetime, cut into K blocks; an engine in block b goes to client b with
probability H, otherwise to a uniformly random client. So P(client k) = H * 1[b = k] + (1 - H) / K:
H = 0 is an IID split, H = 1 gives each client one band of lifetimes, i.e. one slice of the RUL range.
"""
from __future__ import annotations

import numpy as np

_VAL_STREAM = 1          # second entry of the SeedSequence that routes validation engines (PT9)


def assign_clients(lifetimes: dict[int, int], K: int, H: float, seed: int) -> dict[int, int]:
    """Engine id -> client 0..K-1 (PT2, PT3, PT8).

    Engines are sorted by (lifetime, engine id) and cut into K contiguous blocks as equal as possible
    (199 -> 40, 40, 40, 40, 39). In that sorted order two vectors are drawn from default_rng(seed): u (uniform in
    [0, 1)) and then alt (a uniformly random client). An engine goes to its own block's client if u < H, else to alt.
    Both are drawn for every engine whatever H is, so one seed fixes one pair of vectors and H only moves the
    threshold: partitions at different H share their randomness, and H = 1 is exactly the blocks.
    """
    if not lifetimes:
        raise ValueError("no engines to assign")
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")
    if not 0.0 <= H <= 1.0:                                       # also rejects NaN
        raise ValueError(f"H must be in [0, 1], got {H}")
    n = len(lifetimes)
    if K > n:
        raise ValueError(f"K = {K} clients but only {n} engines")

    order = sorted((int(e) for e in lifetimes), key=lambda e: (int(lifetimes[e]), e))
    block = np.repeat(np.arange(K), [len(b) for b in np.array_split(np.arange(n), K)])
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    alt = rng.integers(0, K, size=n)
    client = np.where(u < H, block, alt)
    return {e: int(c) for e, c in sorted(zip(order, client))}


def val_seed(seed: int) -> int:
    """Seed for routing validation engines (PT9). Derived from the run seed alone, but a different stream from the
    training draws: with the raw seed, the i-th sorted training engine and the i-th sorted validation engine would
    get the same coin flips."""
    return int(np.random.SeedSequence([int(seed), _VAL_STREAM]).generate_state(1)[0])


def check_min_engines(assignment: dict[int, int], K: int, minimum: int) -> None:
    """PT4, PT10: every client needs at least `minimum` engines. Raises, never redraws."""
    counts = np.bincount(np.fromiter(assignment.values(), dtype=np.int64), minlength=K)[:K]
    if counts.min() < minimum:
        raise ValueError(f"client engine counts {counts.tolist()}: client {int(counts.argmin())} has fewer than "
                         f"{minimum} (PT4). Not redrawn: this partition is reported, not hidden")


def _whole_number_labels(y, what: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"{what}: expected a non-empty 1-D array of labels")
    if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.rint(y)):
        raise ValueError(f"{what}: labels must be whole numbers >= 0 (RUL in cycles)")
    return y.astype(np.int64)


def skew(labels_by_client: list[np.ndarray], all_labels: np.ndarray) -> float:
    """Mean over clients of the Wasserstein-1 distance, in cycles, between that client's training labels and all
    training labels (PT6, PT11).

    Labels are whole numbers, so W1 = sum over v of |CDF_client(v) - CDF_all(v)| exactly (unit grid).
    """
    if len(labels_by_client) == 0:
        raise ValueError("no clients")
    all_y = _whole_number_labels(all_labels, "all_labels")
    clients = [_whole_number_labels(y, f"client {k}") for k, y in enumerate(labels_by_client)]
    top = max(int(all_y.max()), *(int(c.max()) for c in clients)) + 1
    cdf = lambda y: np.cumsum(np.bincount(y, minlength=top)) / y.size
    cdf_all = cdf(all_y)
    return float(np.mean([np.abs(cdf(c) - cdf_all).sum() for c in clients]))
