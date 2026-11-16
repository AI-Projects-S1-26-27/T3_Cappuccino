"""Partitioner: label-skew client assignment, skew metric and manifests (DECISIONS PT1-PT12)."""
import json
from pathlib import Path

import numpy as np
import pytest

from rulfl.partition.label_skew import assign_clients, check_min_engines, skew, val_seed
from rulfl.partition.make_manifests import (build_manifest, load_manifest, make_all, manifest_hash,
                                            manifest_path, write_manifest)

REPO = Path(__file__).resolve().parents[1]


def window_labels(lifetime: int, window: int = 30, cap: int = 125) -> np.ndarray:
    """Training labels of one engine, as the pipeline makes them (D6): the window ending at cycle t = window..T
    has label min(T - t, cap). Checked against the real pipeline in test_window_labels_match_the_pipeline."""
    return np.minimum(np.arange(lifetime - window, -1, -1), cap)


def fd004_like_lifetimes(n=199, seed=0):
    """Hand-made lifetimes shaped like FD004's training engines (min 128, long right tail)."""
    rng = np.random.default_rng(seed)
    return {e: int(128 + rng.gamma(2.0, 60.0)) for e in range(1, n + 1)}


def client_labels(assignment, lifetimes, K):
    return [np.concatenate([window_labels(lifetimes[e]) for e, c in assignment.items() if c == k]) for k in range(K)]


# ---------- assign_clients (PT2, PT3, PT8) ----------

def test_h1_gives_exactly_the_lifetime_blocks():
    """TEAM_TASKS first test. 199 engines -> blocks 40, 40, 40, 40, 39 (PT2); shortest-lived go to client 0."""
    life = fd004_like_lifetimes()
    a = assign_clients(life, K=5, H=1.0, seed=1)
    assert np.bincount(list(a.values())).tolist() == [40, 40, 40, 40, 39]
    for k in range(4):
        assert max(life[e] for e, c in a.items() if c == k) <= min(life[e] for e, c in a.items() if c == k + 1)
    assert assign_clients(life, 5, 1.0, seed=2) == a                     # at H = 1 the seed plays no part


def test_uneven_blocks_give_the_extra_engines_to_the_shortest_lived():
    """PT2 'as equal as possible', made exact: np.array_split's convention, extras in the first blocks."""
    for n, K, sizes in [(7, 5, [2, 2, 1, 1, 1]), (199, 10, [20] * 9 + [19]), (50, 5, [10] * 5)]:
        a = assign_clients({e: e for e in range(n)}, K, 1.0, seed=0)
        assert np.bincount(list(a.values()), minlength=K).tolist() == sizes


def test_ties_are_broken_by_engine_id():
    life = {e: 100 for e in range(1, 11)}                                  # every lifetime equal
    a = assign_clients(life, K=5, H=1.0, seed=0)
    assert [a[e] for e in range(1, 11)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_input_order_does_not_matter():
    life = fd004_like_lifetimes()
    shuffled = dict(sorted(life.items(), key=lambda kv: -kv[0]))
    for H in (0.0, 0.5, 1.0):
        assert assign_clients(life, 5, H, 3) == assign_clients(shuffled, 5, H, 3)


def test_only_the_lifetime_order_matters():
    life = fd004_like_lifetimes()
    stretched = {e: 2 * t + 1 for e, t in life.items()}                     # same order, different values
    assert assign_clients(life, 5, 0.5, 4) == assign_clients(stretched, 5, 0.5, 4)


def test_same_seed_same_assignment_different_seed_different():
    life = fd004_like_lifetimes()
    for H in (0.0, 0.25, 0.5, 0.75):
        assert assign_clients(life, 5, H, 1) == assign_clients(life, 5, H, 1)
        assert assign_clients(life, 5, H, 1) != assign_clients(life, 5, H, 2)


def test_every_engine_assigned_exactly_once_to_a_valid_client():
    life = fd004_like_lifetimes()
    a = assign_clients(life, 5, 0.5, 1)
    assert sorted(a) == sorted(life)
    assert all(type(e) is int and type(c) is int and 0 <= c < 5 for e, c in a.items())


def test_own_block_probability_matches_the_rule():
    """P(client k) = H * 1[b = k] + (1 - H) / K, so an engine lands in its own block with probability H + (1-H)/K."""
    life = {e: e for e in range(20000)}                                   # sorted order = id order
    blocks = assign_clients(life, 5, 1.0, 0)
    for H in (0.0, 0.5):
        a = assign_clients(life, 5, H, 7)
        own = np.mean([a[e] == blocks[e] for e in life])
        assert own == pytest.approx(H + (1 - H) / 5, abs=0.02)


def test_partitions_at_different_h_share_their_randomness():
    """PT8: one seed draws one fixed pair of vectors and H only moves the threshold, so an engine in its own block
    at some H stays there at every larger H."""
    life = fd004_like_lifetimes()
    blocks = assign_clients(life, 5, 1.0, 0)
    own = [{e for e, c in assign_clients(life, 5, H, 9).items() if c == blocks[e]} for H in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for smaller, larger in zip(own, own[1:]):
        assert smaller <= larger
    assert own[-1] == set(life)


@pytest.mark.parametrize("kwargs", [dict(K=0, H=0.5), dict(K=5, H=-0.1), dict(K=5, H=1.1),
                                    dict(K=5, H=float("nan")), dict(K=300, H=0.5)])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        assign_clients(fd004_like_lifetimes(), seed=1, **kwargs)


def test_empty_lifetimes_raise():
    with pytest.raises(ValueError):
        assign_clients({}, 5, 0.5, 1)


# ---------- validation routing (PT5, PT9) ----------

def test_validation_seed_is_fixed_distinct_and_not_the_run_seed():
    seeds = [1, 2, 3, 4, 5, 101, 102, 103]
    vs = [val_seed(s) for s in seeds]
    assert vs == [val_seed(s) for s in seeds]                              # deterministic
    assert len(set(vs)) == len(seeds)
    assert all(v != s for v, s in zip(vs, seeds))


def test_validation_draws_are_independent_of_training_draws():
    """With the raw run seed, the i-th sorted train engine and the i-th sorted val engine would get the same coin
    flips. Routed with val_seed, they do not."""
    life = {e: e for e in range(50)}
    assert assign_clients(life, 5, 0.0, 1) != assign_clients(life, 5, 0.0, val_seed(1))


# ---------- minimum engines per client (PT4, PT10) ----------

def test_min_engines_check():
    a = {e: (0 if e < 9 else 1) for e in range(30)}                        # client 0 has 9, client 1 has 21
    with pytest.raises(ValueError):
        check_min_engines(a, K=2, minimum=10)
    check_min_engines(a, K=2, minimum=9)                                   # exactly the minimum is allowed


def test_a_client_with_no_engines_fails_the_check():
    with pytest.raises(ValueError):
        check_min_engines({1: 0, 2: 0, 3: 0}, K=2, minimum=1)


# ---------- skew (PT6, PT11) ----------

def test_skew_is_zero_for_identical_distributions():
    y = np.array([0, 5, 5, 125, 30, 30], dtype=np.float32)
    assert skew([y, y], y) == 0.0


def test_skew_hand_calculated():
    assert skew([np.array([0, 0]), np.array([1, 1])], np.array([0, 1])) == pytest.approx(0.5)
    a = np.array([0, 1, 2, 3]); b = a + 7
    assert skew([a, b], np.concatenate([a, b])) == pytest.approx(3.5)       # checked against scipy, which we don't ship


def test_skew_is_the_mean_over_clients_not_the_worst():
    # all = [0,0,0,1]: client [0,0,0] is 0.25 cycles away, client [1] is 0.75 -> mean 0.5 (max would be 0.75)
    assert skew([np.array([0, 0, 0]), np.array([1])], np.array([0, 0, 0, 1])) == pytest.approx(0.5)


def test_skew_matches_sorted_sample_formula():
    """For equal-size samples, Wasserstein-1 = mean |sorted(a) - sorted(b)|: an independent way to compute it."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.integers(0, 126, 300), rng.integers(0, 126, 300)
        assert skew([a], b) == pytest.approx(np.mean(np.abs(np.sort(a) - np.sort(b))), rel=1e-12)


@pytest.mark.parametrize("bad", [np.array([1.5]), np.array([-1.0]), np.array([np.nan]), np.array([])])
def test_skew_rejects_labels_that_are_not_whole_numbers(bad):
    with pytest.raises(ValueError):
        skew([bad], np.array([0, 1, 2]))


def test_skew_rejects_no_clients():
    with pytest.raises(ValueError):
        skew([], np.array([0, 1, 2]))


def test_seed_averaged_skew_strictly_increases_over_the_h_grid(slice_cfg):
    """PT6, on FD004-like lifetimes: the knob H really controls label skew."""
    life, K = fd004_like_lifetimes(), slice_cfg["fl"]["K"]
    all_labels = np.concatenate([window_labels(t) for t in life.values()])
    means = []
    for H in slice_cfg["partition"]["H_grid"]:
        vals = [skew(client_labels(assign_clients(life, K, H, s), life, K), all_labels)
                for s in slice_cfg["seeds"]["main"]]
        means.append(np.mean(vals))
    assert all(a < b for a, b in zip(means, means[1:])), means


def test_window_labels_match_the_pipeline(slice_ds):
    """The helper above must be the pipeline's labelling, or the skew test tests something else."""
    for e, T in slice_ds.train_engine_lifetimes.items():
        np.testing.assert_array_equal(np.sort(slice_ds.y_train[slice_ds.engine_train == e]), np.sort(window_labels(T)))


# ---------- manifests (PT7, PT11, PT12) ----------

@pytest.fixture()
def cfg(slice_cfg):
    """The slice has 20 training engines, so K = 5 gives 4 per client: PT4's 10 is impossible on it."""
    slice_cfg["partition"]["min_engines_per_client"] = 2
    return slice_cfg


def test_manifest_covers_every_engine_once(cfg, slice_ds):
    m = build_manifest(cfg, slice_ds, H=0.5, seed=3)
    assert set(m["train"]) == set(slice_ds.train_engine_lifetimes)
    assert set(m["val"]) == set(slice_ds.val_engine_lifetimes)
    assert (m["H"], m["seed"], m["K"]) == (0.5, 3, cfg["fl"]["K"])
    assert sum(m["n_train_per_client"]) == len(slice_ds.train_engine_lifetimes)
    assert sum(m["n_val_per_client"]) == len(slice_ds.val_engine_lifetimes)
    assert isinstance(m["skew"], float) and m["skew"] >= 0                  # read by eval/plots.py


def test_manifest_validation_uses_its_own_seed(cfg, slice_ds):
    """Several seeds: with only 5 slice validation engines, one seed can route identically by chance (seed 3 at
    H = 0 does, about a 1-in-3,000 coincidence)."""
    K, life = cfg["fl"]["K"], slice_ds.val_engine_lifetimes
    differs = 0
    for seed in (1, 2, 4, 5):
        m = build_manifest(cfg, slice_ds, H=0.5, seed=seed)
        assert m["val"] == assign_clients(life, K, 0.5, val_seed(seed))
        differs += m["val"] != assign_clients(life, K, 0.5, seed)
    assert differs >= 3


def test_manifest_enforces_the_training_minimum(cfg, slice_ds):
    cfg["partition"]["min_engines_per_client"] = 5                         # 4 per client at H = 1
    with pytest.raises(ValueError):
        build_manifest(cfg, slice_ds, H=1.0, seed=1)


def test_hash_covers_the_assignment_and_nothing_else(cfg, slice_ds):
    m = build_manifest(cfg, slice_ds, H=0.5, seed=3)
    assert m["sha256"] == manifest_hash(m)
    assert manifest_hash({**m, "skew": 99.0, "n_val_per_client": [0]}) == m["sha256"]   # diagnostics excluded
    for field in ("train", "val"):
        changed = dict(m[field]); e = next(iter(changed)); changed[e] = (changed[e] + 1) % m["K"]
        assert manifest_hash({**m, field: changed}) != m["sha256"]
    assert manifest_hash({**m, "H": 0.75}) != m["sha256"]
    assert manifest_hash({**m, "seed": 4}) != m["sha256"]


def test_write_then_load_round_trip(cfg, slice_ds, tmp_path):
    m = build_manifest(cfg, slice_ds, H=0.25, seed=2)
    path = write_manifest(m, tmp_path)
    assert path.name == "H0.25_s2.json" == manifest_path(tmp_path, 0.25, 2).name
    back = load_manifest(path)
    assert back["train"] == m["train"] and back["val"] == m["val"]          # int keys again after JSON
    assert back["sha256"] == m["sha256"] == manifest_hash(back)


def test_load_refuses_a_hand_edited_manifest(cfg, slice_ds, tmp_path):
    path = write_manifest(build_manifest(cfg, slice_ds, H=0.5, seed=1), tmp_path)
    raw = json.loads(path.read_text())
    e = next(iter(raw["train"])); raw["train"][e] = (raw["train"][e] + 1) % raw["K"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_write_refuses_to_replace_a_different_manifest(cfg, slice_ds, tmp_path):
    write_manifest(build_manifest(cfg, slice_ds, H=0.5, seed=1), tmp_path)
    write_manifest(build_manifest(cfg, slice_ds, H=0.5, seed=1), tmp_path)  # identical: fine
    other = build_manifest(cfg, slice_ds, H=0.5, seed=1)
    e = next(iter(other["train"])); other["train"][e] = (other["train"][e] + 1) % other["K"]
    other["sha256"] = manifest_hash(other)
    with pytest.raises(FileExistsError):
        write_manifest(other, tmp_path)


def test_simulator_runs_on_a_loaded_manifest(cfg, slice_ds, tmp_path):
    """The format contract with run_federated (its docstring: {'train': {engine: client}, 'val': {...}})."""
    from rulfl.fl.simulate import run_federated
    cfg["model"] = {"hidden": 8, "layers": 1, "lr": 0.003, "dropout": 0.0}
    cfg["fl"].update(E=1, R=1, lr=0.003, mu=0.0)
    m = load_manifest(write_manifest(build_manifest(cfg, slice_ds, H=1.0, seed=1), tmp_path))
    out = run_federated(cfg, slice_ds, m, "fedavg", seed=1)
    assert len(out["client_val_mae"][1]) == cfg["fl"]["K"]


# ---------- make_all (PT12) ----------

def _force_skew(monkeypatch, skew_of_h):
    """Real manifests, but with each one's skew replaced by skew_of_h(H), so make_all's PT6 check can be steered."""
    import rulfl.partition.make_manifests as mm
    real = mm.build_manifest

    def fake(cfg, ds, H, seed, K=None):
        m = real(cfg, ds, H, seed, K)
        m["skew"] = skew_of_h(H)
        return m

    monkeypatch.setattr(mm, "build_manifest", fake)


def test_make_all_writes_every_h_and_seed(cfg, slice_ds, tmp_path, monkeypatch):
    """H grid x every seed the plan uses. Skew is replaced by an increasing stand-in: 20 slice engines are too few
    for PT6 to hold reliably, and PT6 itself is tested above on FD004-like lifetimes."""
    cfg["fl"]["K"] = 2                     # 20 slice engines over 5 clients can leave one with a single engine
    _force_skew(monkeypatch, lambda H: float(H))
    make_all(cfg, slice_ds, tmp_path)
    seeds = sorted(set(cfg["seeds"]["main"]) | set(cfg["seeds"]["selection"]) | set(cfg["seeds"]["pilot"]))
    names = sorted(p.name for p in tmp_path.glob("H*_s*.json"))
    assert names == sorted(f"H{H:.2f}_s{s}.json" for H in cfg["partition"]["H_grid"] for s in seeds)


def test_make_all_refuses_non_monotonic_skew_and_writes_nothing(cfg, slice_ds, tmp_path, monkeypatch):
    cfg["fl"]["K"] = 2
    _force_skew(monkeypatch, lambda H: 1.0)                                  # flat: H does nothing
    with pytest.raises(ValueError, match="monoton"):
        make_all(cfg, slice_ds, tmp_path)
    assert list(tmp_path.glob("*.json")) == []


# ---------- the committed manifests ----------

COMMITTED = sorted((REPO / "manifests").glob("H*_s*.json"))


@pytest.mark.skipif(not COMMITTED, reason="manifests not generated yet (python -m rulfl.partition.make_manifests)")
def test_committed_manifests_are_intact(slice_cfg):
    """PT7: committed once, never hand-edited. Needs no data: each file's embedded hash is rechecked."""
    seeds = sorted(set(slice_cfg["seeds"]["main"]) | set(slice_cfg["seeds"]["selection"]) | set(slice_cfg["seeds"]["pilot"]))
    expected = sorted(f"H{H:.2f}_s{s}.json" for H in slice_cfg["partition"]["H_grid"] for s in seeds)
    assert sorted(p.name for p in COMMITTED) == expected
    for p in COMMITTED:
        m = load_manifest(p)
        assert p.name == f"H{m['H']:.2f}_s{m['seed']}.json"
        assert set(m["train"].values()) == set(range(m["K"]))
        assert min(m["n_train_per_client"]) >= slice_cfg["partition"]["min_engines_per_client"]
