"""Analysis from run folders (rule 6, pre-registration): fake runs with known answers."""
import csv
import json

import pytest

from rulfl.analysis import analyse, holm

HS = (0.0, 0.25, 0.5, 0.75, 1.0)


def write_run(root, group, name, **final):
    d = root / group / name
    d.mkdir(parents=True)
    (d / "final.json").write_text(json.dumps({"group": group, "name": name, **final}))
    (d / "DONE").write_text("done\n")


def run_fields(mae):
    return {"test": {"mae": mae, "rmse": mae * 1.5, "nasa": mae * 100},
            "life_stage_mae": {"near_failure": mae + 3, "mid": mae, "healthy": mae - 3,
                               "n_near_failure": 79, "n_mid": 102, "n_healthy": 67},
            "worst_client_mae": mae + 1, "round_to_threshold": 10, "threshold_reached": True, "total_bytes": 1000}


@pytest.fixture()
def fake(tmp_path):
    """FedAvg MAE = 10 + 2H + 0.1*seed. FedLA = FedAvg - 0.5 everywhere. FedProx = FedAvg +/- 0.05, alternating."""
    runs = tmp_path / "runs"
    for H in HS:
        for s in range(1, 6):
            avg = 10 + 2 * H + 0.1 * s
            for algo, mae in (("fedavg", avg), ("fedla", avg - 0.5), ("fedprox", avg + (0.05 if s % 2 else -0.05))):
                write_run(runs, "main", f"{algo}_H{H:.2f}_s{s}", algo=algo, H=H, seed=s, **run_fields(mae))
                write_run(runs, "probe", f"{algo}_H{H:.2f}_s{s}", algo=algo, H=H, seed=s, **run_fields(mae - 1))
    for s in range(1, 6):
        write_run(runs, "central", f"FD004_s{s}", algo=None, H=None, seed=s, **run_fields(9.0))
    write_run(runs, "central", "FD001_s1", algo=None, H=None, seed=1, **run_fields(99.0))     # not FD004: ignored
    for H in (0.0, 1.0):
        for c in range(5):
            write_run(runs, "local", f"H{H:.2f}_s1_c{c}", algo=None, H=H, seed=1, client=c, **run_fields(20.0 + H))
    man = tmp_path / "manifests"
    man.mkdir()
    for H in HS:
        (man / f"H{H:.2f}_s1.json").write_text(json.dumps({"H": H, "seed": 1, "skew": 1 + 9 * H}))
    out = analyse(runs, tmp_path / "results", man, with_figures=False)
    return out


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_holm_adjustment():
    assert holm({"P1": 0.01, "P2": 0.2}) == {"P1": 0.02, "P2": 0.2}
    assert holm({"P1": 0.04, "P2": 0.03}) == {"P2": 0.06, "P1": 0.06}      # step-down keeps order monotone


def test_main_table_means(fake):
    t = {(r["algo"], float(r["H"])): r for r in rows(fake / "main_table.csv")}
    assert float(t[("fedavg", 0.0)]["test_mae_mean"]) == pytest.approx(10.3)       # 10 + 0.1 * mean(1..5)
    assert float(t[("fedla", 1.0)]["test_mae_mean"]) == pytest.approx(11.8)
    assert float(t[("fedavg", 0.5)]["mae_near_failure_mean"]) == pytest.approx(14.3)
    assert int(t[("fedla", 0.5)]["n"]) == 5


def test_primary_hypotheses_follow_the_preregistration(fake):
    p = {r["hypothesis"]: r for r in rows(fake / "primary.csv")}
    assert int(p["P1"]["n_pairs"]) == 15 and int(p["P2"]["n_pairs"]) == 15          # pooled over H >= 0.5
    assert float(p["P1"]["mean_diff"]) == pytest.approx(-0.5)
    assert float(p["P1"]["p_less"]) == pytest.approx(1 / 2 ** 15)                    # all 15 pairs better
    assert p["P1"]["significant_at_0.05"] == "True"
    assert p["P2"]["significant_at_0.05"] == "False"


def test_premise_and_per_H(fake):
    prem = rows(fake / "premise.csv")[0]
    assert float(prem["mean_diff"]) == pytest.approx(2.0)                             # FedAvg H=1 minus H=0
    per = rows(fake / "paired_per_H.csv")
    assert len(per) == 10                                                               # 2 methods x 5 H
    la0 = next(r for r in per if r["algo"] == "fedla" and float(r["H"]) == 0.0)
    assert float(la0["wilcoxon_p_less"]) == pytest.approx(1 / 32)                     # 5 pairs: smallest p


def test_baselines_probe_and_summary(fake):
    b = {(r["baseline"], float(r["H"]) if r["H"] else None): r for r in rows(fake / "baselines.csv")}
    assert float(b[("centralized", None)]["test_mae_mean"]) == pytest.approx(9.0)      # FD001 run ignored
    assert float(b[("local_only", 1.0)]["test_mae_mean"]) == pytest.approx(21.0)
    pr = next(r for r in rows(fake / "probe.csv") if r["algo"] == "fedavg" and float(r["H"]) == 1.0)
    assert float(pr["test_mae_federated"]) - float(pr["test_mae_head_refit"]) == pytest.approx(1.0)
    s = (fake / "summary.md").read_text(encoding="utf8")
    assert "P1" in s and "Premise" in s and "Head-refit" in s


def test_unfinished_runs_are_ignored(tmp_path):
    d = tmp_path / "runs" / "main" / "fedavg_H0.00_s1"
    d.mkdir(parents=True)
    (d / "final.json").write_text("{}")                                                # no DONE
    with pytest.raises(FileNotFoundError):
        analyse(tmp_path / "runs", tmp_path / "out", tmp_path, with_figures=False)


def test_figures_are_drawn(fake, tmp_path):
    pytest.importorskip("matplotlib")
    from rulfl.analysis import figures, main_table  # noqa: F401  (import check only)
    out = analyse(fake.parent / "runs", tmp_path / "with_figs", fake.parent / "manifests", with_figures=True)
    assert {"fig_mae_vs_H.png", "fig_life_stage.png", "fig_probe.png", "fig_skew.png"} <= {p.name for p in out.iterdir()}


def test_life_stage_split_of_the_skew_penalty(fake):
    """Fake: every stage MAE rises by 2 from H = 0 to H = 1, so near failure contributes (79 / 248) * 2 = 0.637."""
    s = (fake / "summary.md").read_text(encoding="utf8")
    assert "| fedavg | near_failure | 13.30 | 15.30 | +2.00 | +0.637 |" in s
    assert "| fedavg | healthy | 7.30 | 9.30 | +2.00 | +0.540 |" in s                # 67 / 248 * 2
    t = rows(fake / "main_table.csv")[0]
    assert (int(t["n_near_failure"]), int(t["n_mid"]), int(t["n_healthy"])) == (79, 102, 67)


def test_probe_change_has_a_paired_interval(fake):
    pr = next(r for r in rows(fake / "probe.csv") if r["algo"] == "fedla" and float(r["H"]) == 0.5)
    assert float(pr["change"]) == pytest.approx(-1.0)
    assert float(pr["change_ci_lo"]) == pytest.approx(-1.0) and float(pr["change_ci_hi"]) == pytest.approx(-1.0)
    assert int(pr["n"]) == 5


def test_client_label_mix_on_the_slice(slice_cfg, slice_ds):
    from rulfl.analysis import client_label_mix
    from rulfl.partition.label_skew import assign_clients
    m = {"H": 1.0, "seed": 1, "K": 5, "train": assign_clients(slice_ds.train_engine_lifetimes, 5, 1.0, 1)}  # slice too small for PT4
    mix = client_label_mix(slice_ds, m, slice_cfg["fedla"]["bin_edges"], slice_cfg["data"]["rul_cap"])
    assert [r["client"] for r in mix] == ["all", 0, 1, 2, 3, 4]
    assert sum(r["engines"] for r in mix[1:]) == mix[0]["engines"] == len(slice_ds.train_engine_lifetimes)
    assert sum(r["windows"] for r in mix[1:]) == mix[0]["windows"] == len(slice_ds.y_train)
    for r in mix:
        assert sum(v for k, v in r.items() if k.startswith("share_")) == pytest.approx(1.0)
    caps = [r["share_125"] for r in mix[1:]]
    assert caps == sorted(caps)                    # H = 1: client k holds lifetime block k, longer lives = more capped
