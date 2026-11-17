"""Batch tools: job lists per stage, parallel launcher, results index (DECISIONS E15, M2, F5, F7, E5)."""
import csv
import json

import pytest
import yaml

from rulfl.batch import build_index, job_succeeded, plan, run_jobs


@pytest.fixture()
def cfg(slice_cfg):
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    return slice_cfg


def test_tuning_plan_is_the_12_config_screen_on_seed_1(cfg):
    jobs = plan("tune", cfg)
    assert len(jobs) == len(set(jobs)) == 12
    assert all(j.startswith("tune --seed 1 ") for j in jobs)
    for j in jobs:                                                       # M2: dropout 0.2 only with 2 layers
        assert ("model.layers=2" in j) == ("model.dropout=0.2" in j)


def test_main_grid_plan_is_75_paired_runs(cfg):
    jobs = plan("main", cfg)
    assert len(jobs) == len(set(jobs)) == 75
    for seed in range(1, 6):
        for H in (0.0, 0.25, 0.5, 0.75, 1.0):
            algos = {j.split("--algo ")[1].split()[0] for j in jobs if f"--seed {seed} " in j and f"--H {H} " in j + " "}
            assert algos == {"fedavg", "fedprox", "fedla"}


def test_other_plans_have_the_planned_sizes(cfg):
    assert len(plan("central", cfg)) == 5                                # E5, seeds 1-5
    assert len(plan("pilot", cfg)) == 16                                 # F7: E x lr x H x seeds 101-102
    assert len(plan("musel", cfg)) == 12                                 # F5: 4 mu x seeds 101-103
    assert len(plan("local", cfg)) == 50                                 # E5: H 0 and 1 x 5 seeds x 5 clients
    assert plan("probe", cfg) == [j.replace("main ", "probe ", 1) for j in plan("main", cfg)]   # E6: one per main run
    assert all("--seed 1" not in j.replace("--seed 10", "") for j in plan("pilot", cfg) + plan("musel", cfg))


def test_pilot_uses_the_tuned_lr_and_its_third(cfg):
    lrs = {j.split("fl.lr=")[1].split()[0] for j in plan("pilot", cfg)}
    assert lrs == {"0.003", "0.000999"}                                  # 0.003 x lr_factors [1.0, 0.333]


def test_run_jobs_keeps_going_after_a_failure(tmp_path, cfg):
    cfg["train"]["max_epochs"] = 1
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    extra = ["--config", str(cfg_file), "--runs-dir", str(tmp_path / "runs"), "--allow-dirty"]
    failed = run_jobs(["central --seed 101", "central --seed 1"], jobs=2, extra=extra)   # 101: not a central seed
    assert failed == ["central --seed 101"]
    assert (tmp_path / "runs" / "central" / "FD004_s1" / "DONE").exists()


def test_index_has_one_row_per_finished_run(tmp_path):
    def fake(group, name, done, **final):
        d = tmp_path / "runs" / group / name
        d.mkdir(parents=True)
        (d / "final.json").write_text(json.dumps({"group": group, "name": name, "test": {"mae": 20.0}, **final}))
        if done:
            (d / "DONE").write_text("done\n")

    fake("main", "fedla_H0.50_s3", True, seed=3, algo="fedla", H=0.5, best_val_mae=18.0)
    fake("central", "FD004_s1", True, seed=1, best_val_mae=15.0)
    fake("main", "fedavg_H0.50_s3", False, seed=3)                       # not finished: never indexed
    out = build_index(tmp_path / "runs", tmp_path / "results" / "index.csv")
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["group"], r["name"]) for r in rows] == [("central", "FD004_s1"), ("main", "fedla_H0.50_s3")]
    assert rows[1]["test_mae"] == "20.0" and rows[1]["best_val_mae"] == "18.0" and rows[0]["algo"] == ""


def test_success_is_the_runners_done_line_not_the_exit_code():
    """M8: on Windows, CUDA's cuDNN crashes while the process exits after a 2-layer GRU with dropout
    (0xC0000409), AFTER the run folder is complete. The runner prints DONE/SKIPPED only once the folder is final."""
    assert job_succeeded(0, "loading\nDONE runs/main/fedla_H0.50_s3\n")
    assert job_succeeded(0xC0000409, "DONE runs/main/fedla_H0.50_s3\n")              # crash at exit, run saved
    assert job_succeeded(0xC0000409, "SKIPPED (already done) runs/main/x\n")
    assert not job_succeeded(1, "Traceback (most recent call last): ...\n")          # real failure
    assert not job_succeeded(0xC0000409, "epoch 3\n")                                # crashed mid-run
    assert not job_succeeded(0, "")
