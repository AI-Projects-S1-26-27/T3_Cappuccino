"""Experiment runner tests (DECISIONS E8, E15)."""
import csv
import json
import subprocess

import numpy as np
import pytest
import torch

from rulfl.fl.simulate import run_federated
from rulfl.model.gru import RULModel, predict_rul
from dataclasses import replace

from rulfl.fl.simulate import client_rows
from rulfl.model.train import train_centralized
import yaml

from rulfl.partition.make_manifests import build_manifest, write_manifest
from rulfl.run import centralized_run, federated_run, git_commit, in_run_folder, load_config, local_run, main, run_name


def test_names_follow_e15_examples():
    assert run_name("tune", {"hidden": 64, "layers": 2, "lr": 0.003, "seed": 1}) == "h64_l2_lr0.003_s1"
    assert run_name("central", {"fd": "FD004", "seed": 1}) == "FD004_s1"
    assert run_name("pilot", {"algo": "fedavg", "H": 1, "seed": 101, "E": 3, "lr": 0.001}) \
        == "fedavg_H1.00_s101_E3_lr0.001"
    assert run_name("musel", {"algo": "fedprox", "H": 1.0, "seed": 101, "mu": 0.01}) == "fedprox_H1.00_s101_mu0.01"
    assert run_name("main", {"algo": "fedla", "H": 0.5, "seed": 3}) == "fedla_H0.50_s3"
    assert run_name("local", {"H": 1, "seed": 2, "client": 4}) == "H1.00_s2_c4"


def test_mu_selection_runs_get_different_folders():
    """The bug E15 exists for: 4 mu values at the same (algo, H, seed) must not share one folder."""
    names = {run_name("musel", {"algo": "fedprox", "H": 1, "seed": 101, "mu": mu}) for mu in (0.001, 0.01, 0.1, 1)}
    assert len(names) == 4


def test_pilot_lr_divided_by_3_gives_a_short_readable_name():
    """The pilot tries tuned lr / 3 (F7): 0.001 / 3 must not become a 20-digit folder name."""
    assert run_name("pilot", {"algo": "fedavg", "H": 0, "seed": 101, "E": 1, "lr": 0.001 / 3}) \
        == "fedavg_H0.00_s101_E1_lr0.000333333"


def test_unknown_group_or_missing_setting_raises():
    with pytest.raises(ValueError, match="group"):
        run_name("grid", {"algo": "fedavg", "H": 0, "seed": 1})
    with pytest.raises(ValueError, match="mu"):
        run_name("musel", {"algo": "fedprox", "H": 1, "seed": 101})


# ---------- one run folder: skip, refuse, build in .tmp (E15) ----------

def write_result(folder):
    (folder / "final.json").write_text("{}")


def test_a_finished_run_has_config_result_and_done(tmp_path):
    folder, ran = in_run_folder(tmp_path, "main", "fedavg_H0.00_s1", {"fl": {"R": 2}}, write_result)
    assert ran and folder == tmp_path / "main" / "fedavg_H0.00_s1"
    assert {p.name for p in folder.iterdir()} == {"config.yaml", "final.json", "DONE"}
    assert not (tmp_path / "main" / "fedavg_H0.00_s1.tmp").exists()


def test_same_run_again_is_skipped_without_training(tmp_path):
    in_run_folder(tmp_path, "main", "x", {"fl": {"R": 2}}, write_result)
    calls = []
    folder, ran = in_run_folder(tmp_path, "main", "x", {"fl": {"R": 2}}, calls.append)
    assert not ran and calls == []


def test_same_folder_with_a_different_config_is_refused(tmp_path):
    """A naming slip (two different runs, one name) must stop, never skip or overwrite."""
    in_run_folder(tmp_path, "main", "x", {"fl": {"R": 2}}, write_result)
    with pytest.raises(RuntimeError, match="different config"):
        in_run_folder(tmp_path, "main", "x", {"fl": {"R": 3}}, write_result)
    assert "R: 2" in (tmp_path / "main" / "x" / "config.yaml").read_text()


def test_a_crash_leaves_an_invalid_folder_and_a_rerun_replaces_it(tmp_path):
    def crash(folder):
        (folder / "half.txt").write_text("partial")
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError):
        in_run_folder(tmp_path, "main", "x", {"a": 1}, crash)
    assert not (tmp_path / "main" / "x").exists()                     # never looks finished
    bad = tmp_path / "main" / "x.invalid"
    assert (bad / "half.txt").exists() and not (bad / "DONE").exists()

    folder, ran = in_run_folder(tmp_path, "main", "x", {"a": 1}, write_result)   # same seed, same manifest
    assert ran and (folder / "DONE").exists() and not bad.exists()


# ---------- git state (rule 3: never a registered run from a dirty tree) ----------

def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_commit_returns_the_hash_and_refuses_a_dirty_tree(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("1")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    sha = git_commit(tmp_path)
    assert len(sha) == 40

    (tmp_path / "b.txt").write_text("untracked")                      # untracked files count as dirty too
    with pytest.raises(RuntimeError, match="dirty"):
        git_commit(tmp_path)
    assert git_commit(tmp_path, allow_dirty=True) == sha


# ---------- a federated run's files (E8, E15) ----------

@pytest.fixture()
def fl_cfg(slice_cfg):
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    slice_cfg["fl"].update(K=2, E=1, R=2, lr=0.003, mu=0.01)
    return slice_cfg


def two_clients(ds):
    deal = lambda engines: {int(e): i % 2 for i, e in enumerate(sorted(set(engines.tolist())))}
    return {"train": deal(ds.engine_train), "val": deal(ds.engine_val), "sha256": "abc123"}


def test_federated_run_writes_rounds_checkpoint_predictions(tmp_path, fl_cfg, slice_ds):
    man = two_clients(slice_ds)
    out = federated_run(tmp_path, fl_cfg, slice_ds, man, "fedavg", seed=1, threshold=1e9)
    ref = run_federated(fl_cfg, slice_ds, man, "fedavg", seed=1)              # same seed -> same run

    rounds = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rounds] == [0, 1, 2]
    assert [r["val_mae"] for r in rounds] == ref["val_mae"]
    assert [r["bytes"] for r in rounds] == ref["bytes"]

    m = fl_cfg["model"]
    model = RULModel(slice_ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    model.load_state_dict(torch.load(tmp_path / "best.pt"))
    pred = predict_rul(model, slice_ds.X_test, 125)
    with open(tmp_path / "test_predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["engine"]) for r in rows] == slice_ds.engine_test.tolist()
    assert np.allclose([float(r["pred"]) for r in rows], pred)
    assert out["test"]["mae"] == pytest.approx(np.abs(pred - slice_ds.y_test).mean())


def test_federated_final_fields(tmp_path, fl_cfg, slice_ds):
    man = two_clients(slice_ds)
    out = federated_run(tmp_path, fl_cfg, slice_ds, man, "fedla", seed=1, threshold=1e9)
    ref = run_federated(fl_cfg, slice_ds, man, "fedla", seed=1)
    b = ref["best_round"]
    assert out["best_round"] == b and out["best_val_mae"] == ref["val_mae"][b]
    assert out["worst_client_mae"] == max(ref["client_val_mae"][b])
    assert out["total_bytes"] == sum(ref["bytes"])
    assert out["manifest_sha256"] == "abc123"
    assert (out["round_to_threshold"], out["threshold_reached"]) == (0, True)    # huge threshold: round 0 counts
    assert set(out["life_stage_mae"]) >= {"near_failure", "mid", "healthy"}
    json.dumps(out, allow_nan=False)                                              # final.json must be strict JSON


def test_federated_run_without_a_threshold_is_refused(tmp_path, fl_cfg, slice_ds):
    """E3 needs the centralized threshold; a federated run must not silently skip rounds-to-110%."""
    with pytest.raises(ValueError, match="threshold"):
        federated_run(tmp_path, fl_cfg, slice_ds, two_clients(slice_ds), "fedavg", seed=1, threshold=None)


# ---------- centralized, tuning and local-only runs (E5, M2) ----------

@pytest.fixture()
def c_cfg(fl_cfg):
    fl_cfg["train"]["max_epochs"] = 2
    return fl_cfg


def test_centralized_run_is_train_centralized_with_files(tmp_path, c_cfg, slice_ds):
    out = centralized_run(tmp_path, c_cfg, slice_ds, seed=2)
    _, hist = train_centralized(c_cfg, slice_ds, seed=2)
    assert json.loads((tmp_path / "epochs.json").read_text()) == hist
    assert out["best_epoch"] == hist["best_epoch"]
    assert out["best_val_mae"] == hist["val_mae"][hist["best_epoch"]]
    assert {"best.pt", "test_predictions.csv"} <= {p.name for p in tmp_path.iterdir()}
    json.dumps(out, allow_nan=False)


def test_local_run_trains_on_one_clients_engines_only(tmp_path, c_cfg, slice_ds):
    """E5: client 1 alone, its own training and validation engines; scored on the full test set."""
    man = two_clients(slice_ds)
    out = local_run(tmp_path, c_cfg, slice_ds, man, client=1, seed=2)
    tr = client_rows(slice_ds.engine_train, man["train"], 2)[1]
    va = client_rows(slice_ds.engine_val, man["val"], 2)[1]
    mine = replace(slice_ds, X_train=slice_ds.X_train[tr], y_train=slice_ds.y_train[tr],
                   engine_train=slice_ds.engine_train[tr], X_val=slice_ds.X_val[va], y_val=slice_ds.y_val[va],
                   engine_val=slice_ds.engine_val[va])
    _, hist = train_centralized(c_cfg, mine, seed=2)
    assert out["best_val_mae"] == hist["val_mae"][hist["best_epoch"]]
    assert out["manifest_sha256"] == "abc123" and out["client"] == 1
    with open(tmp_path / "test_predictions.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(slice_ds.X_test)


def test_local_run_refuses_a_client_without_validation_engines(tmp_path, c_cfg, slice_ds):
    man = two_clients(slice_ds)
    man["val"] = {e: 0 for e in man["val"]}                       # client 1 has no validation engines
    with pytest.raises(ValueError, match="validation"):
        local_run(tmp_path, c_cfg, slice_ds, man, client=1, seed=2)


# ---------- the command line: python -m rulfl.run <group> ... ----------

@pytest.fixture()
def cli(tmp_path, c_cfg):
    """A config file pointing at the slice, tiny model, manifests and runs in tmp. Returns a runner."""
    c_cfg["partition"].update(min_engines_per_client=2, manifests_dir=str(tmp_path / "manifests"))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(c_cfg))
    runs = tmp_path / "runs"

    def run(*argv):
        return main([*argv, "--config", str(cfg_file), "--runs-dir", str(runs), "--allow-dirty"])
    run.cfg, run.runs = c_cfg, runs
    return run


def fake_central_runs(runs, maes):
    """Five finished centralized FD004 runs; only best_val_mae is read for the E3 threshold."""
    for seed, v in zip(range(1, 6), maes):
        d = runs / "central" / f"FD004_s{seed}"
        d.mkdir(parents=True)
        (d / "final.json").write_text(json.dumps({"best_val_mae": v}))
        (d / "DONE").write_text("done\n")


def test_cli_central_run_end_to_end_and_skip(cli):
    folder = cli("central", "--seed", "1")
    assert folder == cli.runs / "central" / "FD004_s1"
    final = json.loads((folder / "final.json").read_text())
    assert final["group"] == "central" and final["seed"] == 1
    assert len(final["git_commit"]) == 40 and isinstance(final["git_dirty"], bool) and len(final["data_hash"]) == 64
    assert final["seconds"] > 0 and final["test"]["mae"] > 0

    before = (folder / "final.json").stat().st_mtime_ns
    assert cli("central", "--seed", "1") == folder                     # second time: skipped
    assert (folder / "final.json").stat().st_mtime_ns == before


def test_cli_main_run_uses_the_manifest_and_the_centralized_threshold(cli, slice_ds):
    man = build_manifest(cli.cfg, slice_ds, 0.5, 1)
    write_manifest(man, cli.cfg["partition"]["manifests_dir"])
    fake_central_runs(cli.runs, [10, 20, 30, 40, 50])                   # mean 30 -> threshold 1.10 x 30 = 33
    folder = cli("main", "--algo", "fedla", "--H", "0.5", "--seed", "1")
    assert folder == cli.runs / "main" / "fedla_H0.50_s1"
    final = json.loads((folder / "final.json").read_text())
    assert final["manifest_sha256"] == man["sha256"]
    curve = [json.loads(line)["val_mae"] for line in (folder / "rounds.jsonl").read_text().splitlines()]
    first = next((r for r, v in enumerate(curve) if v <= 33.0), None)
    assert final["round_to_threshold"] == first and final["threshold_reached"] == (first is not None)


def test_cli_tuning_names_the_folder_from_the_config(cli):
    folder = cli("tune", "--seed", "1", "--set", "model.hidden=8", "--set", "model.lr=0.001")
    assert folder.name == "h8_l1_lr0.001_s1"
    assert yaml.safe_load((folder / "config.yaml").read_text())["model"]["hidden"] == 8


def test_cli_refuses_wrong_seeds_algos_and_missing_baselines(cli, slice_ds):
    with pytest.raises(ValueError, match="seed"):
        cli("main", "--algo", "fedavg", "--H", "0", "--seed", "101")        # selection seed in the main grid
    with pytest.raises(ValueError, match="seed"):
        cli("musel", "--algo", "fedprox", "--H", "1", "--seed", "1")        # reported seed for mu selection
    with pytest.raises(ValueError, match="fedprox"):
        cli("musel", "--algo", "fedavg", "--H", "1", "--seed", "101")
    write_manifest(build_manifest(cli.cfg, slice_ds, 0.0, 1), cli.cfg["partition"]["manifests_dir"])
    with pytest.raises(FileNotFoundError, match="central"):
        cli("main", "--algo", "fedavg", "--H", "0", "--seed", "1")          # no centralized runs yet (E3)
    assert not (cli.runs / "main").exists()                               # failed before any folder was made


def test_set_changes_existing_keys_only_except_settings_chosen_later(tmp_path):
    """model.* is created by tuning (M2); fl.E, fl.R, fl.lr by the pilot (F7); fl.mu by mu selection (F5)."""
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"fl": {"K": 5}}))
    cfg = load_config(f, ["fl.K=2", "model.hidden=64", "fl.E=3", "fl.R=60", "fl.lr=0.001", "fl.mu=0.1"])
    assert cfg["fl"] == {"K": 2, "E": 3, "R": 60, "lr": 0.001, "mu": 0.1} and cfg["model"]["hidden"] == 64
    with pytest.raises(ValueError, match="fl.Mu"):
        load_config(f, ["fl.Mu=0.1"])                                     # a typo must not pass silently


def test_pilot_runs_before_mu_exists(cli, slice_ds):
    """F7 comes before F5: a FedAvg pilot run must not need fl.mu."""
    del cli.cfg["fl"]["mu"]
    cfg_file = cli.runs.parent / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cli.cfg))
    write_manifest(build_manifest(cli.cfg, slice_ds, 1.0, 101), cli.cfg["partition"]["manifests_dir"])
    fake_central_runs(cli.runs, [10, 20, 30, 40, 50])
    folder = cli("pilot", "--algo", "fedavg", "--H", "1", "--seed", "101", "--set", "fl.E=1", "--set", "fl.lr=0.001")
    assert folder.name == "fedavg_H1.00_s101_E1_lr0.001"


def test_cli_probe_refits_the_head_of_a_finished_main_run(cli, slice_ds):
    """E6: probe/<name> starts from main/<name>/best.pt; its GRU is unchanged, its head is new."""
    with pytest.raises(FileNotFoundError, match="main"):
        cli("probe", "--algo", "fedla", "--H", "0.5", "--seed", "1")              # no main run yet
    write_manifest(build_manifest(cli.cfg, slice_ds, 0.5, 1), cli.cfg["partition"]["manifests_dir"])
    fake_central_runs(cli.runs, [10, 20, 30, 40, 50])
    main_dir = cli("main", "--algo", "fedla", "--H", "0.5", "--seed", "1")
    folder = cli("probe", "--algo", "fedla", "--H", "0.5", "--seed", "1")
    assert folder == cli.runs / "probe" / "fedla_H0.50_s1"
    a, b = torch.load(main_dir / "best.pt"), torch.load(folder / "best.pt")
    assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("gru."))
    assert not all(torch.equal(a[k], b[k]) for k in a if k.startswith("head."))
    final = json.loads((folder / "final.json").read_text())
    assert final["source_run"] == "main/fedla_H0.50_s1" and final["diagnostic"] is True
    assert final["source_test_mae"] == json.loads((main_dir / "final.json").read_text())["test"]["mae"]
