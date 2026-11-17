"""Experiment runner: one command runs one experiment and saves everything a report number needs (DECISIONS E8, E15).

Folder = runs/<group>/<name>/, where <name> holds every setting that varies inside its group.

    python -m rulfl.run tune    --seed 1   --set model.hidden=64 --set model.layers=2 --set model.lr=0.003 --set model.dropout=0.2
    python -m rulfl.run central --seed 1                                   (FD001 check: --set data.fd=FD001)
    python -m rulfl.run pilot   --seed 101 --algo fedavg  --H 1   --set fl.E=3 --set fl.lr=0.001
    python -m rulfl.run musel   --seed 101 --algo fedprox --H 1   --set fl.mu=0.01
    python -m rulfl.run main    --seed 3   --algo fedla   --H 0.5
    python -m rulfl.run local   --seed 2   --H 1 --client 4
    python -m rulfl.run probe   --seed 3   --algo fedla   --H 0.5          (head refit of main/fedla_H0.50_s3, E6)
    python -m rulfl.run probectl --seed 3                                  (same refit on an untrained GRU, E20)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import string
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from rulfl.data.pipeline import Dataset, build_dataset
from rulfl.eval.metrics import life_stage_mae, metrics, rounds_to_threshold
from rulfl.fl.simulate import client_rows, run_federated
from rulfl.model.gru import RULModel, predict_rul
from rulfl.model.train import train_centralized
from rulfl.partition.make_manifests import load_manifest, manifest_path
from rulfl.probe import refit_head

# One name pattern per group (E15). lr and mu use :g so lr / 3 stays short (0.000333333, not 0.0003333333333333333).
NAME_FORMATS = {
    "tune": "h{hidden}_l{layers}_lr{lr:g}_s{seed}",
    "central": "{fd}_s{seed}",
    "pilot": "{algo}_H{H:.2f}_s{seed}_E{E}_lr{lr:g}",
    "musel": "{algo}_H{H:.2f}_s{seed}_mu{mu:g}",
    "main": "{algo}_H{H:.2f}_s{seed}",
    "local": "H{H:.2f}_s{seed}_c{client}",
    "probe": "{algo}_H{H:.2f}_s{seed}",
    "probectl": "untrained_s{seed}",
}


def run_name(group: str, params: dict) -> str:
    """Folder name for one run inside runs/<group>/.
    Example: run_name('musel', {'algo': 'fedprox', 'H': 1, 'seed': 101, 'mu': 0.01}) -> 'fedprox_H1.00_s101_mu0.01'."""
    if group not in NAME_FORMATS:
        raise ValueError(f"unknown group {group!r}; expected one of {sorted(NAME_FORMATS)}")
    fmt = NAME_FORMATS[group]
    needed = [field for _, field, _, _ in string.Formatter().parse(fmt) if field]
    missing = [f for f in needed if f not in params]
    if missing:
        raise ValueError(f"group {group!r} needs {missing} to name the run")
    return fmt.format(**params)


def in_run_folder(root, group: str, name: str, cfg: dict, work: Callable[[Path], None]) -> tuple[Path, bool]:
    """Run `work(folder)` for one run, safely (E15). Returns (final folder, True if it ran / False if skipped).

    - final folder exists with DONE and the same config.yaml  -> skipped, work is not called (resume after a crash)
    - final folder exists with anything else                  -> refused (a naming slip must never overwrite a run)
    - otherwise: build in <name>.tmp, write config.yaml, call work, write DONE last, then rename to <name>.
      If work raises, the folder becomes <name>.invalid (kept to inspect) and the error is re-raised;
      a later successful rerun deletes it."""
    parent = Path(root) / group
    target, tmp, invalid = parent / name, parent / f"{name}.tmp", parent / f"{name}.invalid"
    cfg_text = yaml.safe_dump(cfg, sort_keys=True)

    if target.exists():
        done = (target / "DONE").exists()
        same = (target / "config.yaml").exists() and (target / "config.yaml").read_text(encoding="utf8") == cfg_text
        if done and same:
            return target, False
        raise RuntimeError(f"{target} already exists with a different config or without DONE; refusing to touch it")

    if tmp.exists():                                         # leftover from a run killed mid-way
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "config.yaml").write_text(cfg_text, encoding="utf8")
    try:
        work(tmp)
        (tmp / "DONE").write_text("done\n", encoding="utf8")
    except BaseException:
        if invalid.exists():
            shutil.rmtree(invalid)
        tmp.rename(invalid)
        raise
    tmp.rename(target)
    if invalid.exists():
        shutil.rmtree(invalid)
    return target, True


def git_commit(repo=".", allow_dirty: bool = False) -> str:
    """Current commit hash. Raises if the tree has any change or untracked file (CLAUDE.md rule 3), unless
    allow_dirty (development only, never for a registered run)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain") and not allow_dirty:
        raise RuntimeError("git tree is dirty: commit or remove the changes before a registered run (rule 3)")
    return sha


def _clean(x):
    """NaN / inf -> None, numpy numbers -> plain Python, so final.json is strict JSON (an empty life-stage
    group or a client without validation engines gives NaN)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def save_model_outputs(folder: Path, model: RULModel, ds: Dataset, cfg: dict) -> dict:
    """Shared by every run kind: best.pt, test_predictions.csv, and the test scores (E1, E4)."""
    cap = cfg["data"]["rul_cap"]
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder / "best.pt")   # loads on any machine
    pred = predict_rul(model, ds.X_test, cap)
    with open(folder / "test_predictions.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["engine", "pred", "true", "true_uncapped"])
        for row in zip(ds.engine_test, pred, ds.y_test, ds.y_test_uncapped):
            w.writerow([int(row[0]), *(float(v) for v in row[1:])])
    return {"test": metrics(pred, ds.y_test),
            "life_stage_mae": life_stage_mae(pred, ds.y_test, cfg["eval"]["life_stage_edges"])}


def federated_run(folder, cfg: dict, ds: Dataset, manifest: dict, algo: str, seed: int, threshold) -> dict:
    """One federated run on the reference simulator (F11: registered runs use the simulator).
    Writes rounds.jsonl (one line per round 0..R), best.pt, test_predictions.csv; returns final.json's run fields.
    `threshold` = 1.10 x mean best centralized validation MAE (E3), required."""
    if threshold is None:
        raise ValueError("federated runs need the rounds-to-110% threshold from the centralized runs (E3)")
    folder = Path(folder)
    log = run_federated(cfg, ds, manifest, algo, seed)
    with open(folder / "rounds.jsonl", "w", encoding="utf8") as f:
        for rnd, (v, c, b) in enumerate(zip(log["val_mae"], log["client_val_mae"], log["bytes"])):
            f.write(json.dumps(_clean({"round": rnd, "val_mae": v, "client_val_mae": c, "bytes": b})) + "\n")

    m = cfg["model"]
    model = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"])
    model.load_state_dict(log["best_state"])
    best = log["best_round"]
    clients = [v for v in log["client_val_mae"][best] if math.isfinite(v)]
    reached_round, reached = rounds_to_threshold(log["val_mae"], threshold)
    out = save_model_outputs(folder, model, ds, cfg)
    out.update(best_round=best, best_val_mae=log["val_mae"][best],
               worst_client_mae=max(clients) if clients else None,                     # E2
               round_to_threshold=reached_round, threshold_reached=reached,             # E3
               total_bytes=sum(log["bytes"]), manifest_sha256=manifest["sha256"])       # F8, PT11
    return _clean(out)


def centralized_run(folder, cfg: dict, ds: Dataset, seed: int) -> dict:
    """One centralized training (tuning M2, baseline E5, FD001 check). cfg['model'] says which model.
    Writes epochs.json (train_centralized's history), best.pt, test_predictions.csv."""
    folder = Path(folder)
    model, hist = train_centralized(cfg, ds, seed)
    (folder / "epochs.json").write_text(json.dumps(_clean(hist)), encoding="utf8")
    out = save_model_outputs(folder, model, ds, cfg)
    out.update(best_epoch=hist["best_epoch"], best_val_mae=hist["val_mae"][hist["best_epoch"]])
    return _clean(out)


def local_run(folder, cfg: dict, ds: Dataset, manifest: dict, client: int, seed: int) -> dict:
    """Local-only baseline (E5): one client trains alone on its own training engines, early-stops on its own
    validation engines, and is scored on the full test set like every other run."""
    K = cfg["fl"]["K"]
    tr = client_rows(ds.engine_train, manifest["train"], K)[client]
    va = client_rows(ds.engine_val, manifest["val"], K)[client]
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"client {client} has no training or validation windows; it cannot early-stop alone")
    mine = replace(ds, X_train=ds.X_train[tr], y_train=ds.y_train[tr], engine_train=ds.engine_train[tr],
                   X_val=ds.X_val[va], y_val=ds.y_val[va], engine_val=ds.engine_val[va])
    out = centralized_run(folder, cfg, mine, seed)
    out.update(client=client, manifest_sha256=manifest["sha256"])
    return out


def probe_run(folder, cfg: dict, ds: Dataset, source, seed: int) -> dict:
    """Head-refit probe (E6, diagnostic only): the main run's best global model, GRU frozen, fresh head trained on
    all training windows with early stopping on validation. Writes epochs.json (head training), best.pt,
    test_predictions.csv; records which run it came from and that run's test MAE.
    source=None is the control (E20): the same refit on an untrained GRU initialised from `seed`, which shows how far
    a head trained on pooled data gets without learned features."""
    folder = Path(folder)
    m = cfg["model"]
    torch.manual_seed(seed)
    model = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"]).to(cfg["train"].get("device", "cpu"))
    if source is not None:
        source = Path(source)
        model.load_state_dict(torch.load(source / "best.pt", map_location=next(model.parameters()).device))
    refit, hist = refit_head(model, cfg, ds, seed)
    (folder / "epochs.json").write_text(json.dumps(_clean(hist)), encoding="utf8")
    out = save_model_outputs(folder, refit, ds, cfg)
    src = None if source is None else json.loads((source / "final.json").read_text(encoding="utf8"))
    out.update(best_epoch=hist["best_epoch"], best_val_mae=hist["val_mae"][hist["best_epoch"]],
               source_run=None if source is None else f"main/{source.name}",
               source_test_mae=None if src is None else src["test"]["mae"], diagnostic=True)
    return _clean(out)


# ---------- command line ----------

FEDERATED = ("pilot", "musel", "main")
CHOSEN_LATER = {"fl.E", "fl.R", "fl.lr", "fl.mu"}       # pilot (F7) and mu selection (F5) add these to base.yaml


def load_config(path, overrides: list[str]) -> dict:
    """base.yaml plus `section.key=value` overrides (value parsed as YAML: 64 -> int, 0.01 -> float).
    Only existing keys can change, so a typo fails loudly. Exceptions: `model.*` (exists once tuning picks the
    model, M2) and CHOSEN_LATER (exist once the pilot and mu selection are frozen)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    for item in overrides:
        key, sep, value = item.partition("=")
        parts = key.split(".")
        if not sep or len(parts) < 2:
            raise ValueError(f"--set {item!r}: expected section.key=value")
        if parts[0] == "model":
            cfg.setdefault("model", {})
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"--set {key}: no section {part!r} in the config")
            node = node[part]
        if parts[-1] not in node and parts[0] != "model" and key not in CHOSEN_LATER:
            raise ValueError(f"--set {key}: no such key in the config (typo?)")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def allowed_seeds(cfg: dict, group: str) -> list[int]:
    """Seeds each group may use: choices are made on seeds 101-103 only, results reported on 1-5 (F5, F7, M6)."""
    s, t = cfg["seeds"], cfg["tuning"]
    return {"tune": t["seeds_first_pass"] + t["seeds_confirm"], "central": s["main"], "pilot": s["pilot"],
            "musel": s["selection"], "main": s["main"], "local": s["main"], "probe": s["main"],
            "probectl": s["main"]}[group]


def name_params(group: str, args, cfg: dict) -> dict:
    """The settings run_name needs, read from the resolved config, so the folder name always matches what ran."""
    p = {"seed": args.seed, "algo": args.algo, "H": args.H, "client": args.client, "fd": cfg["data"]["fd"]}
    if group == "tune":
        p.update(hidden=cfg["model"].get("hidden"), layers=cfg["model"].get("layers"), lr=cfg["model"].get("lr"))
    elif group == "pilot":
        p.update(E=cfg["fl"].get("E"), lr=cfg["fl"].get("lr"))
    elif group == "musel":
        p.update(mu=cfg["fl"].get("mu"))
    return {k: v for k, v in p.items() if v is not None}


def check_args(group: str, args, cfg: dict) -> None:
    if args.seed not in allowed_seeds(cfg, group):
        raise ValueError(f"seed {args.seed} is not allowed in group {group!r}; allowed: {allowed_seeds(cfg, group)}")
    missing = [k for k in ("hidden", "layers", "lr", "dropout") if k not in cfg.get("model", {})]
    if missing:
        raise ValueError(f"config has no model.{missing[0]}: tuning must freeze it, or pass --set model.{missing[0]}=...")
    if group in FEDERATED:
        if args.algo not in ("fedavg", "fedprox", "fedla"):
            raise ValueError(f"group {group!r} needs --algo fedavg, fedprox or fedla")
        if group == "pilot" and args.algo != "fedavg":
            raise ValueError("the pilot runs fedavg only (F7)")
        if group == "musel" and args.algo != "fedprox":
            raise ValueError("mu selection runs fedprox only (F5)")
        needed = ("E", "R", "lr", "mu") if args.algo == "fedprox" else ("E", "R", "lr")   # pilot runs before F5
        missing = [k for k in needed if k not in cfg["fl"]]
        if missing:
            raise ValueError(f"config has no fl.{missing[0]}: the pilot / mu selection must freeze it (F5, F7)")
    if group == "probe" and args.algo not in ("fedavg", "fedprox", "fedla"):
        raise ValueError("a probe names the main run it refits: --algo fedavg, fedprox or fedla")
    if group == "probectl" and (args.algo is not None or args.H is not None):
        raise ValueError("the probe control refits an untrained GRU: no --algo, no --H")
    if group in FEDERATED + ("local", "probe") and args.H is None:
        raise ValueError(f"group {group!r} needs --H")
    if (group == "local") != (args.client is not None):
        raise ValueError("--client is required for local runs and only for local runs")


def central_threshold(runs_root, cfg: dict) -> float:
    """E3: eval.threshold (1.10) x mean best validation MAE of the 5 finished centralized runs on this dataset."""
    folders = [Path(runs_root) / "central" / run_name("central", {"fd": cfg["data"]["fd"], "seed": s})
               for s in cfg["seeds"]["main"]]
    missing = [str(f) for f in folders if not (f / "DONE").exists()]
    if missing:
        raise FileNotFoundError(f"the rounds-to-110% threshold (E3) needs the finished central runs first: {missing}")
    maes = [json.loads((f / "final.json").read_text(encoding="utf8"))["best_val_mae"] for f in folders]
    return cfg["eval"]["threshold"] * float(np.mean(maes))


def data_hash(cfg: dict) -> str:
    """One SHA-256 over the three raw NASA files this run read (P4), to prove which data produced it."""
    d, h = cfg["data"], hashlib.sha256()
    for part in ("train", "test", "RUL"):
        h.update((Path(d["raw_dir"]) / f"{part}_{d['fd']}.txt").read_bytes())
    return h.hexdigest()


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(prog="python -m rulfl.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=["tune", "central", "pilot", "musel", "main", "local", "probe", "probectl"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--algo")
    ap.add_argument("--H", type=float)
    ap.add_argument("--client", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--allow-dirty", action="store_true", help="development only, never for a registered run")
    args = ap.parse_args(argv)
    group = args.group

    cfg = load_config(args.config, args.set)
    check_args(group, args, cfg)
    name = run_name(group, name_params(group, args, cfg))
    commit = git_commit(allow_dirty=args.allow_dirty)                     # rule 3, checked before any work
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())

    manifest = threshold = None                                           # fail fast: before any folder exists
    if group in FEDERATED + ("local",):
        manifest = load_manifest(manifest_path(cfg["partition"]["manifests_dir"], args.H, args.seed))
    if group in FEDERATED:
        threshold = central_threshold(args.runs_dir, cfg)
    source = Path(args.runs_dir) / "main" / name                          # probe: the main run it refits (E6)
    if group == "probe" and not (source / "DONE").exists():
        raise FileNotFoundError(f"the head-refit probe needs the finished main run {source}")

    def work(folder: Path) -> None:
        torch.set_num_threads(cfg["train"]["threads"])                    # M5
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        ds = build_dataset(cfg)
        if group in FEDERATED:
            out = federated_run(folder, cfg, ds, manifest, args.algo, args.seed, threshold)
        elif group in ("probe", "probectl"):
            out = probe_run(folder, cfg, ds, source if group == "probe" else None, args.seed)
        elif group == "local":
            out = local_run(folder, cfg, ds, manifest, args.client, args.seed)
        else:
            out = centralized_run(folder, cfg, ds, args.seed)
        out.update(group=group, name=name, seed=args.seed, algo=args.algo, H=args.H, client=args.client,
                   git_commit=commit, git_dirty=dirty, data_hash=data_hash(cfg),
                   device=cfg["train"].get("device", "cpu"), torch_version=torch.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   seconds=time.perf_counter() - start)
        (folder / "final.json").write_text(json.dumps(_clean(out), indent=1, allow_nan=False), encoding="utf8")

    folder, ran = in_run_folder(args.runs_dir, group, name, cfg, work)
    print(("DONE " if ran else "SKIPPED (already done) ") + str(folder))
    return folder


if __name__ == "__main__":
    main()
