"""Batch tools around the runner (DECISIONS E15).

    python -m rulfl.batch plan tune  > jobs_tune.txt     every job of one stage, one runner command per line
    python -m rulfl.batch run jobs_tune.txt --jobs 3     run them, N processes at a time (each uses 3 threads, M5)
    python -m rulfl.batch index                          rebuild results/index.csv from every finished run

A job line is what follows `python -m rulfl.run`, e.g. `main --seed 3 --algo fedla --H 0.5`.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def plan(group: str, cfg: dict) -> list[str]:
    """Every job of one stage, straight from the config, so nobody types 75 commands by hand."""
    s, t, f = cfg["seeds"], cfg["tuning"], cfg["fl"]
    if group == "tune":                                       # M2 screen on seed 1; the top-2 rerun is chosen by hand
        return [f"tune --seed {seed} --set model.hidden={h} --set model.layers={l} --set model.lr={lr:g} "
                f"--set model.dropout={t['dropout_if_2_layers'] if l == 2 else 0.0}"
                for seed in t["seeds_first_pass"] for h in t["hidden"] for l in t["layers"] for lr in t["lr"]]
    if group == "central":                                    # E5 baseline; also sets E3's threshold
        return [f"central --seed {seed}" for seed in s["main"]]
    if group == "pilot":                                      # F7: FedAvg only, lr relative to the tuned lr
        p = f["pilot"]
        return [f"pilot --seed {seed} --algo fedavg --H {float(H)} --set fl.E={E} "
                f"--set fl.lr={cfg['model']['lr'] * k:g} --set fl.R={p['R']}"
                for seed in s["pilot"] for H in p["H"] for E in p["E"] for k in p["lr_factors"]]
    if group == "musel":                                      # F5: FedProx at H = 1
        return [f"musel --seed {seed} --algo fedprox --H 1.0 --set fl.mu={mu:g}"
                for seed in s["selection"] for mu in f["mu_grid"]]
    if group == "main":                                       # the 75 paired runs
        return [f"main --seed {seed} --algo {algo} --H {float(H)}"
                for seed in s["main"] for H in cfg["partition"]["H_grid"] for algo in ("fedavg", "fedprox", "fedla")]
    if group == "local":                                      # E5: each client alone, at the two extremes
        return [f"local --seed {seed} --H {H} --client {k}"
                for seed in s["main"] for H in (0.0, 1.0) for k in range(f["K"])]
    raise ValueError(f"no plan for group {group!r}")


def job_succeeded(returncode: int, stdout: str) -> bool:
    """A job succeeded if the runner's last line is DONE or SKIPPED: it prints that only after the run folder is
    final. Not the exit code: on Windows, cuDNN crashes while the process exits after training a 2-layer GRU with
    dropout (0xC0000409), after everything is saved (M8). `returncode` is kept for the log only."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].startswith(("DONE ", "SKIPPED"))


def run_jobs(lines: list[str], jobs: int, extra: list[str] = ()) -> list[str]:
    """Run each line as `python -m rulfl.run <line> <extra>`, `jobs` at a time. A failed job doesn't stop the rest
    (its folder is left as .invalid); returns the failed lines so they can be rerun with the same seed."""
    def one(line: str) -> bool:
        p = subprocess.run([sys.executable, "-m", "rulfl.run", *shlex.split(line), *extra],
                           stdout=subprocess.PIPE, text=True)            # stderr (tracebacks) still shows live
        print(p.stdout, end="", flush=True)
        return job_succeeded(p.returncode, p.stdout)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        ok = list(pool.map(one, lines))
    return [line for line, good in zip(lines, ok) if not good]


COLUMNS = ["group", "name", "seed", "algo", "H", "client", "test_mae", "test_rmse", "test_nasa", "best_val_mae",
           "best_round", "best_epoch", "worst_client_mae", "round_to_threshold", "threshold_reached", "total_bytes",
           "manifest_sha256", "git_commit", "git_dirty", "seconds"]


def build_index(runs_root, out) -> Path:
    """One row per finished run (folders with DONE), rebuilt from scratch every time, never appended (E15)."""
    rows = []
    for done in sorted(Path(runs_root).glob("*/*/DONE")):
        final = json.loads((done.parent / "final.json").read_text(encoding="utf8"))
        for k, v in final.pop("test", {}).items():
            final[f"test_{k}"] = v
        rows.append({c: "" if final.get(c) is None else final[c] for c in COLUMNS})
    rows.sort(key=lambda r: (r["group"], r["name"]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rulfl.batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("group")
    p.add_argument("--config", default="configs/base.yaml")
    r = sub.add_parser("run")
    r.add_argument("file")
    r.add_argument("--jobs", type=int, default=1)
    i = sub.add_parser("index")
    i.add_argument("--runs-dir", default="runs")
    i.add_argument("--out", default="results/index.csv")
    args, extra = ap.parse_known_args(argv)                   # anything unknown is passed to every run (run only)

    if args.cmd == "plan":
        print("\n".join(plan(args.group, yaml.safe_load(Path(args.config).read_text(encoding="utf8")))))
    elif args.cmd == "run":
        lines = [ln.strip() for ln in Path(args.file).read_text(encoding="utf8").splitlines() if ln.strip()]
        failed = run_jobs(lines, args.jobs, extra)
        print(f"{len(lines) - len(failed)} ok, {len(failed)} failed" + "".join(f"\n  FAILED: {x}" for x in failed))
        sys.exit(1 if failed else 0)
    else:
        print(build_index(args.runs_dir, args.out))


if __name__ == "__main__":
    main()
