"""Every number and figure in the report, generated from run folders only (CLAUDE.md rule 6, DECISIONS E8, E15).

    python -m rulfl.analysis            reads runs/, writes results/ (tables .csv, summary.md, figures .png)

Tests follow the pre-registration (docs/PREREGISTRATION.md) and E7: differences are (method - FedAvg) in test MAE,
paired by (H, seed); negative = method better.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rulfl.eval.stats import paired_diff_bootstrap_ci, wilcoxon_signed_rank_exact_one_sided

ALGOS = ("fedavg", "fedprox", "fedla")
STAGES = ("near_failure", "mid", "healthy")


def load_runs(runs_root, group: str) -> list[dict]:
    """final.json of every finished run (folder has DONE) in one group."""
    return [json.loads((d.parent / "final.json").read_text(encoding="utf8"))
            for d in sorted(Path(runs_root).glob(f"{group}/*/DONE"))]


def by_key(runs: list[dict]) -> dict:
    """(algo, H, seed) -> run."""
    return {(r["algo"], float(r["H"]), int(r["seed"])): r for r in runs}


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values. {'P1': 0.01, 'P2': 0.2} -> {'P1': 0.02, 'P2': 0.2}."""
    names = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * pvalues[n]))
        out[n] = running
    return out


def paired(main: dict, algo: str, Hs, seeds, metric=lambda r: r["test"]["mae"]):
    """Differences (algo - FedAvg) for every (H, seed) pair present for both."""
    a, b = [], []
    for H in Hs:
        for s in seeds:
            if (algo, H, s) in main and ("fedavg", H, s) in main:
                a.append(metric(main[(algo, H, s)]))
                b.append(metric(main[("fedavg", H, s)]))
    return np.array(a), np.array(b)


def main_table(main: dict, Hs, seeds) -> list[dict]:
    """Per method x H: mean and sd over seeds of the headline and secondary metrics."""
    rows = []
    for algo in ALGOS:
        for H in Hs:
            rs = [main[(algo, H, s)] for s in seeds if (algo, H, s) in main]
            if not rs:
                continue
            col = lambda f: np.array([f(r) for r in rs], dtype=float)
            mae = col(lambda r: r["test"]["mae"])
            reached = [r for r in rs if r["threshold_reached"]]
            rows.append({
                "algo": algo, "H": H, "n": len(rs),
                "test_mae_mean": mae.mean(), "test_mae_sd": mae.std(ddof=1) if len(rs) > 1 else 0.0,
                "test_rmse_mean": col(lambda r: r["test"]["rmse"]).mean(),
                "test_nasa_mean": col(lambda r: r["test"]["nasa"]).mean(),
                "worst_client_val_mae_mean": np.nanmean(col(lambda r: np.nan if r["worst_client_mae"] is None
                                                            else r["worst_client_mae"])),
                "rounds_to_110_median_reached": (float(np.median([r["round_to_threshold"] for r in reached]))
                                                 if reached else None),
                "n_reached_110": len(reached),
                "total_bytes_mean": col(lambda r: r["total_bytes"]).mean(),
                **{f"mae_{st}_mean": np.nanmean(col(lambda r, st=st: np.nan if r["life_stage_mae"][st] is None
                                                     else r["life_stage_mae"][st])) for st in STAGES},
            })
    return rows


def paired_rows(main: dict, Hs, seeds) -> list[dict]:
    """S2: per H and per comparison method, mean difference, bootstrap 95% CI, one-sided Wilcoxon (exploratory)."""
    rows = []
    for algo in ("fedprox", "fedla"):
        for H in Hs:
            a, b = paired(main, algo, [H], seeds)
            if len(a) == 0:
                continue
            mean, lo, hi = paired_diff_bootstrap_ci(a, b)
            rows.append({"algo": algo, "H": H, "n_pairs": len(a), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p_less": wilcoxon_signed_rank_exact_one_sided(a - b, "less")})
    return rows


def primary(main: dict, Hs, seeds) -> list[dict]:
    """P1 (FedLA) and P2 (FedProx) vs FedAvg pooled over H >= 0.5, one-sided Wilcoxon, Holm over the two."""
    pooled_H = [H for H in Hs if H >= 0.5]
    res = {}
    for name, algo in (("P1", "fedla"), ("P2", "fedprox")):
        a, b = paired(main, algo, pooled_H, seeds)
        mean, lo, hi = paired_diff_bootstrap_ci(a, b)
        res[name] = {"hypothesis": name, "algo": algo, "n_pairs": len(a), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                     "p_less": wilcoxon_signed_rank_exact_one_sided(a - b, "less")}
    adj = holm({k: v["p_less"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = adj[k]
        v["significant_at_0.05"] = adj[k] < 0.05
    return list(res.values())


def premise(main: dict, seeds) -> dict:
    """S1: FedAvg at H = 1 minus FedAvg at H = 0, paired by seed."""
    a = np.array([main[("fedavg", 1.0, s)]["test"]["mae"] for s in seeds if ("fedavg", 1.0, s) in main
                  and ("fedavg", 0.0, s) in main])
    b = np.array([main[("fedavg", 0.0, s)]["test"]["mae"] for s in seeds if ("fedavg", 1.0, s) in main
                  and ("fedavg", 0.0, s) in main])
    mean, lo, hi = paired_diff_bootstrap_ci(a, b)
    return {"comparison": "fedavg H=1 minus H=0", "n_pairs": len(a), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi}


def baselines(runs_root) -> list[dict]:
    """Centralized (FD004) and local-only per H: mean test MAE."""
    rows = []
    cen = [r for r in load_runs(runs_root, "central") if r["name"].startswith("FD004")]
    if cen:
        m = np.array([r["test"]["mae"] for r in cen])
        rows.append({"baseline": "centralized", "H": None, "n": len(m), "test_mae_mean": m.mean(),
                     "test_mae_sd": m.std(ddof=1) if len(m) > 1 else 0.0})
    loc = load_runs(runs_root, "local")
    for H in sorted({float(r["H"]) for r in loc}):
        m = np.array([r["test"]["mae"] for r in loc if float(r["H"]) == H])
        rows.append({"baseline": "local_only", "H": H, "n": len(m), "test_mae_mean": m.mean(),
                     "test_mae_sd": m.std(ddof=1) if len(m) > 1 else 0.0})
    return rows


def probe_rows(runs_root, main: dict) -> list[dict]:
    """S4: per method x H, mean test MAE of the federated model and of the same model after the head refit."""
    probe = by_key(load_runs(runs_root, "probe"))
    rows = []
    for algo in ALGOS:
        for H in sorted({k[1] for k in probe if k[0] == algo}):
            keys = [k for k in probe if k[0] == algo and k[1] == H and k in main]
            rows.append({"algo": algo, "H": H, "n": len(keys),
                         "test_mae_federated": float(np.mean([main[k]["test"]["mae"] for k in keys])),
                         "test_mae_head_refit": float(np.mean([probe[k]["test"]["mae"] for k in keys]))})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def skew_by_H(manifests_dir, seeds) -> dict:
    """Mean partition skew per H over the main seeds, read from the committed manifests (PT11)."""
    out = {}
    for p in Path(manifests_dir).glob("H*_s*.json"):
        m = json.loads(p.read_text(encoding="utf8"))
        if int(m["seed"]) in seeds:
            out.setdefault(float(m["H"]), []).append(m["skew"])
    return {H: float(np.mean(v)) for H, v in sorted(out.items())}


def figures(out: Path, table: list[dict], base: list[dict], probe: list[dict], skew: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    colors = {"fedavg": "#1f77b4", "fedprox": "#ff7f0e", "fedla": "#2ca02c"}
    names = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedla": "FedLA"}
    cen = next((b for b in base if b["baseline"] == "centralized"), None)

    fig, ax = plt.subplots(figsize=(6, 4))
    for algo in ALGOS:
        rs = [r for r in table if r["algo"] == algo]
        ax.errorbar([r["H"] for r in rs], [r["test_mae_mean"] for r in rs], yerr=[r["test_mae_sd"] for r in rs],
                    marker="o", capsize=3, label=names[algo], color=colors[algo])
    if cen:
        ax.axhline(cen["test_mae_mean"], color="grey", ls="--", label="Centralized")
    ax.set_xlabel("Label skew H"); ax.set_ylabel("Test MAE (cycles)"); ax.set_title("Test MAE vs label skew (mean ± sd, 5 seeds)")
    ax.legend(); fig.tight_layout(); p = out / "fig_mae_vs_H.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for ax, st in zip(axes, STAGES):
        for algo in ALGOS:
            rs = [r for r in table if r["algo"] == algo]
            ax.plot([r["H"] for r in rs], [r[f"mae_{st}_mean"] for r in rs], marker="o", label=names[algo],
                    color=colors[algo])
        ax.set_title({"near_failure": "Near failure (RUL < 50)", "mid": "Mid (50-124)", "healthy": "Healthy (>= 125)"}[st])
        ax.set_xlabel("Label skew H")
    axes[0].set_ylabel("Test MAE (cycles)"); axes[0].legend()
    fig.tight_layout(); p = out / "fig_life_stage.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if probe:
        fig, ax = plt.subplots(figsize=(6, 4))
        for algo in ALGOS:
            rs = [r for r in probe if r["algo"] == algo]
            ax.plot([r["H"] for r in rs], [r["test_mae_federated"] for r in rs], marker="o", color=colors[algo],
                    label=f"{names[algo]} federated")
            ax.plot([r["H"] for r in rs], [r["test_mae_head_refit"] for r in rs], marker="s", ls=":", color=colors[algo],
                    label=f"{names[algo]} head refit")
        if cen:
            ax.axhline(cen["test_mae_mean"], color="grey", ls="--", label="Centralized")
        ax.set_xlabel("Label skew H"); ax.set_ylabel("Test MAE (cycles)")
        ax.set_title("Head-refit probe (diagnostic)"); ax.legend(fontsize=7)
        fig.tight_layout(); p = out / "fig_probe.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if skew:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(list(skew), list(skew.values()), marker="o", color="black")
        ax.set_xlabel("Label skew H"); ax.set_ylabel("Mean client W1 distance (cycles)")
        ax.set_title("Partition skew by H (seeds 1-5)")
        fig.tight_layout(); p = out / "fig_skew.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def summary_md(table, prim, prem, per_h, base, probe, skew) -> str:
    L = ["# Results summary (generated by `python -m rulfl.analysis`; do not edit by hand)", ""]
    L += ["## Main results: test MAE (cycles), mean ± sd over seeds", "",
          "| H | " + " | ".join(a for a in ALGOS) + " |", "|---|" + "---|" * len(ALGOS)]
    for H in sorted({r["H"] for r in table}):
        cells = []
        for a in ALGOS:
            r = next((r for r in table if r["algo"] == a and r["H"] == H), None)
            cells.append("n/a" if r is None else f"{r['test_mae_mean']:.2f} ± {r['test_mae_sd']:.2f}")
        L.append(f"| {H} | " + " | ".join(cells) + " |")
    L += ["", "## Primary hypotheses (pooled H ≥ 0.5, one-sided Wilcoxon, Holm over P1 and P2)", "",
          "| | method | pairs | mean diff vs FedAvg | 95% CI | p | p (Holm) | significant |", "|---|---|---|---|---|---|---|---|"]
    for p in prim:
        L.append(f"| {p['hypothesis']} | {p['algo']} | {p['n_pairs']} | {fmt(p['mean_diff'], 3)} | "
                 f"[{fmt(p['ci_lo'], 3)}, {fmt(p['ci_hi'], 3)}] | {fmt(p['p_less'], 4)} | {fmt(p['p_holm'], 4)} | "
                 f"{'yes' if p['significant_at_0.05'] else 'no'} |")
    L += ["", f"**Premise (S1):** FedAvg test MAE at H = 1 minus H = 0: {fmt(prem['mean_diff'], 3)} cycles, "
          f"95% CI [{fmt(prem['ci_lo'], 3)}, {fmt(prem['ci_hi'], 3)}] ({prem['n_pairs']} seeds).", ""]
    L += ["## Per-H differences vs FedAvg (S2, exploratory)", "",
          "| method | H | mean diff | 95% CI | Wilcoxon p (less) |", "|---|---|---|---|---|"]
    for r in per_h:
        L.append(f"| {r['algo']} | {r['H']} | {fmt(r['mean_diff'], 3)} | [{fmt(r['ci_lo'], 3)}, {fmt(r['ci_hi'], 3)}] | "
                 f"{fmt(r['wilcoxon_p_less'], 4)} |")
    L += ["", "## Baselines", "", "| baseline | H | runs | test MAE |", "|---|---|---|---|"]
    for b in base:
        L.append(f"| {b['baseline']} | {b['H'] if b['H'] is not None else '-'} | {b['n']} | "
                 f"{b['test_mae_mean']:.2f} ± {b['test_mae_sd']:.2f} |")
    if probe:
        L += ["", "## Head-refit probe (S4, diagnostic only)", "",
              "| method | H | federated test MAE | after head refit |", "|---|---|---|---|"]
        for r in probe:
            L.append(f"| {r['algo']} | {r['H']} | {r['test_mae_federated']:.2f} | {r['test_mae_head_refit']:.2f} |")
    if skew:
        L += ["", "## Partition skew (mean W1, cycles)", "", "| H | skew |", "|---|---|"]
        L += [f"| {H} | {v:.2f} |" for H, v in skew.items()]
    L += ["", "Full tables: `main_table.csv`, `paired_per_H.csv`, `primary.csv`, `baselines.csv`, `probe.csv`."]
    return "\n".join(L) + "\n"


def analyse(runs_root, out_dir, manifests_dir, seeds=(1, 2, 3, 4, 5), with_figures=True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main = by_key(load_runs(runs_root, "main"))
    if not main:
        raise FileNotFoundError(f"no finished main-grid runs under {runs_root}/main")
    Hs = sorted({k[1] for k in main})
    seeds = [s for s in seeds if any(k[2] == s for k in main)]
    table, per_h = main_table(main, Hs, seeds), paired_rows(main, Hs, seeds)
    prim, prem = primary(main, Hs, seeds), premise(main, seeds)
    base, probe = baselines(runs_root), probe_rows(runs_root, main)
    skew = skew_by_H(manifests_dir, seeds)
    write_csv(table, out / "main_table.csv")
    write_csv(per_h, out / "paired_per_H.csv")
    write_csv(prim, out / "primary.csv")
    write_csv([prem], out / "premise.csv")
    write_csv(base, out / "baselines.csv")
    write_csv(probe, out / "probe.csv")
    if with_figures:
        figures(out, table, base, probe, skew)
    (out / "summary.md").write_text(summary_md(table, prim, prem, per_h, base, probe, skew), encoding="utf8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rulfl.analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--manifests", default="manifests")
    args = ap.parse_args(argv)
    print(analyse(args.runs_dir, args.out, args.manifests) / "summary.md")


if __name__ == "__main__":
    main()
