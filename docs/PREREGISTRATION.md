# Pre-registration: federated RUL prediction under label skew (FD004)

Approved by Fidel on 2026-11-17, committed before the first main-grid run. The commit timestamp is the proof of order.
Tests follow DECISIONS E7 and `src/rulfl/eval/stats.py`. A difference is (method − FedAvg) in **test MAE** (cycles),
paired by (H, seed): negative means the method is better.

## Design (frozen before this file)
- Data FD004; model GRU 128 × 2, lr 0.001, dropout 0.2 (M7); K = 5 simulated clients; E, R, federated lr from the
  pilot (F7) and μ from μ selection (F5), both on validation data and seeds 101–103 only; values are in `configs/base.yaml`
  at this commit.
- Main grid: FedAvg, FedProx, FedLA × H ∈ {0, 0.25, 0.5, 0.75, 1} × seeds 1–5 = 75 runs on the committed manifests;
  the three methods of an (H, seed) pair share one manifest and one machine.

## Primary hypotheses
- **P1, FedLA helps under skew.** FedLA test MAE < FedAvg test MAE, pooled over H ≥ 0.5 (15 pairs).
  One-sided exact Wilcoxon signed-rank, alternative "less".
- **P2, FedProx barely helps** (replication of Ozdemir et al.). Same test, FedProx vs FedAvg. We expect no meaningful
  improvement; the result is reported either way.
- Holm correction over P1 and P2, family-wise α = 0.05.

## Secondary analyses (descriptive / exploratory; no significance claims)
- **S1 premise:** FedAvg test MAE rises with H: paired difference (H = 1) − (H = 0) over seeds 1–5 with a 95% bootstrap interval.
- **S2 per H:** for each H, mean difference vs FedAvg with a paired bootstrap 95% percentile interval (n_boot = 10000, seed = 0)
  and a per-H one-sided Wilcoxon, labelled exploratory (with 5 pairs the smallest possible p is 1/32 ≈ 0.031).
- **S3 where the damage is:** test MAE by life stage (near failure < 50, mid 50–124, healthy ≥ 125) per method and H (E4).
- **S4 features or head:** head-refit probe on each main run's best global model (E6). Diagnostic only, labelled as such.
- Also reported: RMSE, NASA score, worst-client validation MAE (E2), rounds to 110% of centralized validation MAE
  (threshold 9.894; runs that never reach it are counted "not reached", never dropped, E3), bytes communicated.
- Baselines: centralized (Gate 1, seeds 1–5) and local-only at H = 0 and H = 1 (E5).

## Rules
- No run is dropped or rerun because of its result. A crashed run is rerun with the same seed and manifest.
- Null results are reported as null results.
- Wilcoxon: zero differences dropped, differences rounded to 10 decimals before ranking, tied magnitudes get mid-ranks.

## Deviations disclosed in advance
- Tuning ran on CPU; everything from Gate 1 on runs on the GPU (DECISIONS M8).
- On Windows, cuDNN crashes at process exit after training; runs are judged by their DONE marker (M9).
- FedLA bins and clip are fixed, not selected on validation (LA1, LA2).
- Test error was first looked at in Gate 1 (centralized baseline), never during tuning, the pilot or μ selection.
