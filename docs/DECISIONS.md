# Decisions

Choices the proposal left open, decided before the code that uses them is built.
Derived from our own design (CLAUDE.md); the reference repo's list was used only as a cross-check.
Once a section is committed it is frozen: changing a decision needs a new entry here saying what changed and why,
made before any registered run uses it. Numbers that live in code are in `configs/base.yaml`.

Status: Data ✅ · Model ✅ · Partition ✅ · FL (FedAvg / FedProx) ✅ · FedLA ✅ · Evaluation and analysis ✅

## Project-level

| # | Decision | Reason |
|---|---|---|
| P1 | Python 3.12; every package pinned in `requirements.lock` (torch is the CPU build) | One environment for everyone; 3.12 is the version the pinned stack was tested on |
| P2 | A package is added to `requirements.txt` only when code needs it, with the reason in a comment | Every dependency can be justified (CLAUDE.md rule 7) |
| P3 | Tests run on a committed, hashed slice of the real NASA files (`tests/data/`, made by `tests/make_slice.py`), not synthetic data. Deviation from the CLAUDE.md plan (`tests/fakes.py`) | Real data has quirks a generator gets wrong (we found three: settings jitter, near-constant sensors, lifetimes > 125). CI can run real-format tests without the full data |
| P4 | Raw NASA files are never committed; their SHA-256 is in `data/raw/CHECKSUMS.txt` | Everyone checks they have byte-identical inputs |
| P5 | matplotlib 3.11.2 added (figures in `rulfl.analysis`), approved by Fidel 2026-11-17. Installed with the lock files as constraints, so no existing pin changed; both lock files regenerated with `pip freeze` | Figures for the report; rule 7 |

## Data pipeline

| # | Decision | Reason |
|---|---|---|
| D1 | Validation = 20% of FD004 training engines (50 of 249), chosen once with `split_seed: 0`, fixed forever. Split by engine before windowing | ~40 train engines per client at K = 5; ~10 validation engines per client for worst-client MAE |
| D2 | Operating condition of a row = its settings rounded to whole numbers: altitude, Mach × 100, throttle (key like `42_84_100`). The set of conditions is read from training-split rows | Real jitter is ≤ 0.008, so rounding is exact; no clustering library or random init needed. Whole numbers avoid FD001's jitter around Mach 0 splitting into "0.0" and "-0.0" (found by a test) |
| D3 | Each sensor is z-scored per operating condition, using mean and std from training-split rows only, pooled over all clients | Same input features at every H and every arm, so differences come only from label skew. Limitation for the report: a real federation could not pool raw statistics |
| D4 | Drop a sensor column only if its std is zero (≤ 1e-8) within every operating condition, measured on training-split rows; the same columns are then dropped from train, validation and test. Raw files are never modified | Such a sensor only encodes the condition (already in the one-hot, D5) and would divide by zero in per-condition z-scoring. On real FD004 this drops s1, s5, s18, s19 (NASA numbering) and keeps 17. A first proposal (std < 0.1% of mean) was rejected after checking real data: it would have dropped s2, s8, s13, which trend with wear. Where a kept sensor is constant in one condition only (s16 in 4 of 6 FD004 conditions), its normalised value in that condition is set to 0 (no division by zero) |
| D5 | Input per cycle = kept normalised sensors + one-hot of the operating condition | Normalisation removes the condition's level; the one-hot keeps which condition the cycle was in |
| D6 | Train and validation windows: every 30-cycle window (stride 1), label `min(T − t, 125)` at the window's last cycle. Validation MAE uses all validation windows | All-window validation MAE is stable; it drives model selection |
| D7 | Test input = last 30 cycles per test engine; engines shorter than 30 are left-padded by repeating their first cycle | A repeated early cycle looks like the healthy engine it is; zero padding would insert fake "average" readings |
| D8 | Test labels capped at 125 for the headline metric; uncapped values kept as `y_test_uncapped` | Comparable to published FD004 results; a model trained on capped labels cannot predict above 125 |

Frozen signature:

```python
build_dataset(cfg: dict) -> Dataset
# Dataset fields:
#   X_train (n, 30, f), y_train (n,), engine_train (n,)   one row per training window
#   X_val, y_val, engine_val                               same for validation engines
#   X_test (n_test_engines, 30, f), y_test, y_test_uncapped
#   engine_test (n_test_engines,)                         test engine id per row, sorted (E15 predictions file)   [added when built]
#   feature_names: list[str]
#   train_engine_lifetimes: dict[int, int]                 engine id -> total cycles (for the partitioner)
#   val_engine_lifetimes: dict[int, int]                   same for validation engines (routing, PT5)   [added when built]
#   scaler: Scaler                                         conditions, kept/dropped sensors, per-condition mean/std [added when built]
```

As built (real FD004): 43,064 train windows (199 engines), 10,964 validation windows (50 engines), 248 test windows,
23 features (17 sensors + 6 condition one-hot). Builds in under a second, so it is rebuilt in memory, not cached.

## Model

| # | Decision | Reason |
|---|---|---|
| M1 | GRU (batch-first) → hidden state at the last time step → head: Linear(hidden, hidden/2) → ReLU → Linear(hidden/2, 1). The head is its own submodule, `model.head` | The head-refit probe freezes `model.gru` and retrains only `model.head`; the boundary must exist in code |
| M2 | Tuned once, centralized, on validation MAE only: hidden {32, 64, 128} × layers {1, 2} × lr {0.001, 0.003} (12 configs, seed 1). Top 2 rerun with seeds 1–3; lower mean wins. Dropout 0.2 between GRU layers when layers = 2, else 0. Then frozen for every arm | Small grid a laptop can run; covers common FD004 sizes. Never retuned on federated results (CLAUDE.md) |
| M3 | Loss = MSE on y/125 (model outputs RUL/125). Adam, batch 256, gradient-norm clip 1.0. Centralized: max 100 epochs, early stop after 10 epochs without validation-MAE improvement, keep best epoch | Targets in 0–1 train more smoothly; clipping guards against GRU gradient spikes; early stopping prevents overfitting |
| M4 | Evaluation (validation and test, every arm) uses one function: prediction = clamp(model(x) × 125, 0, 125). Training loss is unclamped | RUL cannot be negative and labels never exceed 125. Clamping during training would block gradients; a sigmoid output would saturate on the many labels at the cap |
| M5 | All registered runs on CPU, seeded (init, batch order) with `torch.use_deterministic_algorithms(True)`. Revisit after tuning: if the frozen model is too slow on CPU, ALL registered runs move to GPU together, never mixed | Measured on a 12-thread laptop CPU (random data, real shapes, 40k windows): 1.7 s/epoch (32×1) to 10.8 s/epoch (128×2), i.e. ~7–42 min per federated run. Mixing devices would add numerical noise to paired comparisons, and the expected FedAvg/FedProx differences are small. One CPU lock file for everyone. [added when built] CPU thread count also changes results (~7th digit; measured 1 vs 3 vs 12 threads), so every registered run uses the same count: 3 (from the parallel-throughput benchmark), set by the runner |
| M6 | [changes M2's confirmation seeds] The top 2 tuning configs are rerun on seeds 101–103 (`seeds.selection`), not 1–3. Seed 1 is still used for the 12-config screen | Seeds 1–5 are the reported seeds; choosing the model on them would bias the centralized validation MAE that sets E3's 110% threshold. Same rule as F5 for μ: no choice is ever made on a reported seed |
| M7 | [M2 result, frozen] `model: hidden 128, layers 2, lr 0.001, dropout 0.2`. Screen (seed 1, best validation MAE, cycles): 128×1 lr .001 8.980 · 128×2 lr .001 8.985 · 64×1 lr .003 9.023 · 64×2 lr .001 9.068 · 32×2 lr .001 9.083 · 64×1 lr .001 9.202 · 128×1 lr .003 9.220 · 32×2 lr .003 9.344 · 64×2 lr .003 9.405 · 128×2 lr .003 9.509 · 32×1 lr .001 9.572 · 32×1 lr .003 9.958. Confirm (seeds 101–103): 128×2 9.196 / 9.139 / 9.227, mean 9.187; 128×1 9.310 / 9.439 / 9.449, mean 9.399. Runs: `runs/tune/`, commit f90f8e6, clean tree, CPU, 3 threads | Lower mean validation MAE on the confirmation seeds wins (M2, M6); 128×2 is lower on all three seeds. The screen's top 2 differed by 0.005, which is why confirmation exists. Test error was not looked at. Cost on CPU (3 runs in parallel): about 22 s per epoch |
| M8 | [changes M5, the "revisit after tuning" clause] From Gate 1 on, every registered run is on the GPU: `train.device: cuda`, NVIDIA RTX 3050 4 GB Laptop on both run machines, `requirements-gpu.lock` (identical to `requirements.lock` except `torch==2.14.1+cu126`; 2.15.0 only exists as a CUDA 13.2 build, which needs a newer driver than ours). Seeded and deterministic as before: `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set in `rulfl/__init__.py`). The model is built on CPU from the seed, then moved, so start weights match the CPU ones. `predict_rul` and `client_update` follow the model's device; checkpoints are saved as CPU tensors; `final.json` records `device`, `torch_version`, `gpu`. Tests and CI stay on CPU; `tests/test_gpu.py` runs only where CUDA exists. Tuning (M7) ran on CPU and stands: it only chose the model. Gate 1 is rerun on GPU; the CPU Gate 1 runs are not used. All 3 arms of an (H, seed) pair run on the same machine | Measured on the frozen 128×2 model: 1.73 s/epoch on GPU vs 20.4 s on CPU (3 threads), so the full plan with no cuts takes about 2–3 h of machine time instead of about 18 h, which fits the submission deadline. `tests/test_gpu.py`: the same seed gives identical weights twice on GPU for centralized and all 3 FL arms. No device mixing among registered runs (M5's concern); pairing per machine keeps CPU/GPU-model differences out of every comparison |
| M9 | [as built, Windows + GPU] After training a 2-layer GRU with dropout on CUDA (our M7 model), the process crashes while exiting (Windows 0xC0000409, inside cuDNN), after `main()` has returned. `rulfl.batch` therefore judges a job by the runner's last line (`DONE` / `SKIPPED`, printed only once the run folder is final), not by its exit code. Reproduced with a 15-line script (no project code besides `RULModel`); not seen with 1 layer or dropout 0 | Every file and `DONE` are written before the crash; GPU tests show exact repeats and Gate 1 matches the CPU numbers, so results are unaffected. The model is frozen (M7), so the dropout layer stays |

Frozen signatures:

```python
class RULModel(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float): ...
    def forward(self, x: Tensor) -> Tensor:   # x (batch, 30, n_features) -> (batch,)  RUL / 125
    # submodules: self.gru, self.head

predict_rul(model, X, cap) -> np.ndarray       # clamp(model(X) * cap, 0, cap), cap = data.rul_cap (M4)
train_centralized(cfg, ds, seed) -> tuple[RULModel, dict]   # dict: per-epoch train_loss, val_mae, best_epoch
    # [added when built] reads cfg['model'] = {hidden, layers, lr, dropout} (set by tuning, M2) and cfg['train'];
    # best_epoch = 0-based index into the per-epoch lists; the returned model has that epoch's weights
```

## Partition

Rule (agreed in the proposal): sort training engines by lifetime, cut into K blocks; engine in block b goes to
client b with probability H, otherwise to a uniformly random client (so P(client k) = H·1[b = k] + (1 − H)/K).

| # | Decision | Reason |
|---|---|---|
| PT1 | H ∈ {0, 0.25, 0.5, 0.75, 1} | Evenly spaced from IID to extreme skew; enough points to see a trend |
| PT2 | Sort by (lifetime, engine id); K contiguous blocks as equal as possible (199 engines → 40, 40, 40, 40, 39) | Tie rule makes the blocks deterministic |
| PT3 | Partition randomness comes from the run seed (main seeds 1–5). The same (H, seed) partition feeds all three arms | Several partitions per H, so no result rests on one lucky split; identical partitions keep arms paired |
| PT4 | No redraws. If any client gets fewer than 10 training engines, raise an error | Expected ~40 per client, so this should never fire; a silent redraw would hide it if it did |
| PT5 | The 50 validation engines are routed to clients by the same rule, H and seed (their own lifetime blocks). Best-round selection uses all 50; worst-client MAE uses each client's routed ones | Gives every client its own validation set without moving any engine across the fixed train/val split |
| PT6 | Skew = mean over clients of the Wasserstein-1 distance (in cycles) between that client's training RUL labels and all training labels. A test checks the seed-averaged skew strictly increases over the H grid | Required by the proposal; Wasserstein reads as "cycles of label shift", easy to explain; computed with numpy only |
| PT7 | Each (H, seed) partition is written once to `manifests/H{H:.2f}_s{seed}.json` (engine → client for train and val, plus SHA-256), committed, never hand-edited | Every arm and every rerun reads the identical partition (CLAUDE.md rule 4) |
| PT8 | Assignment draw [as built]: in (lifetime, engine id) order, `default_rng(seed)` draws `u = random(n)` then `alt = integers(0, K, n)`; an engine goes to its own block if u < H, else to `alt`. Both vectors are drawn for every engine whatever H is. Blocks follow `np.array_split`: when engines don't divide evenly, the extra engines go to the shortest-lived blocks (7 engines, K = 5 → 2, 2, 1, 1, 1) | One seed fixes one pair of vectors and H only moves the threshold, so partitions at different H share their randomness (common random numbers): differences across H come from H, not from re-rolled luck. An engine in its own block at some H stays there at every larger H. H = 0 is exactly uniform and H = 1 exactly the blocks |
| PT9 | Validation engines are routed with `val_seed(seed)` = first 32-bit word of `SeedSequence([seed, 1])`, over their own lifetime blocks (50 engines → 10 per block). No minimum on validation: PT4 applies to training only | PT5's "same seed" taken literally would give the i-th sorted training engine and the i-th sorted validation engine the same coin flips. Limitation for the report: at mid H a client can get 4–6 validation engines, so worst-client MAE (E2) is noisy |
| PT10 | `assign_clients` keeps its frozen signature and never checks counts. PT4 is `check_min_engines(assignment, K, minimum)`, called by `build_manifest` on training engines with `partition.min_engines_per_client` | Validation routing must not trip a training rule; the frozen signature stays as agreed. Measured risk: at K = 5 (199 engines, ~40 per client) PT4 essentially never fires; at the K = 10 fallback (~20 per client) there is about a 26% chance it fires in the H = 0 manifests alone. Decide before switching to K = 10 whether to lower the minimum |
| PT11 | Manifest file: `H`, `seed`, `K`, `sha256`, `skew`, `n_train_per_client`, `n_val_per_client`, `train`, `val`. `sha256` hashes the assignment only (H, seed, K, train, val, canonical JSON with string keys); `skew` and the counts are diagnostics outside the hash. Runs record the embedded `sha256` as `manifest_sha256` in `final.json` (E8). `load_manifest` rechecks the hash and refuses a hand-edited file; `write_manifest` refuses to replace a different partition. `skew` requires whole-number labels ≥ 0 | `eval/validate.py` accepts a run if its hash matches the file bytes or the embedded field; the embedded one survives Windows line-ending conversion, the byte hash does not. `eval/plots.py` reads `H` and `skew`. Diagnostics are kept out of the hash so recomputing skew can't invalidate a partition. The CDF-sum formula for Wasserstein-1 is exact only on a unit grid |
| PT12 | Manifests are written for every H in `partition.H_grid` × every planned seed (`seeds.main` ∪ `seeds.selection` ∪ `seeds.pilot`: 5 × 8 = 40 files). `make_all` builds them all in memory, checks PT6 on the main seeds (seed-averaged skew strictly increasing over H), and writes only if it holds | Every run, pilot and μ-selection included, finds its partition by (H, seed). A broken knob is caught before any file exists, so nothing half-valid gets committed |

Frozen signatures:

```python
assign_clients(lifetimes: dict[int, int], K: int, H: float, seed: int) -> dict[int, int]   # engine id -> client 0..K-1
skew(labels_by_client: list[np.ndarray], all_labels: np.ndarray) -> float               # mean Wasserstein-1, cycles
# [added when built] src/rulfl/partition/label_skew.py and make_manifests.py
val_seed(seed: int) -> int                                                               # PT9
check_min_engines(assignment: dict[int, int], K: int, minimum: int) -> None              # PT4, PT10; raises
build_manifest(cfg, ds, H, seed, K=None) -> dict                                         # PT11 layout
write_manifest(manifest, out_dir) -> Path;  load_manifest(path) -> dict                  # PT7, PT11
make_all(cfg, ds, out_dir, K=None) -> list[dict]                                         # PT12
```

## FL (FedAvg / FedProx)

| # | Decision | Reason |
|---|---|---|
| F1 | All K clients train every round (full participation) | With K = 5, client sampling would add randomness unrelated to label skew |
| F2 | Each client builds a fresh Adam optimiser every round; batch 256, gradient-norm clip 1.0 (as centralized) | No hidden client state carried across rounds; matches a stateless Flower client (Gate 2) |
| F3 | Server aggregation = average of client weights, weighted by each client's number of training windows | Agreed in the proposal |
| F4 | FedProx client loss = MSE + (μ/2)·Σ(p − g)², g = global weights copied before local training. FedAvg is the same code with μ = 0 | One code path for both arms, so they stay comparable (proposal) |
| F5 | μ chosen once from {0.001, 0.01, 0.1, 1} by lowest mean validation MAE at H = 1, seeds 101–103 | H = 1 is where FedProx should help most (its best shot); selection seeds never overlap the reported seeds 1–5 |
| F6 | Every run does all R rounds (no early stopping); global validation MAE (all validation engines) after every round; the reported model is the best-validation round | Identical round counts for reference and Flower; enables rounds-to-110% |
| F7 | E, R and federated lr set in the pilot: E ∈ {1, 3} × lr ∈ {tuned lr, tuned lr / 3}, R = 60, H ∈ {0, 1}, seeds 101–102, validation only. Pick the setting where FedAvg learns best at H = 0. If FedAvg does not degrade at H = 1, switch to K = 10 (proposal fallback) before the main grid | Settings chosen before any result that matters; the fallback keeps the study meaningful |
| F8 | Bytes communicated per round = 2 × K × n_parameters × 4 (float32 down + up); FedLA adds its one-time histograms | Defines the communication metric exactly |
| F9 | The Flower server is always Flower's plain `FedAvg` (`fraction_train=1.0`, `min_train_nodes=min_available_nodes=K`, `fraction_evaluate=0.0`, `weighted_by_key="num-examples"`), for all three arms. Training goes through a pluggable `train_fn(arrays, client_id, config) -> (new_arrays, n_training_windows)`; the client is stateless and never uses `context.state` | One server for FedAvg, FedProx and FedLA, since μ and sample weights live in the client loss (F4, LA5). Flower's built-in `FedProx` would apply μ a second time, and it subclasses `FedAvg`, so the test checks `type(s) is FedAvg`, not `isinstance` |
| F10 | Evaluation is server-side only (`evaluate_fn`), on all validation engines, after every round including round 0 (the initial weights). The best round is the lowest validation MAE; ties keep the earlier round | Implements F6 the same way as the reference simulator, so Gate 2 compares like with like. Round 0 gives the untrained baseline for every run |
| F11 | Gate 2: Flower must reproduce `run_federated` on the test slice over 2 rounds for FedAvg, FedProx (mu > 0) and FedLA. Compared: per-round validation MAE (absolute 1e-4 cycles), per-client validation MAE (same), `best_state` weights (rtol 1e-5, atol 1e-7), and `bytes` exactly. Round 0 must match to 1e-12. `best_round` is compared only when the best round beats the runner-up by more than the tolerance. Registered runs use the reference simulator; Flower shows the same algorithm runs in a real FL framework | Both engines call the same `client_update`, so only the server's addition order can differ: Flower aggregates replies as they arrive, the simulator always goes client 0..K-1, and float32 addition is not associative. Measured on the slice (2 CPUs, clients effectively serial): validation MAE identical bit for bit, weights within 1.2e-07 (one float32 ULP). The tolerance is headroom for a machine where replies arrive out of order, not an observed gap |
| F12 | `run_flower_federated(cfg, ds, manifest, algo, seed)` (`src/rulfl/flower_app/rul.py`) returns exactly the dict `run_federated` returns (`val_mae`, `client_val_mae`, `bytes`, `best_round`, `best_state`). Weights travel as an ordered list of arrays; the order is `state_keys(cfg, n_features)`, the state-dict key order of a freshly built model | Field-by-field comparison at Gate 2, and either engine can fill a run folder (E8). Every process builds the same architecture from the same config, so the key order is identical in the server and in every worker |
| F13 | Flower clients run in separate worker processes, so each one: sets `torch.set_num_threads(cfg['train']['threads'])` itself (M5; new config key, value 3), rebuilds the dataset from `cfg` once per process (nothing large is shipped between processes), and computes the FedLA global histogram as the sum of every client's own histogram from the manifest rather than exchanging it over the network. Ray is asked for `min(threads, os.cpu_count())` CPUs per client. The simulation is started with `flwr.simulation.run_simulation`, which 1.42 marks deprecated in favour of the `flwr run` CLI | The histogram sum is identical to `run_federated`'s and no result depends on the exchange being real: a simulation shortcut of the same kind as pooled scaling statistics (D5), to be stated in the report. Asking for more CPUs than the machine has would leave Ray waiting forever for a client it can never place. `run_simulation` is used because Gate 2 must call both engines in one process and compare them; `flwr run` would need a subprocess |
| F14 | [F7 result, frozen] `fl.E = 1`, `fl.lr = 0.001`, `fl.R = 60`, K stays 5. Pilot (FedAvg, GPU, seeds 101–102, best validation MAE): E1 lr .001: H0 8.867 / 8.951 (mean 8.909), H1 mean 10.330 · E1 lr .000333: H0 mean 9.203, H1 10.966 · E3 lr .000333: H0 mean 9.252, H1 10.259 · E3 lr .001: H0 mean 9.324, H1 10.475. Runs: `runs/pilot/`, commit 471a23b | F7's rule: best FedAvg at H = 0 wins; it degrades at H = 1 (8.909 → 10.330), so the study's premise holds and the K = 10 fallback is not needed. Best rounds at H = 0 were 35–58 of 60, so R = 60 is kept. Test error not looked at |
| F15 | [F5 result, frozen] `fl.mu = 0.001`. FedProx at H = 1, seeds 101–103, best validation MAE: μ 0.001: 11.180 / 11.060 / 11.147 (mean 11.129) · μ 0.01: mean 12.133 · μ 0.1: mean 12.589 · μ 1: mean 13.274. Runs: `runs/musel/`, commit 2b0702e | F5's rule: lowest mean wins. The winner is the smallest value in the grid and error rises steadily with μ; the grid is not extended (fixed before the results). Observation for the report: even μ = 0.001 is worse than FedAvg on validation at H = 1 (pilot, same partitions since H = 1 is deterministic: FedAvg 10.368 / 10.292 vs FedProx 11.180 / 11.060 on seeds 101 / 102). Likely cause: the loss is MSE on RUL/125 (about 0.006), so even a small proximal term is of the same size. `tests/test_simulate.py` shows FedProx with μ = 0 equals FedAvg exactly |

Frozen signatures:

```python
client_seed(run_seed: int, rnd: int, client: int) -> int                               # see the change note below
client_update(model, global_weights, X, y, cfg, mu: float, seed: int, sample_weight=None) -> tuple[dict, int]  # new weights, n_examples
run_federated(cfg, ds, manifest, algo: str, seed: int) -> dict   # per round: global val MAE, per-client val MAE, bytes
```

`sample_weight` lets FedLA reuse `client_update` unchanged.

[changed when built] `client_update` gains a required `seed: int` argument (before `sample_weight`):
`client_update(model, global_weights, X, y, cfg, mu, seed, sample_weight=None)`. The seed fixes the client's
batch order and dropout masks. It comes from `client_seed(run_seed, round, client)`, which depends only on
those three numbers. Reason: the frozen signature had no source of randomness that is the same in the plain
simulator and in Flower (whose clients run in separate workers), so Gate 2 could not be met exactly.
`client_update` reads `cfg['fl']` E, lr (set by the pilot, F7), batch and clip. `X, y` are windows and labels in
cycles; it divides by `rul_cap` itself, as `train_centralized` does. `global_weights` is snapshotted with real
copies before training (`state_dict()` returns live references; without the copy FedProx's (p − g) stays 0).
`client_update` also (a) turns on `torch.use_deterministic_algorithms(True)` itself, because Flower clients
run in fresh worker processes where nothing else sets it (M5); the thread count (M5: 3) must still be set in each
Flower worker by the Flower `train_fn`; (b) casts `sample_weight` to float32, so FedLA's loss is computed in the
same precision as FedAvg's (`fedla_weights` returns float64, which measurably changed the trained weights).

[as built] `run_federated(cfg, ds, manifest, algo, seed)` (`src/rulfl/fl/simulate.py`), signature unchanged:
- Start weights: `torch.manual_seed(seed)` then build the model, exactly as `train_centralized`, so a federated and
  a centralized run with the same seed start from identical weights.
- Rounds are numbered 1..R (like Flower's `server_round`); client k in round r uses `client_seed(seed, r, k)`.
- `manifest` is read as `{"train": {engine: client}, "val": {engine: client}}`; keys may be int or str (JSON).
  This is what the partitioner's manifest file (PT7) must contain.
- `algo`: `"fedavg"` (μ = 0), `"fedprox"` (μ = `cfg['fl']['mu']`, set by F5) or `"fedla"` (μ = 0 plus
  `sample_weight`, LA5). Anything else raises `ValueError`. FedLA: before round 1 each client's
  `label_histogram`, server sums them, each client gets `fedla_weights` once (LA4) from `cfg['fedla']`.
- LA6 bytes, made exact: K histograms of 6 counts up + the global histogram down to each of the K clients
  (down counted per client, as F8 does for weights), 4 bytes per count: 2 × K × 6 × 4 (240 bytes at K = 5),
  added once, to round 1's bytes.
- Returns `{"val_mae": [R+1], "client_val_mae": [(R+1) × K], "bytes": [R+1], "best_round": int, "best_state": dict}`.
  Index 0 = round 0, the untrained start weights, scored before any training (F10, as the Flower app does);
  its bytes are 0. List index = round number. `client_val_mae` is nan for a client with no routed validation
  engine. `best_round` (0..R) = first round with the lowest global validation MAE (ties keep the earlier
  round, F10); `best_state` = that round's global weights (F6).
- Server aggregation `fedavg(updates)` with `updates = [(weights, n_windows), ...]` (F3).
- Test of "K = 1 FedAvg equals centralized": one client, one round of E epochs equals a hand-written centralized
  loop with the same start weights and batch order (to 1e-6). Not compared with `train_centralized` itself:
  its batch order comes from `seed`, a client's from `client_seed`, and F2 restarts Adam every round.

## FedLA

Evidence (real FD004 training engines, H = 1, K = 5): every engine lives > 125 cycles, so each passes through every
RUL value 0–124 once. Within 0–124 every client's labels are flat (client 1: ~19% per 25-cycle bin; client 5: ~8%).
The skew is in the capped bin (RUL = 125, "healthy"): 8% of client 1's windows vs 62% of client 5's.
Client sizes also differ (6,673 vs 16,243 windows), so window-weighted aggregation gives long-lived clients ~2.4× the say.

| # | Decision | Reason |
|---|---|---|
| LA1 | Bins: [0,25), [25,50), [50,75), [75,100), [100,125) and the capped bin {125}: 6 bins. Fixed, not selected on validation (deviation from CLAUDE.md "bin width fixed on validation") | Labels are flat inside 0–124, so bin width barely changes the weights; selection runs would choose between near-identical options |
| LA2 | Clip weights to [0.1, 10], then renormalise to mean 1 on the client. Fixed, not selected (same deviation). Revisit only if the K = 10 fallback is used | Largest unclipped weight at K = 5, H = 1 is ≈ 5.4 (client 1, capped bin), so the clip never binds; it is a safety net |
| LA3 | Histograms count training windows (what the loss sees), not engines | Weights must match the per-window loss |
| LA4 | Weights computed once before round 1 and fixed for all rounds | Labels never change |
| LA5 | FedLA = FedAvg code path with `sample_weight`, μ = 0, same window-count aggregation | Only the weights differ, so FedLA and FedAvg stay paired |
| LA6 | Communication: K × 6 counts up and 6 down, once; added to bytes communicated | Exact accounting |
| LA7 | Limitation stated in the report: histograms reveal each client's label distribution | No privacy guarantees are claimed (CLAUDE.md rule 8) |
| LA8 | Labels outside [0, cap], or bin edges that do not rise strictly from 0 to cap, raise ValueError | A bad label would otherwise land silently in the wrong bin |
| LA9 | `fedla_weights` raises if `local_counts` differs from `label_histogram(y)`, or if global < local in any bin | Catches a caller passing the wrong client's histogram |
| LA10 | "Mean 1 before clipping" holds only if the client has windows in every bin with global mass; true on real FD004 (every engine outlives 125 cycles). Renormalisation guarantees mean 1 regardless | States the limit of the note under LA7 |

Note: before clipping the weights already have mean 1 on each client (Σ_b p_local(b)·p_global(b)/p_local(b) = 1),
so renormalisation only matters after clipping. Required tests: weights = 1 when local = global; mean = 1 after
renormalising; clipping applied.

Frozen signatures:

```python
label_histogram(y: np.ndarray, edges: list[int], cap: int) -> np.ndarray      # 6 window counts
fedla_weights(y: np.ndarray, local_counts, global_counts, edges, cap, clip) -> np.ndarray   # per-window weights, mean 1
```

## Evaluation and analysis

| # | Decision | Reason |
|---|---|---|
| E1 | Headline: test MAE over the 248 last-window test predictions (clamped, M4; capped labels, D8). Also RMSE and NASA score: d = predicted − true, cost exp(d/10) − 1 if d ≥ 0 (late) else exp(−d/13) − 1 (early), summed over engines | Proposal metrics; late predictions are penalised harder because they are dangerous |
| E2 | Worst-client MAE = max over clients of the global model's validation MAE on that client's routed validation engines (PT5), at the best round | Shows whether skew hurts some clients far more than others |
| E3 | Rounds-to-110%: first round whose global validation MAE ≤ 1.10 × mean best validation MAE of the 5 centralized runs. Runs that never reach it are recorded as "not reached" (censored), never dropped | Proposal: censored runs counted |
| E4 | Life-stage breakdown on test engines by true RUL: near failure < 50 (79 engines), mid 50–124 (102), healthy ≥ 125 / capped (67). Computed from saved predictions | Balanced groups; answers where the damage happens. Counts come from labels only, no model output |
| E5 | Baselines: centralized (all training engines, seeds 1–5); local-only (each client alone) at H = 0 and H = 1, seeds 1–5 | Centralized = the target; local-only shows what federation adds, at the two extremes only to save compute |
| E6 | Head-refit probe (diagnostic, labelled as such): take a run's best-round global model, freeze `model.gru`, re-initialise `model.head`, train only the head on all training windows with M3 settings (early stop on validation), then score on test | If test MAE returns near centralized, the damage is in the head; if not, it is in the features |
| E7 | Comparisons are paired by (H, seed). Per H: mean difference vs FedAvg with a 95% bootstrap interval over seeds. One-sided exact Wilcoxon signed-rank per H and pooled over H ≥ 0.5 (15 pairs). Implemented with numpy only | Paired design removes partition luck. With 5 pairs the smallest one-sided p is 1/32 ≈ 0.031, so per-H tests are weak; the intervals and the pooled test carry the evidence. Hypotheses written down before the main grid |
| E8 | Every run saves its config, manifest hash, per-round log, final metrics, best checkpoint and test predictions. Every report number and figure is generated from these files | CLAUDE.md rule 6 |
| E9 | `life_stage_mae`: an empty group returns NaN, never 0 | A missing group must not look like a perfect score |
| E10 | Every function in `eval/metrics.py` raises `ValueError` if `pred` or `true` is empty, not 1-D, or the two differ in length | A silent mismatch would corrupt every result |
| E11 | `life_stage_mae` returns `near_failure`, `mid`, `healthy` and the counts `n_near_failure`, `n_mid`, `n_healthy` | Reporting needs both the scores and the group sizes |
| E12 | Non-finite values (NaN or ±inf) in `pred` or `true` raise `ValueError` | Surfaces a diverged model instead of letting it enter evaluation |
| E13 | Limitation (for the report): the NASA score is a sum, so one very late prediction can dominate it | Read alongside MAE, never on its own |
| E14 | `rounds_to_threshold(val_curve, threshold_value) -> (round, reached)` reads `val_curve` as `run_federated` returns `val_mae`: index = round, index 0 = untrained round 0 (F10). First round with value ≤ threshold; None/NaN entries skipped; never reached gives `(None, False)` (censored, E3). Threshold must be finite and > 0; the caller computes it (1.10 × mean best centralized validation MAE) | One indexing convention with the simulator; a 1-based count would report every run one round late |
| E15 | Run folder layout (details E8). Folder = `runs/<group>/<name>/`; groups `tune`, `central`, `pilot`, `musel`, `main`, `local`, `probe`. The name holds every setting that varies inside its group: `tune/h64_l2_lr0.003_s1`, `central/FD004_s1`, `pilot/fedavg_H1.00_s101_E3_lr0.001`, `musel/fedprox_H1.00_s101_mu0.01`, `main/fedla_H0.50_s3`, `local/H1.00_s2_c4`. Fixed files: `config.yaml` (resolved config), `rounds.jsonl` (federated: one line per round 0..R with `val_mae`, `client_val_mae`, `bytes`) or `epochs.json` (centralized/local: `train_centralized` history), `final.json`, `best.pt`, `test_predictions.csv` (`engine`, `pred`, `true`, `true_uncapped`; real engine ids), `DONE` (written last). `final.json`: `test` (mae, rmse, nasa), `life_stage_mae`, `best_val_mae`, `best_round`, `worst_client_mae`, `round_to_threshold`, `threshold_reached`, `total_bytes`, `manifest_sha256` (the manifest's embedded hash, PT11), `git_commit`, `data_hash`, `seconds`. A folder with `DONE` and the same `config.yaml` is skipped; one with a different `config.yaml` is refused, never skipped or overwritten. `results/index.csv` (one row per run, committed) is rebuilt from every `final.json` by one command, never appended to while runs are going | The task list's `<kind>_<algo>_H<H>_s<seed>` gives the 4 μ-selection runs (and the pilot's E/lr variants) one shared folder, so the DONE skip would silently drop all but the first. The config check catches any naming slip. Rebuilding the index keeps the git tree clean during a grid (the runner refuses a dirty tree) and lets parallel runs never write the same file |
| E16 | Runner [as built]: `python -m rulfl.run <group> --seed S [--algo A] [--H h] [--client k] [--set section.key=value]`; `python -m rulfl.batch plan|run|index`. The folder name is computed from the resolved config, never typed. Each group only accepts its own seeds: tune = `tuning.seeds_first_pass` + `seeds_confirm`; central, main, local = `seeds.main`; pilot = `seeds.pilot`; musel = `seeds.selection`. Pilot is FedAvg only, musel FedProx only. `--set` changes existing keys only, except `model.*`, `fl.E`, `fl.R`, `fl.lr`, `fl.mu` (not in base.yaml until tuning / pilot / μ selection freeze them). Federated runs need the 5 finished `central` runs (E3 threshold) and the hash-checked manifest before any folder is made. `final.json` adds `group`, `name`, `seed`, `algo`, `H`, `client`, `git_dirty`. Registered runs use the simulator (F11) | Seed rules make rule 5 and M6 impossible to break by accident. A typo in `--set` fails loudly instead of running the wrong experiment. Failing before the folder exists leaves nothing half-made |
| E17 | Head-refit probe [as built]: `python -m rulfl.run probe --algo A --H h --seed s` refits `runs/main/<same name>`. Its GRU is frozen; its output after the last cycle is computed once per window (eval mode, no dropout) and a fresh head (same architecture, seeded) is trained on those vectors for all training windows with M3 settings and early stopping on validation (`rulfl/probe.py`). `final.json` adds `source_run`, `source_test_mae`, `diagnostic: true` | Equivalent to running the frozen GRU every batch, and much faster. Eval-mode features mean the probe sees the same GRU output the test predictions use |
| E18 | Analysis [as built]: `python -m rulfl.analysis` reads only finished run folders and writes `results/` (main table, per-H paired differences with bootstrap CIs and Wilcoxon, the two pre-registered primary tests with Holm, the premise, baselines, probe, skew by H, figures, `summary.md`). Implements `docs/PREREGISTRATION.md` | Rule 6: every report number is regenerated from logs by one command |

Frozen signatures:

```python
nasa_score(pred: np.ndarray, true: np.ndarray) -> float
metrics(pred, true) -> dict                     # mae, rmse, nasa
rounds_to_threshold(val_curve, threshold_value: float) -> tuple[int | None, bool]   # E14
life_stage_mae(pred, true, edges=(50, 125)) -> dict
```
