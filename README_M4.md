# M4: Flower integration — handoff

## Source of truth

The task-card PDF describes an earlier FedAvg/FedProx experiment and Python 3.11. The repository's `docs/DECISIONS.md` and `OPEN_TASKS.txt` supersede it: use **Python 3.12**, `requirements.lock`, and three arms (**FedAvg, FedProx, FedLA**). Do not change frozen decisions to make tests pass.

## What's implemented

- `src/rulfl/flower_app/app.py`: Flower 1.42 `ClientApp` / `ServerApp` message API, plain `FedAvg` strategy, full client participation, server-side validation at rounds 0..R, best-round tracking.
- `src/rulfl/flower_app/rul.py`: real RUL model adapter, stateless `client_update` calls with `client_seed`, per-client train/val assignment from a manifest, FedLA weights, exact communication-byte accounting, isolated model-array transport, and validation of transport dtypes/finite weights.
- `tests/test_flower_app.py`: five-client toy integration tests.
- `tests/test_flower_rul.py`: adapter and algorithm correctness tests.
- `tests/test_gate2.py`: **Gate 2**, compares Flower to `run_federated` for FedAvg/FedProx/FedLA over two rounds, including validation, best checkpoint weights, client metrics, and bytes.

## Validate on a Python 3.12 machine

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
python -m pytest -q tests/test_flower_rul.py
python -m pytest -q tests/test_flower_app.py
python -m pytest -q tests/test_gate2.py
python -m pytest -q
```

Ray-backed Flower tests require Flower/Ray installed, plus sufficient memory/CPUs. `tests/test_gate2.py` uses the committed test data slice and a small test-only model; it **does not** need full FD004 training data or the final tuned model.

## Run in the future experiment runner

After T3 (experiment runner), T1 (manifests), and T4/T6 (frozen model/FL settings) exist, the runner should call:

```python
from rulfl.flower_app.rul import run_flower_federated
result = run_flower_federated(cfg, ds, manifest, algo="fedavg", seed=1)
# Also supported: algo="fedprox", algo="fedla"
```

The return fields exactly match the reference simulator: `val_mae`, `client_val_mae`, `bytes`, `best_round`, `best_state`. No experiment should be registered before T3 freezes its resolved config and records the manifest hash.

## Notes for reviewers (M7)

- Confirm Gate 2 green in CI **and** on a second machine (task card).
- Keep Flower server strategy as **plain FedAvg**; FedProx loss is already applied on the client.
- Keep client seeding, training implementation and model parameter order identical to the reference.
- FedLA histogram communication bytes are *accounted for* but the clients reconstruct the global histogram from the manifest; disclose this simulation shortcut as required by decision F13.
- Ray's worker scheduling/aggregation order may cause tiny float32 differences; use F11 tolerances, not byte-identical checkpoint hashes.
- At this snapshot `configs/base.yaml` has no frozen `model` or `fl.E/R/lr/mu`, and there is no registered experiment runner. Consequently, **the six M4 sensitivity runs are blocked by T1/T3/T4/T6/Gate 3**. They are not claimed complete.

Once dependencies land, M4 owns the **six H=1 FedProx sensitivity runs** (selected mu/10 and mu×10; seeds 1–3) and reviews M7's runner/CI PR.
