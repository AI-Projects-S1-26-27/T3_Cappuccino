# M7 experiment operations handoff (2026-11-16)

## Scope and status

The current repo already contains `src/rulfl/run.py`, `ops/registry.py`, and GitHub Actions CI. M7 additions: `ops/run_grid.py` plus `tests/test_run.py` and a fix to defer Flower imports for simulator-only runs. `src/rulfl/run.py` saves per-run config, predictions, best checkpoint, test scores, per-round logs, provenance and a DONE marker; it refuses dirty/unversioned registered runs (with a development-only override).

## Commands (from repo root)

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
pytest -q
python ops/run_grid.py --dry-run                   # preview all 75 paired runs
python ops/run_grid.py --seed 1 --dry-run          # preview 15 runs assigned to seed 1
python ops/run_grid.py --seed 1 --jobs 1 --execute # ONLY after manifests, tuning, pilot and Gate 2
python ops/registry.py                            # summarize status
```

For a single experiment after freeze:

```bash
python -m rulfl.run --kind federated --algo fedavg --H 0.5 --seed 1 --engine sim
```

Every run uses 3 CPU threads; the scheduler defaults to one process and permits at most four concurrent processes. The three algorithms for any pair `(H, seed)` share the same manifest. The runner uses `runs/<run_id>/DONE` to skip completed experiments. Failures land in a `.invalid` folder and a registry status. **Do not execute the final grid until hypotheses are pre-registered in DECISIONS.md and upstream gates pass.**

## Pending upstream dependencies

- T1: Generate and commit partition manifests. The current snapshot has no usable final manifests.
- T4/T6: Freeze `model` and `fl.E`, `fl.R`, `fl.lr`, `fl.mu` in `configs/base.yaml` from tuning and pilot results. This package does not invent these values.
- T5: Make Flower Gate 2 green on a fully provisioned machine; use `--engine flower` only after that.
- T2: Confirm all M6 evaluation functions and final E7 decisions have been merged.
- M7/GitHub owner: configure protected main branch with one approving reviewer and required CI check, and replace CODEOWNERS placeholders with real handles. This cannot be done by editing local ZIP files.
- M7 final reproduction: full fresh-machine validation and comparison of numeric experiment outcomes need NASA raw data, frozen configs, manifests, and completed runs.

## Review

Open a PR from a dedicated branch. Assign M6 as reviewer for M7 changes (verify GitHub handle with team). CI must be green before merge. Do not upload model checkpoints or raw NASA files into Git.
