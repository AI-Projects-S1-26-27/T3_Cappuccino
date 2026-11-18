# M7 --- Experiment Operations

## Purpose

M7 is responsible for making the project's experiments **reproducible,
traceable, and schedulable**. M7 coordinates experiment execution for
the team; it does not own or execute a separate model-training
experiment of its own.

This document describes the M7 components currently present in this
repository snapshot:

-   `src/rulfl/run.py` --- reproducible entry point for an individual
    experiment.
-   `ops/run_grid.py` --- deterministic scheduler for the main federated
    experiment grid.
-   `ops/registry.py` --- summary of experiment statuses recorded in the
    registry CSV.
-   `tests/test_run.py` --- tests for the M7 runner and grid contracts.
-   `ops/M7_HANDOFF.md` --- implementation handoff notes and outstanding
    dependencies.

## M7 responsibilities

1.  Provide one consistent command-line entry point for experiment runs.
2.  Record each run's configuration, Git revision, data hash, manifest
    hash (when applicable), metrics, and execution time.
3.  Prevent accidental overwriting of completed runs and avoid treating
    incomplete output as a successful run.
4.  Generate a deterministic main-grid schedule and allow work to be
    divided by seed.
5.  Maintain a simple view of run statuses for coordination with M6.
6.  Support reproducibility checks and hand off run outputs and
    provenance to M6.

## Repository components

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `src/rulfl/run.py`                  Validates run arguments and frozen
                                      settings, executes
                                      centralized/local/federated runs,
                                      saves outputs, and updates the
                                      registry.

  `ops/run_grid.py`                   Builds and previews the main grid;
                                      optionally executes it with a
                                      bounded number of worker processes.

  `ops/registry.py`                   Reads the registry CSV and prints
                                      counts for planned, running, done,
                                      invalid, and failed runs.

  `tests/test_run.py`                 Checks run identifiers,
                                      frozen-configuration validation,
                                      completed-run skipping, atomic
                                      index-row replacement, and the
                                      75-run paired grid.

  `configs/base.yaml`                 Shared configuration input used by
                                      the runner and scheduler.

  `ops/M7_HANDOFF.md`                 Current M7 implementation status,
                                      prerequisites, and handoff notes.
  -----------------------------------------------------------------------

## Environment setup

The project requires **Python 3.12**. Run the following commands from
the repository root.

### Windows Command Prompt

``` cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

### macOS / Linux

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

Install from `requirements.lock` to use the repository's pinned
dependency set. The editable install makes the `rulfl` package
importable from the source tree.

## Running M7 checks

Run the M7-specific tests:

``` bash
python -m pytest -q tests/test_run.py
```

Run the complete test suite:

``` bash
python -m pytest -q
```

The M7-specific test file currently contains five tests. A successful
run of those tests checks the runner contracts; it does **not**, by
itself, prove that the complete federated workflow or final research
grid has run successfully.

## Running one experiment

The runner supports three experiment kinds:

-   `centralized`
-   `local`
-   `federated`

Example centralized run:

``` bash
python -m rulfl.run --kind centralized --seed 1
```

Example federated simulation run:

``` bash
python -m rulfl.run --kind federated --algo fedavg --H 0.5 --seed 1 --engine sim
```

Supported federated algorithms are `fedavg`, `fedprox`, and `fedla`. The
engine can be `sim` or `flower`; `sim` is the default.

For local-only runs, provide a client index:

``` bash
python -m rulfl.run --kind local --H 0.5 --seed 1 --client 0
```

The runner reads `configs/base.yaml` by default. Use `--config PATH` to
select another configuration, and repeat `--set SECTION.KEY=VALUE` to
override existing configuration keys.

### Important run prerequisites

Before a registered run can execute:

-   The model settings (`model.hidden`, `model.layers`, `model.lr`, and
    `model.dropout`) must be frozen in the selected configuration.
-   Federated runs additionally require frozen `fl.E`, `fl.R`, `fl.lr`,
    and `fl.mu` values.
-   The required NASA C-MAPSS raw data files must be present in
    `data/raw/`.
-   Local and federated runs require the corresponding partition
    manifest in `manifests/`.
-   The run must be launched from a Git working tree with no uncommitted
    changes. `--allow-dirty` exists for development only and should not
    be used for registered research runs.

The repository snapshot may not yet contain the final manifests or
frozen model/FL settings. Do not invent these values or bypass the
prerequisites to start the final experiment grid.

## Previewing and scheduling the main grid

The scheduler reads `configs/base.yaml`. The configured main grid is:

-   Heterogeneity values: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`
-   Main seeds: `1`, `2`, `3`, `4`, `5`
-   Algorithms: `fedavg`, `fedprox`, `fedla`

This produces **75 runs** (5 heterogeneity values × 5 seeds × 3
algorithms). The three algorithms for a given `(H, seed)` pair use the
same manifest, supporting paired comparisons.

Preview the complete grid without training:

``` bash
python ops/run_grid.py --dry-run
```

Preview only seed 1:

``` bash
python ops/run_grid.py --seed 1 --dry-run
```

After the manifests, frozen settings, pilot work, and required gates are
complete, execute an assigned seed:

``` bash
python ops/run_grid.py --seed 1 --jobs 1 --execute
```

You can assign more than one allowed main seed by repeating `--seed`.
The scheduler accepts `--jobs` from 1 to 4; each run uses three CPU
threads, so start with one job and increase concurrency only after
checking available resources.

The scheduler checks that federated settings are frozen, the Git tree is
clean, and the required manifests exist before execution. It does not
replace the upstream gates or the team's approval process.

## Registry and run outputs

The default registry path is:

``` text
ops/registry.csv
```

Display a status summary:

``` bash
python ops/registry.py
```

Use a custom registry path:

``` bash
python ops/registry.py --index path/to/registry.csv
```

The runner writes each run under `runs/<run_id>/`. Depending on the run
type, outputs include:

  -----------------------------------------------------------------------
  Output                              Description
  ----------------------------------- -----------------------------------
  `config.yaml`                       Effective configuration used for
                                      the run.

  `final.json`                        Final metrics, run metadata, Git
                                      SHA, data hash, manifest hash (when
                                      applicable), and elapsed time.

  `best.pt`                           Best model checkpoint. Keep
                                      checkpoints in the designated
                                      shared storage; do not commit them
                                      to Git.

  `predictions.csv`                   Per-engine predicted and reference
                                      RUL values.

  `rounds.jsonl`                      Per-round validation/client metrics
                                      and communication bytes for
                                      federated runs.

  `epochs.json`                       Training history for centralized
                                      and local runs.

  `DONE`                              Marker written when the run
                                      finishes successfully.
  -----------------------------------------------------------------------

The runner first writes to a temporary run directory and only promotes
it to the final run directory after successful completion. A failed run
is moved to a `.invalid` directory and recorded with status `invalid`.
An existing run with a `DONE` marker is skipped; an unfinished target is
not silently overwritten.

The registry records run identifiers, run type, algorithm, H, seed,
client, status, test MAE, best validation MAE, Git SHA, manifest SHA,
and elapsed seconds.

## Reproducibility and safe operation

-   Commit code and configuration changes before launching registered
    experiments.
-   Do not run two copies of the same registered run.
-   Do not manually edit generated manifests or completed run folders.
-   Do not commit raw NASA data, model checkpoints, credentials, or
    other large/private artifacts.
-   Keep the same committed configuration, manifest, data files, and
    software lock file when reproducing a run.
-   Share completed run folders and the registry with M6 for evaluation
    and statistical analysis.
-   Do not launch the final grid until the required upstream gates and
    configuration freezes have passed.

## Current verification and limitations

In the Windows Python 3.12 environment used during M7 setup, the
M7-specific test command completed with **5 passed**. The grid test
confirmed 75 unique main-grid runs.

This is evidence for the tested M7 runner contracts and grid generation
only. It is not evidence that:

-   the complete project test suite passes on every machine;
-   Flower-based execution has passed its integration gate;
-   the final real-data manifests and frozen configurations are present;
-   the full 75-run research grid has completed; or
-   clean-machine end-to-end reproduction has passed.

These remain dependent on upstream contributions and the final
integration/reproduction process.

## M7 handoff

M7 hands the run folders, registry, and provenance information to **M6**
for validation and statistical analysis. M6 is the designated reviewer
for M7 changes. The runner should also be reviewed with M4 where Flower
integration is affected.

See [`ops/M7_HANDOFF.md`](ops/M7_HANDOFF.md) for the snapshot-specific
handoff status and outstanding dependencies.
