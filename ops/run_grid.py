"""M7 scheduler: deterministic T8 grid and resumable subprocess execution.

Preview: python ops/run_grid.py --dry-run
Run one assigned seed after gates/config freeze: python ops/run_grid.py --seed 1 --jobs 1 --execute
Outputs a command for each of the three paired algorithms at every skew level.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
import sys
from pathlib import Path

import yaml


def grid(cfg: dict, seeds: list[int] | None = None):
    seeds = list(cfg['seeds']['main'] if seeds is None else seeds)
    for seed in seeds:
        for H in cfg['partition']['H_grid']:
            for algo in ('fedavg', 'fedprox', 'fedla'):
                yield (float(H), int(seed), algo)


def command(H: float, seed: int, algo: str, engine: str) -> list[str]:
    return [sys.executable, '-m', 'rulfl.run', '--kind', 'federated', '--algo', algo,
            '--H', str(H), '--seed', str(seed), '--engine', engine]


def run_one(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--seed', type=int, action='append', help='Main-grid seed(s) assigned to this computer')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--engine', choices=('sim', 'flower'), default='sim')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true', help='Preview the registered grid; no models are trained')
    action.add_argument('--execute', action='store_true', help='Run the grid after gate 2 and pilot freeze')
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    allowed = set(cfg['seeds']['main'])
    if args.seed and not set(args.seed).issubset(allowed):
        parser.error('Only seeds.main may be used for the final grid')
    if args.jobs < 1 or args.jobs > 4:
        parser.error('jobs must be between 1 and 4 (each run uses three CPU threads)')
    tasks = list(grid(cfg, args.seed))
    if args.execute:
        from rulfl.run import require_frozen, git_commit
        require_frozen(cfg, 'federated')
        git_commit()  # --allow-dirty is deliberately unsupported for registered grid runs
        missing = [f'H{H:.2f}_s{seed}.json' for H, seed, _ in tasks
                   if not (Path(cfg['partition']['manifests_dir']) / f'H{H:.2f}_s{seed}.json').is_file()]
        if missing:
            parser.error(f'Missing {len(set(missing))} manifests, generate and commit them before training')
    print(f'{len(tasks)} main-grid experiments; engine={args.engine}; jobs={args.jobs}')
    cmds = [command(*task, args.engine) for task in tasks]
    if args.dry_run:
        for cmd in cmds:
            print(' '.join(cmd))
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, cmds))
    failures = sum(code != 0 for code in results)
    print(f'Finished: {len(results) - failures}; failed: {failures}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
