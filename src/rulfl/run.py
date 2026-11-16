"""Reproducible experiment entry point (OPEN_TASKS T3, DECISIONS E8).

Example:
  python -m rulfl.run --kind federated --algo fedavg --H 0.5 --seed 1
  python -m rulfl.run --kind centralized --seed 1

Requires frozen model settings and real NASA data. No results are fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from dataclasses import replace

import numpy as np
import torch
import yaml

from rulfl.data.pipeline import build_dataset
from rulfl.eval.metrics import life_stage_mae, metrics, rounds_to_threshold
from rulfl.model.gru import RULModel, predict_rul
from rulfl.model.train import train_centralized


def git_commit(*, allow_dirty: bool = False) -> str:
    """Refuse unversioned/dirty registered experiments unless explicitly overridden."""
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        if allow_dirty:
            return 'UNVERSIONED'
        raise RuntimeError('Run inside a git clone, or use --allow-dirty for development') from None
    if dirty and not allow_dirty:
        raise RuntimeError('Git working tree is dirty; commit the changes or use --allow-dirty for development')
    return sha


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path, overrides: list[str]) -> dict:
    with path.open(encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    for item in overrides:
        if '=' not in item or '.' not in item.split('=', 1)[0]:
            raise ValueError('Override must be section.key=value')
        key, value = item.split('=', 1)
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f'Unknown config section in {key}')
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f'Unknown config key {key}')
        target[parts[-1]] = yaml.safe_load(value)
    return cfg


def require_frozen(cfg: dict, kind: str) -> None:
    for name in ('hidden', 'layers', 'lr', 'dropout'):
        if name not in cfg.get('model', {}):
            raise ValueError(f'Model tuning is not frozen: missing model.{name} in config')
    if kind == 'federated':
        for name in ('E', 'R', 'lr', 'mu'):
            if name not in cfg['fl']:
                raise ValueError(f'FL pilot is not frozen: missing fl.{name} in config')


def make_run_id(kind: str, algo: str | None, H: float | None, seed: int, client: int | None) -> str:
    if kind == 'centralized':
        return f'centralized_s{seed}'
    if H is None or not 0 <= H <= 1:
        raise ValueError('Local/federated runs require --H in [0, 1]')
    part = f'{kind}_{algo or "baseline"}_H{H:.2f}_s{seed}'
    return part if client is None else f'{part}_c{client}'


def read_manifest(cfg: dict, H: float, seed: int) -> tuple[dict, Path]:
    path = Path(cfg['partition']['manifests_dir']) / f'H{H:.2f}_s{seed}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing manifest {path}; T1 must generate and commit it first')
    manifest = json.loads(path.read_text(encoding='utf8'))
    if not all(k in manifest for k in ('train', 'val')):
        raise ValueError(f'Manifest {path} must have train and val assignments')
    # Accept either a raw-file SHA or a hash over the manifest excluding its sha256 field.
    if manifest.get('sha256'):
        payload = {k: v for k, v in manifest.items() if k != 'sha256'}
        canonical = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if manifest['sha256'] not in (sha256(path), canonical):
            raise ValueError(f'Manifest integrity mismatch: {path}')
    return manifest, path


def data_hash(cfg: dict) -> str:
    d = cfg['data']
    paths = [Path(d['raw_dir']) / f'{p}_{d["fd"]}.txt' for p in ('train', 'test', 'RUL')]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.name.encode())
        digest.update(sha256(p).encode())
    return digest.hexdigest()


def update_index(path: Path, entry: dict) -> None:
    fields = ['run_id', 'kind', 'algo', 'H', 'seed', 'client', 'status', 'test_mae', 'best_val_mae', 'git_sha', 'manifest_sha', 'seconds']
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline='', encoding='utf8') as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['run_id'] != entry['run_id']]
    rows.append({key: entry.get(key, '') for key in fields})
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['run_id']))
    os.replace(path.with_suffix('.tmp'), path)


def execute(cfg: dict, *, kind: str, algo: str | None, H: float | None,
            seed: int, client: int | None, engine: str, out_dir: Path,
            baseline_mae: float | None) -> dict:
    """Run one experiment, writing the artifacts into a temporary directory."""
    from rulfl.fl.simulate import client_rows, run_federated

    ds = build_dataset(cfg)
    cap = cfg['data']['rul_cap']
    manifest, manifest_path = (read_manifest(cfg, H, seed) if kind != 'centralized' else (None, None))
    history = None
    log = None
    if kind == 'centralized':
        model, history = train_centralized(cfg, ds, seed)
        best_val = float(history['val_mae'][history['best_epoch']])
        best_round = None
    elif kind == 'local':
        if client is None or not 0 <= client < cfg['fl']['K']:
            raise ValueError('Local baseline requires --client from 0 to K-1')
        tr = client_rows(ds.engine_train, manifest['train'], cfg['fl']['K'])[client]
        va = client_rows(ds.engine_val, manifest['val'], cfg['fl']['K'])[client]
        if not len(tr) or not len(va):
            raise ValueError(f'Client {client} has no training/validation windows')
        subset = replace(ds, X_train=ds.X_train[tr], y_train=ds.y_train[tr],
                         engine_train=ds.engine_train[tr], X_val=ds.X_val[va],
                         y_val=ds.y_val[va], engine_val=ds.engine_val[va])
        model, history = train_centralized(cfg, subset, seed)
        best_val = float(history['val_mae'][history['best_epoch']])
        best_round = None
    else:
        if algo not in ('fedavg', 'fedprox', 'fedla'):
            raise ValueError('Federated --algo must be fedavg, fedprox or fedla')
        if algo in ('fedavg', 'fedla'):
            cfg['fl']['mu'] = 0.0
        if engine == 'sim':
            log = run_federated(cfg, ds, manifest, algo, seed)
        else:
            from rulfl.flower_app.rul import run_flower_federated
            log = run_flower_federated(cfg, ds, manifest, algo, seed)
        model = RULModel(ds.X_train.shape[2], cfg['model']['hidden'], cfg['model']['layers'], cfg['model']['dropout'])
        model.load_state_dict(log['best_state'])
        best_round = int(log['best_round'])
        best_val = float(log['val_mae'][best_round])

    predictions = predict_rul(model, ds.X_test, cap)
    headline = metrics(predictions, ds.y_test)
    stage = life_stage_mae(predictions, ds.y_test, cfg['eval']['life_stage_edges'])
    if log is not None:
        client_vals = np.asarray(log['client_val_mae'][best_round], dtype=float)
        finite = client_vals[np.isfinite(client_vals)]
        worst_client = float(finite.max()) if len(finite) else None
        threshold = None if baseline_mae is None else cfg['eval']['threshold'] * baseline_mae
        reached_round, reached = (rounds_to_threshold(log['val_mae'], threshold)
                                  if threshold is not None else (None, None))
        with (out_dir / 'rounds.jsonl').open('w', encoding='utf8') as f:
            for rnd, (v, clients, nbytes) in enumerate(zip(log['val_mae'], log['client_val_mae'], log['bytes'])):
                f.write(json.dumps({'round': rnd, 'val_mae': v, 'client_val_mae': clients, 'bytes': int(nbytes)}) + '\n')
    else:
        worst_client, reached_round, reached = None, None, None
        (out_dir / 'epochs.json').write_text(json.dumps(history, indent=2), encoding='utf8')
    torch.save(model.state_dict(), out_dir / 'best.pt')
    with (out_dir / 'predictions.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['engine', 'pred', 'true', 'true_uncapped'])
        for i, (pred, true, raw) in enumerate(zip(predictions, ds.y_test, ds.y_test_uncapped), 1):
            writer.writerow([i, float(pred), float(true), float(raw)])
    return {'test': headline, 'life_stage_mae': stage, 'best_val_mae': best_val,
            'best_round': best_round, 'worst_client_mae': worst_client,
            'round_to_threshold': reached_round, 'threshold_reached': reached,
            'total_bytes': int(sum(log['bytes'])) if log else None,
            'manifest_sha': sha256(manifest_path) if manifest_path else None,
            'manifest_path': str(manifest_path) if manifest_path else None}


def run(args: argparse.Namespace) -> Path:
    cfg = load_config(Path(args.config), args.set or [])
    require_frozen(cfg, args.kind)
    if args.kind != 'federated' and args.algo:
        raise ValueError('--algo is only valid for federated runs')
    if args.engine == 'flower' and args.kind != 'federated':
        raise ValueError('Flower engine is only for federated runs')
    if args.kind == 'local' and args.client is None:
        raise ValueError('Local-only runs require --client')
    if args.kind != 'local' and args.client is not None:
        raise ValueError('--client is only valid for local runs')
    run_id = make_run_id(args.kind, args.algo, args.H, args.seed, args.client)
    target = Path(args.runs_dir) / run_id
    if (target / 'DONE').exists():
        print(f'SKIPPED: {target} already DONE')
        return target
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite unfinished run {target}')
    git_sha = git_commit(allow_dirty=args.allow_dirty)
    torch.set_num_threads(cfg['train']['threads'])
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tmp = target.with_name(target.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    start = time.monotonic()
    entry = {'run_id': run_id, 'kind': args.kind, 'algo': args.algo or '',
             'H': '' if args.H is None else args.H, 'seed': args.seed,
             'client': '' if args.client is None else args.client, 'git_sha': git_sha}
    try:
        if args.kind == 'federated' and args.algo in ('fedavg', 'fedla'):
            cfg['fl']['mu'] = 0.0
        (tmp / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True), encoding='utf8')
        result = execute(cfg, kind=args.kind, algo=args.algo, H=args.H, seed=args.seed,
                         client=args.client, engine=args.engine, out_dir=tmp,
                         baseline_mae=args.centralized_val_mae)
        result['life_stage_mae'] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                    for k, v in result['life_stage_mae'].items()}
        result.update({'run_id': run_id, 'kind': args.kind, 'algo': args.algo,
                       'H': args.H, 'seed': args.seed, 'client': args.client,
                       'git_sha': git_sha, 'data_hash': data_hash(cfg),
                       'seconds': time.monotonic() - start})
        (tmp / 'final.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
        (tmp / 'DONE').write_text('validated\n', encoding='utf8')
        tmp.rename(target)
        entry.update(status='done', test_mae=result['test']['mae'],
                     best_val_mae=result['best_val_mae'], manifest_sha=result['manifest_sha'],
                     seconds=result['seconds'])
        update_index(Path(args.index), entry)
        print(f'DONE: {target}')
        return target
    except Exception:
        invalid = target.with_name(target.name + '.invalid')
        if invalid.exists():
            shutil.rmtree(invalid)
        tmp.rename(invalid)
        entry['status'] = 'invalid'
        update_index(Path(args.index), entry)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['centralized', 'local', 'federated'], required=True)
    parser.add_argument('--algo', choices=['fedavg', 'fedprox', 'fedla'])
    parser.add_argument('--H', type=float)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--client', type=int)
    parser.add_argument('--engine', choices=['sim', 'flower'], default='sim')
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--index', default='ops/registry.csv')
    parser.add_argument('--centralized-val-mae', type=float,
                        help='Mean of the five centralized best-validation MAEs (E3); leave blank until baselines finish')
    parser.add_argument('--allow-dirty', action='store_true', help='Development only; not for registered runs')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
