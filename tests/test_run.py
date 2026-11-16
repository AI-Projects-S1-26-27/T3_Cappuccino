"""Fast M7 checks: config validation, resume, and the complete paired experiment grid."""
import argparse
import csv
from pathlib import Path

import pytest

from rulfl import run as runner


def test_run_id_and_kind_checks():
    assert runner.make_run_id('federated', 'fedla', 0.5, 3, None) == 'federated_fedla_H0.50_s3'
    assert runner.make_run_id('local', None, 1, 1, 4).endswith('_c4')
    with pytest.raises(ValueError):
        runner.make_run_id('federated', 'fedavg', 1.5, 1, None)


def test_unfrozen_config_fails_before_data_loading():
    with pytest.raises(ValueError, match='Model tuning is not frozen'):
        runner.require_frozen({'model': {}, 'fl': {}}, 'federated')


def test_done_is_skipped_and_no_compute(tmp_path, monkeypatch, capsys):
    cfg = {'model': {'hidden': 32, 'layers': 1, 'lr': 0.001, 'dropout': 0},
           'fl': {'E': 1, 'R': 2, 'lr': 0.001, 'mu': 0.01}, 'train': {'threads': 3}}
    monkeypatch.setattr(runner, 'load_config', lambda *_: cfg)
    run_dir = tmp_path / 'runs' / 'federated_fedavg_H0.50_s1'
    run_dir.mkdir(parents=True)
    (run_dir / 'DONE').write_text('validated\n')
    args = argparse.Namespace(config='unused', set=[], kind='federated', algo='fedavg',
                              H=0.5, seed=1, client=None, engine='sim', runs_dir=str(tmp_path / 'runs'))
    assert runner.run(args) == run_dir
    assert 'SKIPPED' in capsys.readouterr().out


def test_atomic_index_row_replacement(tmp_path):
    path = tmp_path / 'registry.csv'
    runner.update_index(path, {'run_id': 'a', 'status': 'running'})
    runner.update_index(path, {'run_id': 'a', 'status': 'done', 'test_mae': 4.2})
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]['status'] == 'done'
    assert rows[0]['test_mae'] == '4.2'


def test_paired_grid_has_75_unique_runs():
    import importlib.util
    spec = importlib.util.spec_from_file_location('m7_grid', Path(__file__).parents[1] / 'ops' / 'run_grid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {'seeds': {'main': [1, 2, 3, 4, 5]},
           'partition': {'H_grid': [0, 0.25, 0.5, 0.75, 1]}}
    runs = list(module.grid(cfg))
    assert len(runs) == len(set(runs)) == 75
    for seed in range(1, 6):
        for H in cfg['partition']['H_grid']:
            assert {algo for h, s, algo in runs if h == H and s == seed} == {'fedavg', 'fedprox', 'fedla'}
