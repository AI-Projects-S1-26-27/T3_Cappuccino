"""Show experiment registry counts (M7). Usage: python ops/registry.py"""
import argparse
import csv
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--index', default='ops/registry.csv')
args = p.parse_args()
path = Path(args.index)
if not path.exists():
    print('No registered runs yet.')
else:
    with path.open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row['status'] for row in rows)
    print('Total:', len(rows))
    for status in ('planned', 'running', 'done', 'invalid', 'failed'):
        print(f'{status:>8}: {counts[status]}')
