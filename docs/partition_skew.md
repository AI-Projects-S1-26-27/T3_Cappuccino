# Partition skew on real FD004 (PT6 check)

Generated 2026-11-16 by `python -m rulfl.partition.make_manifests` on the full NASA FD004 files
(checksums verified), K = 5, 199 training engines, 50 validation engines, 40 manifests
(H grid x seeds 1-5, 101-103).

| H | Mean skew (cycles) | Min training engines per client | Min validation engines per client |
|---|---|---|---|
| 0.00 | 1.339 | 26 | 6 |
| 0.25 | 2.683 | 27 | 6 |
| 0.50 | 4.642 | 27 | 5 |
| 0.75 | 7.372 | 32 | 6 |
| 1.00 | 10.401 | 39 | 10 |

- **Mean skew** = mean over clients of the Wasserstein-1 distance between that client's training RUL labels and all
  training labels (PT6), averaged over the main seeds 1-5. The minimum columns are over all 8 seeds.
- Skew rises strictly over H, so PT6 holds. H = 1 gives about 7.8x the skew of H = 0. H = 0 is the floor any
  random split of 199 engines produces.
- PT4 (at least 10 training engines per client) holds everywhere, with the lowest count 26.
- At mid H some clients get only 5-6 validation engines, so worst-client MAE (E2) is noisy there (PT9 limitation).

The numbers live in the committed manifests (`skew`, `n_train_per_client`, `n_val_per_client` in each
`manifests/H*_s*.json`); this table is a readable copy, not the source. To regenerate it from the files:

```python
import json, glob, numpy as np
ms = [json.load(open(p)) for p in glob.glob("manifests/H*_s*.json")]
for H in sorted({m["H"] for m in ms}):
    g = [m for m in ms if m["H"] == H]
    print(H, np.mean([m["skew"] for m in g if m["seed"] <= 5]),
          min(min(m["n_train_per_client"]) for m in g), min(min(m["n_val_per_client"]) for m in g))
```
