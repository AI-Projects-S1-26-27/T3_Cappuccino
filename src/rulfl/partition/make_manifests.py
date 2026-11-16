"""Partition manifests: one JSON file per (H, seed), written once, committed, never hand-edited (PT7, PT11, PT12).

    python -m rulfl.partition.make_manifests                 # every H x every planned seed, into manifests/
    python -m rulfl.partition.make_manifests --K 10          # the proposal's fallback (F7), into a fresh --out

File: {"H", "seed", "K", "sha256", "skew", "n_train_per_client", "n_val_per_client", "train", "val"}
  train / val   engine id -> client (what run_federated and the Flower app read)
  sha256        hash of the assignment only (H, seed, K, train, val); runs record it as manifest_sha256 (E8)
  skew, n_*     diagnostics, recomputable, outside the hash (eval/plots.py reads skew)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rulfl.data.pipeline import Dataset
from rulfl.partition.label_skew import assign_clients, check_min_engines, skew, val_seed

_HASHED = ("H", "seed", "K", "train", "val")


def manifest_hash(manifest: dict) -> str:
    """SHA-256 of the canonical JSON of the assignment. Keys are made strings before sorting, so a manifest in memory
    (int keys) and the same manifest read back from JSON (str keys) hash identically."""
    canon = {"H": float(manifest["H"]), "seed": int(manifest["seed"]), "K": int(manifest["K"]),
             "train": {str(e): int(c) for e, c in manifest["train"].items()},
             "val": {str(e): int(c) for e, c in manifest["val"].items()}}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _counts(assignment: dict[int, int], K: int) -> list[int]:
    return np.bincount(np.fromiter(assignment.values(), dtype=np.int64), minlength=K)[:K].tolist()


def build_manifest(cfg: dict, ds: Dataset, H: float, seed: int, K: int | None = None) -> dict:
    """Assign training engines (PT4 checked) and route validation engines with their own seed (PT9)."""
    K = int(K if K is not None else cfg["fl"]["K"])
    train = assign_clients(ds.train_engine_lifetimes, K, H, seed)
    check_min_engines(train, K, cfg["partition"]["min_engines_per_client"])
    val = assign_clients(ds.val_engine_lifetimes, K, H, val_seed(seed))
    labels = [ds.y_train[np.isin(ds.engine_train, [e for e, c in train.items() if c == k])] for k in range(K)]
    m = {"H": float(H), "seed": int(seed), "K": K, "train": train, "val": val}
    return {"H": m["H"], "seed": m["seed"], "K": K, "sha256": manifest_hash(m),
            "skew": skew(labels, ds.y_train),
            "n_train_per_client": _counts(train, K), "n_val_per_client": _counts(val, K),
            "train": train, "val": val}


def manifest_path(out_dir, H: float, seed: int) -> Path:
    return Path(out_dir) / f"H{float(H):.2f}_s{int(seed)}.json"


def _dumps(manifest: dict) -> str:
    return json.dumps(manifest, indent=1) + "\n"


def write_manifest(manifest: dict, out_dir) -> Path:
    """Write once (PT7). Rewriting an identical manifest is a no-op; replacing a different one raises."""
    path = manifest_path(out_dir, manifest["H"], manifest["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")).get("sha256") == manifest["sha256"]:
            return path
        raise FileExistsError(f"{path} already holds a different partition. Manifests are never replaced; "
                              f"write to a new --out folder or delete the old set deliberately")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_dumps(manifest))
    return path


def load_manifest(path) -> dict:
    """Read a manifest, recheck its hash (refuses a hand-edited file), and give engine ids back as ints."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest_hash(m) != m.get("sha256"):
        raise ValueError(f"{path}: hash mismatch. The file was edited by hand or is corrupt; regenerate it")
    m["train"] = {int(e): int(c) for e, c in m["train"].items()}
    m["val"] = {int(e): int(c) for e, c in m["val"].items()}
    return m


def planned_seeds(cfg: dict) -> list[int]:
    """Every seed some run uses: main grid, mu selection and pilot (PT12)."""
    s = cfg["seeds"]
    return sorted(set(s["main"]) | set(s["selection"]) | set(s["pilot"]))


def make_all(cfg: dict, ds: Dataset, out_dir, K: int | None = None) -> list[dict]:
    """Build every (H, seed) manifest in memory, check PT6 on the main seeds, and only then write them.
    If H does not raise skew strictly, nothing is written: the study's knob would be broken."""
    grid = [float(h) for h in cfg["partition"]["H_grid"]]
    manifests = [build_manifest(cfg, ds, H, s, K) for H in grid for s in planned_seeds(cfg)]
    main = set(cfg["seeds"]["main"])
    means = [float(np.mean([m["skew"] for m in manifests if m["H"] == H and m["seed"] in main])) for H in grid]
    if not all(a < b for a, b in zip(means, means[1:])):
        raise ValueError(f"seed-averaged skew is not monotonic in H (PT6): "
                         f"{dict(zip(grid, [round(x, 3) for x in means]))}. Nothing written")
    for m in manifests:
        write_manifest(m, out_dir)
    return manifests


def summary(manifests: list[dict], main_seeds) -> str:
    rows = ["    H   mean skew (cycles)   min train/client   min val/client"]
    for H in sorted({m["H"] for m in manifests}):
        ms = [m for m in manifests if m["H"] == H]
        mean = np.mean([m["skew"] for m in ms if m["seed"] in set(main_seeds)])
        rows.append(f"{H:5.2f}   {mean:18.3f}   {min(min(m['n_train_per_client']) for m in ms):16d}"
                    f"   {min(min(m['n_val_per_client']) for m in ms):14d}")
    return "\n".join(rows)


def main(argv=None) -> None:
    import yaml
    from rulfl.data.pipeline import build_dataset

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None, help="default: partition.manifests_dir from the config")
    ap.add_argument("--K", type=int, default=None, help="default: fl.K from the config")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = args.out or cfg["partition"]["manifests_dir"]
    manifests = make_all(cfg, build_dataset(cfg), out, args.K)
    print(f"wrote {len(manifests)} manifests to {out}/")
    print(summary(manifests, cfg["seeds"]["main"]))


if __name__ == "__main__":
    main()
