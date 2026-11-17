"""Head-refit probe (DECISIONS E6). Diagnostic only, not a deployable method.

Take a main run's best global model, keep its GRU frozen, put a freshly initialised head on top, train only the head
on ALL training windows (pooled, as if centralized) with M3 settings and early stopping on validation, then score on
test. If test MAE drops back near centralized, the skew damage was mainly in the head; if not, it is in the features.

The GRU is frozen, so its output for each window never changes: we compute it once (eval mode, no dropout) and train
the head on those fixed vectors. Same result as running the frozen GRU every batch, much faster.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from rulfl.data.pipeline import Dataset
from rulfl.model.gru import RULModel


def gru_features(model: RULModel, X: np.ndarray, batch: int = 1024) -> torch.Tensor:
    """The frozen GRU's memory after the last flight, one vector per window: (n, hidden), on the model's device."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            h, _ = model.gru(torch.from_numpy(X[i:i + batch]).to(dev))
            out.append(h[:, -1])
    return torch.cat(out)


def refit_head(model: RULModel, cfg: dict, ds: Dataset, seed: int) -> tuple[RULModel, dict]:
    """Returns (model with the original GRU and the refitted head, history like train_centralized's)."""
    m, t, cap = cfg["model"], cfg["train"], cfg["data"]["rul_cap"]
    dev = next(model.parameters()).device
    torch.use_deterministic_algorithms(True)
    F_tr, F_va = gru_features(model, ds.X_train), gru_features(model, ds.X_val)
    y = (torch.from_numpy(ds.y_train) / cap).to(dev)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = RULModel(ds.X_train.shape[2], m["hidden"], m["layers"], m["dropout"]).head.to(dev)   # fresh head (E6)
    opt = torch.optim.Adam(head.parameters(), lr=m["lr"])

    def val_mae() -> float:
        head.eval()
        with torch.no_grad():
            pred = (head(F_va).squeeze(-1) * cap).clamp(0, cap).cpu().numpy()
        head.train()
        return float(np.abs(pred - ds.y_val).mean())

    hist = {"train_loss": [], "val_mae": [], "best_epoch": 0}
    best, best_state = float("inf"), None
    for epoch in range(t["max_epochs"]):
        order = torch.from_numpy(rng.permutation(len(F_tr))).to(dev)
        total = 0.0
        for i in range(0, len(F_tr), t["batch"]):
            idx = order[i:i + t["batch"]]
            loss = torch.nn.functional.mse_loss(head(F_tr[idx]).squeeze(-1), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), t["clip"])
            opt.step()
            total += loss.item() * len(idx)
        hist["train_loss"].append(total / len(F_tr))
        v = val_mae()
        hist["val_mae"].append(v)
        if v < best:
            best, best_state, hist["best_epoch"] = v, copy.deepcopy(head.state_dict()), epoch
        elif epoch - hist["best_epoch"] >= t["patience"]:
            break

    head.load_state_dict(best_state)
    refit = copy.deepcopy(model)
    refit.head = head
    refit.eval()
    return refit, hist
