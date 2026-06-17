#!/usr/bin/env python3
"""
USAD: UnSupervised Anomaly Detection on Multivariate Time Series (Audibert et al., KDD 2020).
Score = α·‖AE1(x)−x‖² + (1−α)·‖AE2(AE1(x))−x‖².
Uses separate encoders rather than a shared one for training stability.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import MetricRecord, N_FEATURES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

WINDOW   = 5          # consecutive samples per input vector
Z_RATIO  = 4         # z_size = w_size // Z_RATIO
ALPHA    = 0.5       # weight on AE1 vs AE2 in anomaly score


def _build_mlp(d_in: int, d_hidden: int, d_out: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(d_in,    d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


class _USADNet(nn.Module if _TORCH else object):
    def __init__(self, w_size: int, z_size: int):
        super().__init__()
        h = max(z_size * 2, 32)
        self.enc1 = _build_mlp(w_size, h, z_size)
        self.dec1 = _build_mlp(z_size, h, w_size)
        self.enc2 = _build_mlp(w_size, h, z_size)
        self.dec2 = _build_mlp(z_size, h, w_size)

    def AE1(self, x):
        return self.dec1(self.enc1(x))

    def AE2(self, x):
        return self.dec2(self.enc2(x))

    def AE2_on_AE1(self, x):
        return self.dec2(self.enc2(self.dec1(self.enc1(x))))


class USADDetector:
    def __init__(self, window: int = 5, alpha: float = ALPHA,
                 threshold_k: float = 3.0):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window      = window
        self.alpha       = alpha
        self.threshold_k = threshold_k
        self._model:     Optional[_USADNet] = None
        self._mu:        Optional[np.ndarray] = None
        self._std:       Optional[np.ndarray] = None
        self._threshold: float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_windows(self, records: list[MetricRecord]
                    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        vals   = np.stack([r.values for r in records]).astype(np.float32)
        labels = np.array([r.label for r in records], dtype=np.float32)
        vals   = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if self._mu is not None:
            vals = (vals - self._mu) / (self._std + 1e-6)
        T, F = vals.shape
        W    = self.window
        if T < W:
            return torch.zeros((0, W * F)), torch.zeros(0)
        n_win = T - W + 1
        X_np  = np.stack([vals[i: i + W].reshape(-1) for i in range(n_win)])
        L_np  = np.array([int(labels[i: i + W].any()) for i in range(n_win)])
        return (torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(L_np, dtype=torch.float32))

    def fit(self, records: list[MetricRecord],
            epochs: int = 40, batch_size: int = 64,
            lr: float = 1e-3) -> "USADDetector":
        vals = np.stack([r.values for r in records]).astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._mu  = vals.mean(axis=0)
        self._std = vals.std(axis=0)

        X, _ = self._to_windows(records)
        if len(X) == 0:
            raise ValueError("Not enough records to form windows.")
        w_size = X.shape[1]
        z_size = max(w_size // Z_RATIO, 4)

        self._model = _USADNet(w_size, z_size).to(self._device)
        opt1 = torch.optim.Adam(
            list(self._model.enc1.parameters()) +
            list(self._model.dec1.parameters()), lr=lr)
        opt2 = torch.optim.Adam(
            list(self._model.enc2.parameters()) +
            list(self._model.dec2.parameters()), lr=lr)

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        self._model.train()
        for ep in range(1, epochs + 1):
            t1 = t2 = 0.0
            for (xb,) in loader:
                xb = xb.to(self._device)
                opt1.zero_grad()
                x1 = self._model.AE1(xb)
                x2 = self._model.AE2_on_AE1(xb)
                l1 = (1/ep) * F.mse_loss(x1, xb) + (1 - 1/ep) * F.mse_loss(x2, xb)
                l1.backward()
                opt1.step()
                t1 += l1.item() * len(xb)
                opt2.zero_grad()
                with torch.no_grad():
                    x1_d = self._model.AE1(xb)
                x2_real = self._model.AE2(xb)
                x2_ae1  = self._model.AE2(x1_d)
                l2 = ((1/ep) * F.mse_loss(x2_real, xb)
                      - (1 - 1/ep) * F.mse_loss(x2_ae1, xb))
                l2.backward()
                opt2.step()
                t2 += l2.item() * len(xb)
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  "
                      f"L1={t1/len(X):.4f}  L2={t2/len(X):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std()) \
                          if len(scores) else 0.0
        print(f"[USAD] Fitted: {len(X):,} windows, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[MetricRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, L = self._to_windows(records)
        if len(X) == 0:
            return np.array([]), np.array([])
        self._model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = X[i: i + 256].to(self._device)
                x1 = self._model.AE1(xb)
                x2 = self._model.AE2_on_AE1(xb)
                s  = (self.alpha * F.mse_loss(x1, xb, reduction="none")
                      + (1 - self.alpha) * F.mse_loss(x2, xb, reduction="none")
                      ).mean(dim=1).cpu().numpy()
                all_scores.append(s)
        return np.concatenate(all_scores), L.numpy().astype(int)

    def predict(self, records: list[MetricRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "window": self.window, "alpha": self.alpha,
            "threshold_k": self.threshold_k, "threshold": self._threshold,
            "mu": self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[USAD] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "USADDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], alpha=meta["alpha"],
                   threshold_k=meta["threshold_k"])
        obj._threshold = meta["threshold"]
        obj._mu  = np.array(meta["mu"],  dtype=np.float32)
        obj._std = np.array(meta["std"], dtype=np.float32)
        w_size   = obj.window * N_FEATURES
        z_size   = max(w_size // Z_RATIO, 4)
        obj._model = _USADNet(w_size, z_size).to(obj._device)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = USADDetector(window=5)
    model.fit(data["train"], epochs=40)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[USAD] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")