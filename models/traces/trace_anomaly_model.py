#!/usr/bin/env python3
"""
TraceAnomaly: Real NVP normalizing flow for trace anomaly detection (Liu et al., ISSRE 2020).
Score = −log p(x); falls back to GMM if PyTorch is unavailable.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import TraceRecord, N_FEATURES

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

THRESHOLD_K = 3.0


if _TORCH:
    class _CouplingLayer(nn.Module):
        def __init__(self, d: int, mask: torch.Tensor, hidden: int = 128):
            super().__init__()
            self.register_buffer("mask", mask.float())
            self.s_net = nn.Sequential(
                nn.Linear(d, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, d), nn.Tanh(),
            )
            self.t_net = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, d),
            )

        def forward(self, x: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
            x_m    = x * self.mask
            s      = self.s_net(x_m) * (1 - self.mask)
            t      = self.t_net(x_m) * (1 - self.mask)
            y      = x_m + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = (s * (1 - self.mask)).sum(dim=-1)
            return y, log_det

    class _RealNVP(nn.Module):
        def __init__(self, d: int, n_layers: int = 6, hidden: int = 128):
            super().__init__()
            layers = []
            for k in range(n_layers):
                mask = torch.zeros(d)
                if k % 2 == 0:
                    mask[::2] = 1.0
                else:
                    mask[1::2] = 1.0
                layers.append(_CouplingLayer(d, mask, hidden))
            self.layers = nn.ModuleList(layers)
            self._log2pi = float(np.log(2 * np.pi))

        def log_prob(self, x: torch.Tensor) -> torch.Tensor:
            log_det = torch.zeros(x.size(0), device=x.device)
            z = x
            for layer in self.layers:
                z, ld = layer(z)
                log_det = log_det + ld
            log_pz = -0.5 * (z ** 2 + self._log2pi).sum(dim=-1)
            return log_pz + log_det


class _GMMDensity:
    def __init__(self, n_components: int = 8):
        from sklearn.mixture import GaussianMixture
        self.gmm = GaussianMixture(n_components=n_components,
                                    covariance_type="full",
                                    n_init=3, random_state=42)

    def fit(self, X: np.ndarray) -> None:
        self.gmm.fit(X)

    def nll(self, X: np.ndarray) -> np.ndarray:
        return -self.gmm.score_samples(X).astype(np.float32)


class TraceAnomalyDetector:
    def __init__(self, n_layers: int = 6, hidden_dim: int = 128,
                 threshold_k: float = THRESHOLD_K):
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim
        self.threshold_k = threshold_k
        self._mean:      Optional[np.ndarray] = None
        self._std:       Optional[np.ndarray] = None
        self._model                           = None
        self._threshold: float               = float("inf")

    def _prep(self, records: list[TraceRecord]
              ) -> tuple[np.ndarray, np.ndarray]:
        X      = np.stack([r.values for r in records]).astype(np.float32)
        labels = np.array([r.label  for r in records], dtype=int)
        return X, labels

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / np.where(self._std > 0, self._std, 1.0)

    def fit(self, records: list[TraceRecord],
            epochs: int = 80, batch_size: int = 64,
            lr: float = 1e-3) -> "TraceAnomalyDetector":
        X, _ = self._prep(records)
        if len(X) == 0:
            raise ValueError("No training records.")
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        X_norm     = self._normalise(X).astype(np.float32)

        if _TORCH:
            self._fit_nvp(X_norm, epochs, batch_size, lr)
        else:
            print("[TraceAnomaly] PyTorch unavailable — using GMM fallback.")
            self._model = _GMMDensity(n_components=8)
            self._model.fit(X_norm)

        scores          = self._nll(X_norm)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std())
        print(f"[TraceAnomaly] Fitted: {len(X):,} windows, "
              f"threshold={self._threshold:.4f}")
        return self

    def _fit_nvp(self, X_norm: np.ndarray, epochs: int,
                 batch_size: int, lr: float) -> None:
        d          = X_norm.shape[1]
        self._model = _RealNVP(d, self.n_layers, self.hidden_dim)
        opt        = optim.Adam(self._model.parameters(), lr=lr)
        sched      = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xt         = torch.tensor(X_norm)
        loader     = DataLoader(TensorDataset(Xt),
                                batch_size=max(batch_size, 1), shuffle=True)

        self._model.train()
        for ep in range(epochs):
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = -self._model.log_prob(batch).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(batch)
            sched.step()
            if ep % 20 == 0:
                print(f"[TraceAnomaly] Epoch {ep:3d}/{epochs}  "
                      f"NLL={total/max(len(X_norm),1):.4f}")
        self._model.eval()

    def _nll(self, X_norm: np.ndarray) -> np.ndarray:
        if _TORCH and isinstance(self._model, _RealNVP):
            with torch.no_grad():
                return (-self._model.log_prob(
                    torch.tensor(X_norm))).numpy().astype(np.float32)
        return self._model.nll(X_norm)

    def score(self, records: list[TraceRecord]
              ) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, labels = self._prep(records)
        return self._nll(self._normalise(X).astype(np.float32)), labels

    def predict(self, records: list[TraceRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels


if __name__ == "__main__":
    from data_loader import load_all
    data = load_all()
    m    = TraceAnomalyDetector(n_layers=6, hidden_dim=128)
    m.fit(data["train"], epochs=80)
    preds, scores, labels = m.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    print(f"[TraceAnomaly] P={prec:.3f}  R={rec:.3f}  "
          f"F1={2*prec*rec/max(prec+rec,1e-9):.3f}")
