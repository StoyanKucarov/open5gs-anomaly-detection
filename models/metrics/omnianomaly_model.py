#!/usr/bin/env python3
"""
OmniAnomaly: GRU + VAE for multivariate metric anomaly detection (Su et al., KDD 2019).
PyTorch reimplementation with z-score normalisation; score = reconstruction MSE.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import MetricRecord, N_FEATURES, FEATURE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

THRESHOLD_K = 3.0


class _OmniAnomalyNet(nn.Module if _TORCH else object):
    def __init__(self, input_dim: int, hidden_dim: int,
                 latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.enc_gru    = nn.GRU(input_dim, hidden_dim, n_layers,
                                 batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)

        self.fc_z2h     = nn.Linear(latent_dim, hidden_dim)
        self.dec_gru    = nn.GRU(hidden_dim, hidden_dim, n_layers,
                                 batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out     = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: "torch.Tensor"):
        h, _ = self.enc_gru(x)           # (B, W, hidden)
        h_last = h[:, -1, :]             # (B, hidden)
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, window: int):
        h = self.fc_z2h(z).unsqueeze(1).expand(-1, window, -1)  # (B,W,hidden)
        out, _ = self.dec_gru(h)
        return self.fc_out(out)          # (B, W, input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z, x.shape[1])
        return x_hat, mu, logvar


class OmniAnomalyDetector:
    def __init__(self,
                 window: int     = 20,
                 latent_dim: int = 8,
                 hidden_dim: int = 64,
                 n_layers: int   = 2,
                 dropout: float  = 0.1,
                 beta: float     = 0.01,
                 threshold_k: float = THRESHOLD_K):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window      = window
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.beta        = beta
        self.threshold_k = threshold_k
        self._model:     Optional[_OmniAnomalyNet] = None
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
            return torch.zeros((0, W, F)), torch.zeros(0)
        n_win = T - W + 1
        X_np  = np.stack([vals[i: i + W] for i in range(n_win)])
        L_np  = np.array([int(labels[i: i + W].any()) for i in range(n_win)])
        return (torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(L_np, dtype=torch.float32))

    def fit(self, records: list[MetricRecord],
            epochs: int = 20, batch_size: int = 64,
            lr: float = 1e-3) -> "OmniAnomalyDetector":
        vals = np.stack([r.values for r in records]).astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._mu  = vals.mean(axis=0)
        self._std = vals.std(axis=0)

        X, _ = self._to_windows(records)
        if len(X) == 0:
            raise ValueError("Not enough records to form windows.")
        n_feat = X.shape[2]

        self._model = _OmniAnomalyNet(
            input_dim=n_feat, hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim, n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        self._model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for (xb,) in loader:
                xb             = xb.to(self._device)
                x_hat, mu, lv  = self._model(xb)
                recon          = F.mse_loss(x_hat, xb)
                kl             = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
                loss           = recon + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
                total += recon.item() * len(xb)
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  recon={total/len(X):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std()) \
                          if len(scores) else 0.0
        print(f"[OmniAnomaly] Fitted: {len(X):,} windows, "
              f"threshold={self._threshold:.4f}")
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
                xb            = X[i: i + 256].to(self._device)
                x_hat, mu, lv = self._model(xb)
                s = F.mse_loss(x_hat, xb, reduction="none"
                               ).mean(dim=(1, 2)).cpu().numpy()
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
            "window": self.window, "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
            "dropout": self.dropout, "beta": self.beta,
            "threshold_k": self.threshold_k, "threshold": self._threshold,
            "mu": self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[OmniAnomaly] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "OmniAnomalyDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], latent_dim=meta["latent_dim"],
                   hidden_dim=meta["hidden_dim"], n_layers=meta["n_layers"],
                   dropout=meta["dropout"], beta=meta["beta"],
                   threshold_k=meta["threshold_k"])
        obj._threshold = meta["threshold"]
        obj._mu  = np.array(meta["mu"],  dtype=np.float32)
        obj._std = np.array(meta["std"], dtype=np.float32)
        obj._model = _OmniAnomalyNet(
            input_dim=N_FEATURES, hidden_dim=obj.hidden_dim,
            latent_dim=obj.latent_dim, n_layers=obj.n_layers,
            dropout=obj.dropout)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = OmniAnomalyDetector(window=20, latent_dim=8, hidden_dim=64, n_layers=2)
    model.fit(data["train"], epochs=20)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[OmniAnomaly] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")