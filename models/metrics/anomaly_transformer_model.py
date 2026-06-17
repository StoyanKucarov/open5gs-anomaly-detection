#!/usr/bin/env python3
"""
Anomaly Transformer for 5G Core metric anomaly detection.
Xu et al., ICLR 2022. Score = recon_error / association_discrepancy.
Replaced MSTSA; this score doesn't invert on crash/OOM faults.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Insert metrics directory first so its data_loader shadows the logs one
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from data_loader import MetricRecord, FEATURE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

N_FEATURES = len(FEATURE_NAMES)


class _AnomalyAttention(nn.Module if _TORCH else object):
    """Single anomaly-attention layer returning (output, series_assoc, prior_assoc)."""

    def __init__(self, d_model: int, n_heads: int, window: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.window  = window

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable sigma for the Gaussian prior (one per head).
        self.log_sigma = nn.Parameter(torch.zeros(n_heads))

    def _prior(self, sigma: torch.Tensor) -> torch.Tensor:
        """Gaussian prior: (n_heads, T, T) normalised per row."""
        T   = self.window
        idx = torch.arange(T, dtype=torch.float32, device=sigma.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()    # (T, T)
        # sigma: (n_heads,) → (n_heads, 1, 1)
        s   = sigma.view(-1, 1, 1)
        p   = torch.exp(-dist.unsqueeze(0) ** 2 / (2 * s ** 2 + 1e-6))
        return p / (p.sum(dim=-1, keepdim=True) + 1e-9)

    def forward(self, x: "torch.Tensor"):
        B, T, _ = x.shape
        H, Dh   = self.n_heads, self.d_head

        def split(y):
            return y.view(B, T, H, Dh).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scale   = Dh ** -0.5
        scores  = (Q @ K.transpose(-2, -1)) * scale
        series  = F.softmax(scores, dim=-1)

        sigma  = self.log_sigma.exp().clamp(1e-3, float(T))
        prior  = self._prior(sigma).unsqueeze(0).expand(B, -1, -1, -1)

        out = (series @ V).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), series, prior


class _AnomalyTransformerLayer(nn.Module if _TORCH else object):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, window: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn  = _AnomalyAttention(d_model, n_heads, window)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h, series, prior = self.attn(x)
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, series, prior


class _AnomalyTransformerNet(nn.Module if _TORCH else object):
    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, window: int):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.layers      = nn.ModuleList([
            _AnomalyTransformerLayer(d_model, n_heads, d_ff, window)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.input_proj(x)
        all_series, all_prior = [], []
        for layer in self.layers:
            h, series, prior = layer(h)
            all_series.append(series)
            all_prior.append(prior)
        recon = self.output_proj(h)
        return recon, all_series, all_prior


def _assoc_discrepancy(series_list: list, prior_list: list) -> "torch.Tensor":
    """
    Mean symmetric KL divergence between series and prior across all layers
    and heads, per batch item.  Returns shape (B,).
    """
    eps = 1e-9
    kl_sum = None
    n = 0
    for series, prior in zip(series_list, prior_list):
        kl_sp = (series * ((series + eps).log() - (prior + eps).log())).sum(-1)
        kl_ps = (prior  * ((prior  + eps).log() - (series + eps).log())).sum(-1)
        kl    = 0.5 * (kl_sp + kl_ps)
        kl    = kl.mean(dim=(1, 2))
        kl_sum = kl if kl_sum is None else kl_sum + kl
        n += 1
    return kl_sum / max(n, 1)


class AnomalyTransformerDetector:

    def __init__(self, window: int = 20, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window   = window
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self._model: _AnomalyTransformerNet | None = None
        self._mu:    np.ndarray | None = None
        self._std:   np.ndarray | None = None
        self._threshold: float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_windows(self, records: list[MetricRecord]
                    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Slide a window over records, return (X, labels) tensors."""
        vals   = np.stack([r.values for r in records]).astype(np.float32)
        labels = np.array([r.label for r in records], dtype=np.float32)
        vals   = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        if self._mu is not None:
            vals = (vals - self._mu) / (self._std + 1e-6)

        T, F  = vals.shape
        W     = self.window
        if T < W:
            return torch.zeros((0, W, F)), torch.zeros(0)

        n_win = T - W + 1
        X_np  = np.stack([vals[i: i + W] for i in range(n_win)])
        L_np  = np.array([int(labels[i: i + W].any()) for i in range(n_win)])
        return (torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(L_np, dtype=torch.float32))

    def fit(self, records: list[MetricRecord],
            epochs: int = 30, batch_size: int = 64,
            lr: float = 1e-3) -> "AnomalyTransformerDetector":
        vals = np.stack([r.values for r in records]).astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._mu  = vals.mean(axis=0)
        self._std = vals.std(axis=0)

        X, _ = self._to_windows(records)
        if len(X) == 0:
            raise ValueError("Not enough records to form windows.")
        _, W, n_feat = X.shape

        self._model = _AnomalyTransformerNet(
            n_features=n_feat, d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, d_ff=self.d_model * 4, window=W,
        ).to(self._device)

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        opt    = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        for epoch in range(1, epochs + 1):
            total = 0.0
            for (xb,) in loader:
                xb            = xb.to(self._device)
                recon, sl, pl = self._model(xb)
                loss_recon    = F.mse_loss(recon, xb)
                loss_disc     = _assoc_discrepancy(sl, pl).mean()
                # Minimise reconstruction; also encourage association discrepancy
                # to be large for normal data (so anomalies score high when it drops).
                loss          = loss_recon - 0.1 * loss_disc
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss_recon.item() * len(xb)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}  recon={total/len(X):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + 3.0 * scores.std()) if len(scores) else 0.0
        print(f"[AnomalyTransformer] Fitted: {len(X):,} windows, "
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
        bs = 256
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb            = X[i: i + bs].to(self._device)
                recon, sl, pl = self._model(xb)
                recon_err     = F.mse_loss(recon, xb, reduction="none"
                                           ).mean(dim=(1, 2)).cpu().numpy()
                disc          = _assoc_discrepancy(sl, pl).cpu().numpy()
                # High recon + low discrepancy → anomalous
                all_scores.append(recon_err / (disc + 1e-6))
        return np.concatenate(all_scores), L.numpy().astype(int)

    def predict(self, records: list[MetricRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        return (scores > self._threshold).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "window": self.window, "d_model": self.d_model,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "threshold": self._threshold,
            "mu": self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "AnomalyTransformerDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], d_model=meta["d_model"],
                   n_heads=meta["n_heads"], n_layers=meta["n_layers"])
        obj._threshold = meta["threshold"]
        obj._mu        = np.array(meta["mu"],  dtype=np.float32)
        obj._std       = np.array(meta["std"], dtype=np.float32)
        n_feat         = len(FEATURE_NAMES)
        W              = meta["window"]
        obj._model = _AnomalyTransformerNet(
            n_features=n_feat, d_model=obj.d_model, n_heads=obj.n_heads,
            n_layers=obj.n_layers, d_ff=obj.d_model * 4, window=W)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj
