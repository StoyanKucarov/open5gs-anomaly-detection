#!/usr/bin/env python3
"""
TranAD: Deep Transformer Networks for Anomaly Detection (Tuli et al., VLDB 2022).
Encoder + two decoders; phase-2 decoder sees (x̂₁−x)² as focus context.
Loss = (1/n)·MSE(x̂₁,x) + (1−1/n)·MSE(x̂₂,x). Score = MSE(x̂₂,x).
d_model = 2×F; learned input/context projections retained for our small training set.
"""

import json
import math
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

WINDOW = 10


class _PositionalEncoding(nn.Module if _TORCH else object):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class _TranADNet(nn.Module if _TORCH else object):
    def __init__(self, n_features: int, nhead: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        d_model         = n_features * 2
        self.d_model    = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.ctx_proj   = nn.Linear(n_features * 2, d_model)

        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        dec_layer1 = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model,
            dropout=dropout, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, num_layers=n_layers)

        dec_layer2 = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model,
            dropout=dropout, batch_first=True)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, n_features)

    def _encode(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.pos_enc(self.input_proj(x))
        return self.encoder(h)

    def _decode(self, x: "torch.Tensor", c: "torch.Tensor",
                memory: "torch.Tensor", decoder: "nn.Module") -> "torch.Tensor":
        # c: zeros for phase-1, (x̂₁-x)² for phase-2
        tgt = self.pos_enc(self.ctx_proj(torch.cat([x, c], dim=-1)))
        h   = decoder(tgt, memory)
        return self.out_proj(h)

    def forward(self, x: "torch.Tensor"
                ) -> tuple["torch.Tensor", "torch.Tensor"]:
        z       = self._encode(x)
        zeros   = torch.zeros_like(x)
        x1      = self._decode(x, zeros,   z, self.decoder1)
        context = (x1.detach() - x) ** 2
        x2      = self._decode(x, context, z, self.decoder2)
        return x1, x2


class TranADDetector:
    def __init__(self, window: int = 10, nhead: int = 2,
                 n_layers: int = 1, dropout: float = 0.1):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window   = window
        self.nhead    = nhead
        self.n_layers = n_layers
        self.dropout  = dropout
        self._model:     Optional[_TranADNet] = None
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
            lr: float = 1e-3) -> "TranADDetector":
        vals = np.stack([r.values for r in records]).astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._mu  = vals.mean(axis=0)
        self._std = vals.std(axis=0)

        X, _ = self._to_windows(records)
        if len(X) == 0:
            raise ValueError("Not enough records to form windows.")
        n_feat = X.shape[2]

        # nhead must divide d_model = 2*n_feat
        nhead = self.nhead
        while (n_feat * 2) % nhead != 0 and nhead > 1:
            nhead -= 1

        self._model = _TranADNet(
            n_features=n_feat, nhead=nhead,
            n_layers=self.n_layers, dropout=self.dropout,
        ).to(self._device)
        opt   = torch.optim.AdamW(self._model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        self._model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for (xb,) in loader:
                xb       = xb.to(self._device)
                x1, x2   = self._model(xb)
                loss      = ((1/ep) * F.mse_loss(x1, xb)
                             + (1 - 1/ep) * F.mse_loss(x2, xb))
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            sched.step()
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(X):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + 3.0 * scores.std()) \
                          if len(scores) else 0.0
        print(f"[TranAD] Fitted: {len(X):,} windows, threshold={self._threshold:.4f}")
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
                xb      = X[i: i + 256].to(self._device)
                _, x2   = self._model(xb)
                s       = F.mse_loss(x2, xb, reduction="none"
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
            "window": self.window, "nhead": self.nhead,
            "n_layers": self.n_layers, "dropout": self.dropout,
            "threshold": self._threshold,
            "mu": self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[TranAD] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "TranADDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], nhead=meta["nhead"],
                   n_layers=meta["n_layers"], dropout=meta["dropout"])
        obj._threshold = meta["threshold"]
        obj._mu  = np.array(meta["mu"],  dtype=np.float32)
        obj._std = np.array(meta["std"], dtype=np.float32)
        n_feat   = N_FEATURES
        nhead    = meta["nhead"]
        while (n_feat * 2) % nhead != 0 and nhead > 1:
            nhead -= 1
        obj._model = _TranADNet(n_features=n_feat, nhead=nhead,
                                n_layers=obj.n_layers, dropout=obj.dropout)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = TranADDetector(window=10)
    model.fit(data["train"], epochs=20)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[TranAD] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")