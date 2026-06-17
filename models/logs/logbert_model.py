#!/usr/bin/env python3
"""
LogBERT: MLM-pretrained Transformer on log key sequences (Guo et al., BigData 2021).
Score = fraction of positions where true key is not in top-k masked predictions.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LogRecord, load_sequences, vocab_size as _vocab_size_fn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False


class _LogBERT(nn.Module if _TORCH else object):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        B, T  = x.shape
        pos   = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h     = self.tok_embed(x) + self.pos_embed(pos)
        h     = self.encoder(h)
        return self.head(h)


class LogBERTDetector:
    def __init__(self, window: int = 10, step: int = 1, top_k: int = 9,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 mask_frac: float = 0.15, dropout: float = 0.1):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window    = window
        self.step      = step
        self.top_k     = top_k
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_layers  = n_layers
        self.mask_frac = mask_frac
        self.dropout   = dropout
        self._vocab:  int = 0
        self._model:  Optional[_LogBERT] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, records: list[LogRecord]
                          ) -> tuple["torch.Tensor", "torch.Tensor"]:
        seqs   = load_sequences(records, window=self.window,
                                step=self.step, group_by_slug=True)
        X_list, L_list = [], []
        for ids, lbl in seqs:
            if len(ids) < self.window:
                continue
            X_list.append(ids[:self.window])
            L_list.append(lbl)
        if not X_list:
            return (torch.zeros((0, self.window), dtype=torch.long),
                    torch.zeros(0))
        X = torch.tensor(X_list, dtype=torch.long).clamp(0, max(self._vocab - 1, 0))
        L = torch.tensor(L_list, dtype=torch.float32)
        return X, L

    def fit(self, records: list[LogRecord],
            epochs: int = 20, batch_size: int = 512) -> "LogBERTDetector":
        self._vocab = _vocab_size_fn()
        MASK        = self._vocab   # use vocab_size as the [MASK] token ID

        self._model = _LogBERT(
            vocab_size=self._vocab, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            d_ff=self.d_model * 4, max_len=self.window + 4,
            dropout=self.dropout,
        ).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        X, _ = self._build_sequences(records)
        if len(X) == 0:
            raise ValueError("No training sequences built.")
        print(f"[LogBERT] Training on {len(X):,} windows, vocab={self._vocab}")

        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        rng    = np.random.default_rng(42)
        self._model.train()
        for ep in range(1, epochs + 1):
            total = 0.0; n_masked = 0
            for (xb,) in loader:
                xb      = xb.clone()
                B, T    = xb.shape
                masked  = xb.clone()
                targets = xb.clone()
                mask_pos = torch.zeros(B, T, dtype=torch.bool)
                for i in range(B):
                    n_m = max(1, int(T * self.mask_frac))
                    pos = rng.choice(T, size=n_m, replace=False)
                    mask_pos[i, pos] = True
                masked[mask_pos] = MASK
                logits  = self._model(masked.to(self._device))
                mp      = mask_pos.to(self._device)
                loss    = nn.functional.cross_entropy(
                    logits[mp], targets.to(self._device)[mp].clamp(0, self._vocab - 1)
                )
                opt.zero_grad(); loss.backward(); opt.step()
                total    += loss.item() * int(mask_pos.sum())
                n_masked += int(mask_pos.sum())
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  "
                      f"mlm_loss={total/max(n_masked,1):.4f}")
        return self

    def score(self, records: list[LogRecord]) -> tuple[np.ndarray, np.ndarray]:
        X, L = self._build_sequences(records)
        if len(X) == 0:
            return np.array([]), np.array([])
        MASK = self._vocab
        B, T = X.shape
        self._model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, B, 64):
                xb = X[i: i + 64]
                b  = xb.shape[0]
                anom = torch.zeros(b, T)
                for pos in range(T):
                    masked          = xb.clone().to(self._device)
                    masked[:, pos]  = MASK
                    logits          = self._model(masked)             # (b, T, vocab+1)
                    preds_at_pos    = logits[:, pos, :].topk(
                        self.top_k, dim=-1).indices.cpu()
                    true_at_pos     = xb[:, pos].unsqueeze(1)
                    anom[:, pos]    = ~(preds_at_pos == true_at_pos).any(dim=1)
                all_scores.append(anom.mean(dim=1).numpy())
        return np.concatenate(all_scores), L.numpy().astype(int)

    def predict(self, records: list[LogRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > 0.5).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "vocab": self._vocab, "window": self.window, "step": self.step,
            "top_k": self.top_k, "d_model": self.d_model,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "mask_frac": self.mask_frac, "dropout": self.dropout,
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[LogBERT] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "LogBERTDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], step=meta["step"],
                   top_k=meta["top_k"], d_model=meta["d_model"],
                   n_heads=meta["n_heads"], n_layers=meta["n_layers"],
                   mask_frac=meta["mask_frac"], dropout=meta["dropout"])
        obj._vocab = meta["vocab"]
        obj._model = _LogBERT(
            vocab_size=obj._vocab, d_model=obj.d_model,
            n_heads=obj.n_heads, n_layers=obj.n_layers,
            d_ff=obj.d_model * 4, max_len=obj.window + 4,
            dropout=obj.dropout)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = LogBERTDetector(window=10, step=1, top_k=9, d_model=64, n_heads=4, n_layers=2)
    model.fit(data["train"], epochs=20, batch_size=512)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[LogBERT] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")