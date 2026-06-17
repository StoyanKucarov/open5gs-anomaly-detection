#!/usr/bin/env python3
"""DeepLog: LSTM next-key prediction (Du et al., CCS 2017)."""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LogRecord, load_sequences, vocab_size as _vocab_size_fn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False


class _DeepLogLSTM(nn.Module if _TORCH else object):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc     = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])


class DeepLogDetector:
    def __init__(self,
                 window: int = 10,
                 step: int = 1,
                 top_k: int = 9,
                 embed_dim: int = 32,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        if not _TORCH:
            raise RuntimeError("PyTorch is required for DeepLog. Install with: pip install torch")
        self.window    = window
        self.step      = step
        self.top_k     = top_k
        self.embed_dim = embed_dim
        self.hidden    = hidden_dim
        self.n_layers  = n_layers
        self.dropout   = dropout
        self._vocab: int = 0
        self._model: Optional[_DeepLogLSTM] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_tensors(self, records: list[LogRecord]
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = load_sequences(records, window=self.window + 1, step=self.step,
                              group_by_slug=True)
        if not seqs:
            return torch.zeros((0, self.window), dtype=torch.long), torch.zeros(0, dtype=torch.long)

        X_list, y_list = [], []
        for ids, _ in seqs:
            if len(ids) < self.window + 1:
                continue
            X_list.append(ids[:self.window])
            y_list.append(ids[self.window])

        X = torch.tensor(X_list, dtype=torch.long)
        y = torch.tensor(y_list, dtype=torch.long)
        return X, y

    def fit(self, records: list[LogRecord],
            epochs: int = 20,
            batch_size: int = 512,
            lr: float = 1e-3) -> "DeepLogDetector":
        self._vocab = _vocab_size_fn()
        torch.manual_seed(42)
        self._model = _DeepLogLSTM(self._vocab, self.embed_dim,
                                   self.hidden, self.n_layers, self.dropout)
        self._model.to(self._device)

        X, y = self._build_tensors(records)
        if len(X) == 0:
            raise ValueError("No training sequences built.")
        print(f"[DeepLog] Training on {len(X):,} sequences, vocab={self._vocab}")

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt     = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                yb = yb.clamp(0, self._vocab - 1)
                xb = xb.clamp(0, self._vocab - 1)
                logits = self._model(xb)
                loss = criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(xb)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(X):.4f}")
        return self

    def _anomaly_scores(self, records: list[LogRecord]
                        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, labels) per window.
        score = 1.0 if true next key not in top-k, else 0.0.
        """
        self._model.eval()
        seqs = load_sequences(records, window=self.window + 1, step=self.step,
                              group_by_slug=True)
        scores, labels = [], []
        with torch.no_grad():
            batch_x, batch_true, batch_labels = [], [], []
            for ids, lbl in seqs:
                if len(ids) < self.window + 1:
                    continue
                batch_x.append(ids[:self.window])
                batch_true.append(ids[self.window])
                batch_labels.append(lbl)

            if not batch_x:
                return np.array([]), np.array([])

            X = torch.tensor(batch_x, dtype=torch.long).clamp(0, self._vocab - 1).to(self._device)
            logits = self._model(X)
            topk   = logits.topk(self.top_k, dim=1).indices.cpu().numpy()
            for i, true_key in enumerate(batch_true):
                anomaly = int(true_key not in topk[i])
                scores.append(float(anomaly))
                labels.append(batch_labels[i])

        return np.array(scores), np.array(labels, dtype=int)

    def score(self, records: list[LogRecord]) -> tuple[np.ndarray, np.ndarray]:
        return self._anomaly_scores(records)

    def predict(self, records: list[LogRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self._anomaly_scores(records)
        preds = (scores > 0.5).astype(int)
        return preds, scores, labels

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "vocab": self._vocab,
            "window": self.window, "step": self.step, "top_k": self.top_k,
            "embed_dim": self.embed_dim, "hidden": self.hidden,
            "n_layers": self.n_layers, "dropout": self.dropout,
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[DeepLog] Saved -> {path}_weights.pt + _meta.json")

    @classmethod
    def load(cls, path: Path) -> "DeepLogDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], step=meta["step"],
                   top_k=meta["top_k"], embed_dim=meta["embed_dim"],
                   hidden_dim=meta["hidden"], n_layers=meta["n_layers"],
                   dropout=meta["dropout"])
        obj._vocab = meta["vocab"]
        obj._model = _DeepLogLSTM(obj._vocab, obj.embed_dim, obj.hidden,
                                  obj.n_layers, obj.dropout)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all

    data  = load_all()
    train = data["train"]
    test  = data["test"]

    model = DeepLogDetector(window=10, step=1, top_k=9)
    model.fit(train, epochs=30, batch_size=512)

    preds, scores, labels = model.predict(test)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n[DeepLog] Test results — windows")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
