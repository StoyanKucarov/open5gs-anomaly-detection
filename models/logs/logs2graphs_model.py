#!/usr/bin/env python3
"""
Logs2Graphs: directed log transition graph + OCDiGCN one-class anomaly detection.
Li et al., arXiv 2307.00527 (ICSE 2024). Each 30s window becomes a weighted directed
graph over template IDs; SVDD pulls normal graph embeddings toward a hypersphere.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LogRecord

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _build_raw_graphs(records: list[LogRecord],
                      window_ns: int = 30_000_000_000,
                      ) -> list[dict]:
    by_slug: dict[str, list[LogRecord]] = defaultdict(list)
    for r in records:
        by_slug[r.slug].append(r)

    graphs = []
    for slug_records in by_slug.values():
        slug_records.sort(key=lambda r: r.timestamp_ns)
        if not slug_records:
            continue
        t_start = slug_records[0].timestamp_ns
        t_end   = slug_records[-1].timestamp_ns
        w_start = t_start
        while w_start <= t_end:
            w_end  = w_start + window_ns
            window = [r for r in slug_records
                      if w_start <= r.timestamp_ns < w_end]
            w_start = w_end
            if len(window) < 2:
                continue

            # Node set: unique template IDs in order of first appearance
            unique_ids = list(dict.fromkeys(r.template_id for r in window))
            node_map   = {tid: i for i, tid in enumerate(unique_ids)}
            n          = len(unique_ids)

            # Directed edges: consecutive template transitions
            edge_counts: dict[tuple[int, int], int] = defaultdict(int)
            for a, b in zip(window[:-1], window[1:]):
                si = node_map[a.template_id]
                di = node_map[b.template_id]
                edge_counts[(si, di)] += 1

            A_fwd = np.zeros((n, n), dtype=np.float32)
            A_bwd = np.zeros((n, n), dtype=np.float32)
            for (si, di), w in edge_counts.items():
                A_fwd[si, di] += w
                A_bwd[di, si] += w

            # Row-normalise
            def _rownorm(A):
                d = A.sum(axis=1, keepdims=True).clip(min=1)
                return A / d

            label = int(any(r.label == 1 for r in window))
            graphs.append({
                "ids":     unique_ids,
                "adj_fwd": _rownorm(A_fwd),
                "adj_bwd": _rownorm(A_bwd),
                "label":   label,
            })
    return graphs


class _DiGCNInceptionLayer(nn.Module if _TORCH else object):
    # no bias to prevent SVDD hypersphere collapse (Ruff et al., 2018)
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_fwd1 = nn.Linear(d_in, d_out, bias=False)   # k=1 forward
        self.W_bwd1 = nn.Linear(d_in, d_out, bias=False)   # k=1 backward
        self.W_fwd2 = nn.Linear(d_in, d_out, bias=False)   # k=2 forward
        self.W_bwd2 = nn.Linear(d_in, d_out, bias=False)   # k=2 backward

    def forward(self, x, adj_fwd, adj_bwd):
        # x: (n, d_in), adj_*: (n, n)
        adj_fwd2 = adj_fwd @ adj_fwd   # 2-hop forward paths
        adj_bwd2 = adj_bwd @ adj_bwd   # 2-hop backward paths
        h1 = adj_fwd  @ self.W_fwd1(x) + adj_bwd  @ self.W_bwd1(x)
        h2 = adj_fwd2 @ self.W_fwd2(x) + adj_bwd2 @ self.W_bwd2(x)
        return F.relu(h1 + h2)


class _OCDiGCN(nn.Module if _TORCH else object):
    def __init__(self, d_emb: int, d_hidden: int):
        super().__init__()
        self.conv1 = _DiGCNInceptionLayer(d_emb,    d_hidden)
        self.conv2 = _DiGCNInceptionLayer(d_hidden, d_hidden)
        self.proj  = nn.Linear(d_hidden, d_hidden, bias=False)

    def forward(self, x, adj_fwd, adj_bwd):
        h = self.conv1(x, adj_fwd, adj_bwd)
        h = self.conv2(h, adj_fwd, adj_bwd)
        g = h.mean(dim=0)
        return self.proj(g)


class Logs2GraphsDetector:

    def __init__(self, window_ns: int = 30_000_000_000,
                 d_emb: int = 32, d_hidden: int = 64):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window_ns  = window_ns
        self.d_emb      = d_emb
        self.d_hidden   = d_hidden
        self._vocab:    int = 0
        self._emb:      Optional[nn.Embedding] = None
        self._net:      Optional[_OCDiGCN] = None
        self._center:   Optional["torch.Tensor"] = None
        self._threshold: float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _graph_to_tensors(self, g: dict) -> tuple:
        """Convert one raw graph dict to device tensors for inference."""
        vocab    = self._vocab
        ids      = torch.tensor([i % vocab for i in g["ids"]],
                                 dtype=torch.long, device=self._device)
        adj_fwd  = torch.tensor(g["adj_fwd"], dtype=torch.float32,
                                 device=self._device)
        adj_bwd  = torch.tensor(g["adj_bwd"], dtype=torch.float32,
                                 device=self._device)
        return ids, adj_fwd, adj_bwd

    def _embed_and_score(self, g: dict) -> "torch.Tensor":
        """Forward pass: embed → DiGCN → SVDD distance."""
        ids, adj_fwd, adj_bwd = self._graph_to_tensors(g)
        x   = self._emb(ids)                    # (n, d_emb) — differentiable
        emb = self._net(x, adj_fwd, adj_bwd)    # (d_hidden,)
        return (emb - self._center).pow(2).sum()

    def fit(self, records: list[LogRecord],
            epochs: int = 30, lr: float = 1e-3,
            weight_decay: float = 1e-4) -> "Logs2GraphsDetector":
        from data_loader import vocab_size
        self._vocab = vocab_size() or 4096

        self._emb = nn.Embedding(self._vocab, self.d_emb).to(self._device)
        self._net = _OCDiGCN(self.d_emb, self.d_hidden).to(self._device)

        raw = _build_raw_graphs(records, self.window_ns)
        raw = [g for g in raw if len(g["ids"]) >= 2]
        if not raw:
            raise ValueError("No valid graphs built from training records.")
        print(f"[Logs2Graphs] {len(raw):,} training graphs (vocab={self._vocab})")

        # Initialise SVDD centre as mean of initial embeddings
        self._net.eval(); self._emb.eval()
        with torch.no_grad():
            embeds = []
            for g in raw:
                ids, af, ab = self._graph_to_tensors(g)
                x   = self._emb(ids)
                embeds.append(self._net(x, af, ab))
            self._center = torch.stack(embeds).mean(dim=0).detach()

        # Train: gradients flow through _emb and _net
        params = list(self._net.parameters()) + list(self._emb.parameters())
        opt    = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        indices = list(range(len(raw)))

        self._net.train(); self._emb.train()
        for ep in range(1, epochs + 1):
            np.random.shuffle(indices)
            total = 0.0
            for i in indices:
                loss = self._embed_and_score(raw[i])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  "
                      f"svdd_loss={total/len(raw):.4f}")

        self._net.eval(); self._emb.eval()
        with torch.no_grad():
            tr_scores = np.array([self._embed_and_score(g).item() for g in raw])
        self._threshold = float(tr_scores.mean() + 3.0 * tr_scores.std())
        print(f"[Logs2Graphs] Fitted, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[LogRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        raw  = _build_raw_graphs(records, self.window_ns)
        raw  = [g for g in raw if len(g["ids"]) >= 2]
        if not raw:
            return np.array([]), np.array([])
        self._net.eval(); self._emb.eval()
        with torch.no_grad():
            scores = np.array([self._embed_and_score(g).item() for g in raw],
                              dtype=np.float32)
        labels = np.array([g["label"] for g in raw], dtype=int)
        return scores, labels

    def predict(self, records: list[LogRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        preds = (scores > self._threshold).astype(int)
        return preds, scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(),  str(path) + "_net.pt")
        torch.save(self._emb.state_dict(),  str(path) + "_emb.pt")
        torch.save(self._center,            str(path) + "_center.pt")
        meta = {
            "window_ns": self.window_ns, "d_emb": self.d_emb,
            "d_hidden": self.d_hidden, "vocab": self._vocab,
            "threshold": self._threshold,
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[Logs2Graphs] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "Logs2GraphsDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window_ns=meta["window_ns"], d_emb=meta["d_emb"],
                   d_hidden=meta["d_hidden"])
        obj._vocab     = meta["vocab"]
        obj._threshold = meta["threshold"]
        obj._emb = nn.Embedding(obj._vocab, obj.d_emb).to(obj._device)
        obj._net = _OCDiGCN(obj.d_emb, obj.d_hidden).to(obj._device)
        obj._emb.load_state_dict(
            torch.load(str(path) + "_emb.pt",    map_location=obj._device))
        obj._net.load_state_dict(
            torch.load(str(path) + "_net.pt",    map_location=obj._device))
        obj._center = torch.load(str(path) + "_center.pt",
                                  map_location=obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = Logs2GraphsDetector()
    model.fit(data["train"], epochs=30)
    preds, scores, labels = model.predict(data["test"])
    if len(preds):
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"[Logs2Graphs] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    else:
        print("[Logs2Graphs] No test predictions generated.")
