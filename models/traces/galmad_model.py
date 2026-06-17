#!/usr/bin/env python3
"""
GAL-MAD: GAT + BiLSTM encoder-decoder over sequences of 30-s trace windows.
Attanayake et al., arXiv 2504.00058. Uses static 3GPP NF topology (no call chain data).
W=8 consecutive windows (4 min) per sequence, anomaly score = reconstruction MSE.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "analysis"))
from data_loader import TraceRecord, SERVICES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

N_NODES  = len(SERVICES)
SVC_IDX  = {s: i for i, s in enumerate(SERVICES)}
D_NODE   = 4                     # per-service features in TraceRecord.values

_SVC_BASE = {s: i * 4 for i, s in enumerate(SERVICES)}

# 3GPP NF reference topology — bidirectional edges + self-loops
# Services: amf=0 ausf=1 bsf=2 nrf=3 nssf=4 pcf=5 scp=6 sepp=7 smf=8 udr=9 udm=10
_EDGES = [
    (i, i) for i in range(N_NODES)          # self-loops
] + [
    (0, 8),  (8, 0),    # AMF ↔ SMF  (N11)
    (0, 1),  (1, 0),    # AMF ↔ AUSF (Nausf)
    (0, 10), (10, 0),   # AMF ↔ UDM  (Nudm)
    (0, 3),  (3, 0),    # AMF ↔ NRF  (Nnrf)
    (0, 4),  (4, 0),    # AMF ↔ NSSF (Nnssf)
    (0, 6),  (6, 0),    # AMF ↔ SCP  (proxy)
    (8, 3),  (3, 8),    # SMF ↔ NRF  (Nnrf)
    (8, 10), (10, 8),   # SMF ↔ UDM  (Nudm-SDM)
    (8, 5),  (5, 8),    # SMF ↔ PCF  (Npcf)
    (8, 6),  (6, 8),    # SMF ↔ SCP  (proxy)
    (3, 6),  (6, 3),    # NRF ↔ SCP  (proxy)
    (10, 9), (9, 10),   # UDM ↔ UDR  (Nudr)
    (5, 9),  (9, 5),    # PCF ↔ UDR  (Nudr)
    (5, 2),  (2, 5),    # PCF ↔ BSF  (Nbsf)
    (7, 3),  (3, 7),    # SEPP ↔ NRF
    (4, 3),  (3, 4),    # NSSF ↔ NRF
]


def _build_adj(n: int, edges: list[tuple[int, int]]) -> "torch.Tensor":
    A = torch.zeros(n, n)
    for i, j in edges:
        A[i, j] = 1.0
    deg  = A.sum(dim=1).clamp(min=1).sqrt()
    return A / deg.unsqueeze(1) / deg.unsqueeze(0)


def _node_features(record: TraceRecord) -> np.ndarray:
    X = np.zeros((N_NODES, D_NODE), dtype=np.float32)
    vals = record.values
    for svc, base in _SVC_BASE.items():
        i = SVC_IDX[svc]
        X[i] = vals[base: base + D_NODE]
    return X


def _make_sequences(records: list[TraceRecord],
                    W: int, mu: np.ndarray,
                    std: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Slide window of W consecutive records per experiment slug.
    Returns (X_seqs, labels) tensors shaped (n_seq, W, N, D) and (n_seq,).
    """
    by_slug: dict[str, list[TraceRecord]] = defaultdict(list)
    for r in records:
        by_slug[r.slug].append(r)

    Xs, Ls = [], []
    for recs in by_slug.values():
        recs.sort(key=lambda r: r.window_us)
        for start in range(len(recs) - W + 1):
            seq   = recs[start: start + W]
            x_seq = np.stack([_node_features(r) for r in seq])
            x_norm = (x_seq - mu) / (std + 1e-6)
            label  = int(any(r.label == 1 for r in seq))
            Xs.append(x_norm)
            Ls.append(label)
    if not Xs:
        return torch.zeros((0, W, N_NODES, D_NODE)), torch.zeros(0)
    return (torch.tensor(np.stack(Xs), dtype=torch.float32),
            torch.tensor(Ls, dtype=torch.float32))


class _GATLayer(nn.Module if _TORCH else object):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W  = nn.Linear(d_in, d_out, bias=False)
        self.a  = nn.Linear(2 * d_out, 1, bias=False)

    def forward(self, x: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
        h   = self.W(x)
        N   = h.size(1)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e   = F.leaky_relu(
            self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
        mask  = (adj > 0).unsqueeze(0)
        e     = e.masked_fill(~mask, -1e9)
        alpha = F.softmax(e, dim=2)
        return F.elu(torch.bmm(alpha, h))


class _GALMADNet(nn.Module if _TORCH else object):
    def __init__(self, d_node: int, d_gat: int, d_lstm: int,
                 n_nodes: int, adj: "torch.Tensor"):
        super().__init__()
        self.d_gat  = d_gat
        self.n_nodes = n_nodes
        d_flat = n_nodes * d_gat

        self.gat_e1   = _GATLayer(d_node, d_gat)
        self.gat_e2   = _GATLayer(d_gat,  d_gat)
        self.bilstm   = nn.LSTM(d_flat, d_lstm, batch_first=True,
                                bidirectional=True)
        self.proj_enc = nn.Linear(d_lstm * 2, d_lstm)

        self.lstm_dec = nn.LSTM(d_lstm, d_flat, batch_first=True)
        self.gat_d1   = _GATLayer(d_gat,  d_gat)
        self.gat_d2   = _GATLayer(d_gat,  d_node)

        self.register_buffer("adj", adj)

    def _gat_encode(self, x_bt: "torch.Tensor") -> "torch.Tensor":
        h = self.gat_e1(x_bt, self.adj)
        h = self.gat_e2(h,    self.adj)
        return h.flatten(1)

    def _gat_decode(self, h_bt: "torch.Tensor") -> "torch.Tensor":
        h = h_bt.view(-1, self.n_nodes, self.d_gat)
        h = self.gat_d1(h, self.adj)
        return self.gat_d2(h, self.adj)    # (B, N, d_node)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        B, W, N, D = x.shape

        enc_seq = torch.stack(
            [self._gat_encode(x[:, t]) for t in range(W)], dim=1)
        _, (hn, _) = self.bilstm(enc_seq)
        z = self.proj_enc(torch.cat([hn[0], hn[1]], dim=1))

        z_rep      = z.unsqueeze(1).expand(-1, W, -1)
        dec_seq, _ = self.lstm_dec(z_rep)
        x_hat = torch.stack(
            [self._gat_decode(dec_seq[:, t]) for t in range(W)], dim=1)
        return x_hat


class GALMADDetector:

    def __init__(self, seq_len: int = 8,
                 d_gat: int = 16, d_lstm: int = 32):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.seq_len = seq_len
        self.d_gat   = d_gat
        self.d_lstm  = d_lstm
        self._mu:  Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._net: Optional[_GALMADNet] = None
        self._threshold: float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_net(self) -> "_GALMADNet":
        adj = _build_adj(N_NODES, _EDGES).to(self._device)
        return _GALMADNet(D_NODE, self.d_gat, self.d_lstm,
                          N_NODES, adj).to(self._device)

    def _score_seqs(self, X: "torch.Tensor") -> np.ndarray:
        self._net.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb    = X[i: i + 256].to(self._device)
                x_hat = self._net(xb)
                mse   = F.mse_loss(x_hat, xb, reduction="none")
                scores.append(mse.mean(dim=(1, 2, 3)).cpu().numpy())
        return np.concatenate(scores)

    def fit(self, records: list[TraceRecord],
            epochs: int = 60, batch_size: int = 64,
            lr: float = 1e-3) -> "GALMADDetector":
        if not records:
            raise ValueError("No training records.")

        Xs_np = np.stack([_node_features(r) for r in records])
        self._mu  = Xs_np.mean(axis=0).astype(np.float32)
        self._std = Xs_np.std(axis=0).astype(np.float32)

        X_all, L_all = _make_sequences(records, self.seq_len,
                                       self._mu, self._std)
        if len(X_all) == 0:
            raise ValueError("Not enough records for sequences.")
        print(f"[GAL-MAD] {len(X_all):,} training sequences "
              f"(W={self.seq_len}, N={N_NODES}, D={D_NODE})")

        self._net = self._build_net()
        loader    = DataLoader(TensorDataset(X_all),
                               batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)

        self._net.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for (xb,) in loader:
                xb    = xb.to(self._device)
                x_hat = self._net(xb)
                loss  = F.mse_loss(x_hat, xb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  mse={total/len(X_all):.5f}")

        tr_scores = self._score_seqs(X_all)
        self._threshold = float(np.percentile(tr_scores, 95))
        print(f"[GAL-MAD] Fitted, threshold={self._threshold:.5f}")
        return self

    def score(self, records: list[TraceRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, L = _make_sequences(records, self.seq_len, self._mu, self._std)
        if len(X) == 0:
            return np.array([]), np.array([])
        return self._score_seqs(X), L.numpy().astype(int)

    def predict(self, records: list[TraceRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), str(path) + "_weights.pt")
        meta = {
            "seq_len": self.seq_len, "d_gat": self.d_gat,
            "d_lstm": self.d_lstm, "threshold": self._threshold,
            "mu":  self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[GAL-MAD] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "GALMADDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(seq_len=meta["seq_len"], d_gat=meta["d_gat"],
                   d_lstm=meta["d_lstm"])
        obj._threshold = meta["threshold"]
        obj._mu  = np.array(meta["mu"],  dtype=np.float32)
        obj._std = np.array(meta["std"], dtype=np.float32)
        obj._net = obj._build_net()
        obj._net.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._net.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = GALMADDetector()
    model.fit(data["train"], epochs=60)
    preds, scores, labels = model.predict(data["test"])
    if len(preds):
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"[GAL-MAD] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    else:
        print("[GAL-MAD] No predictions generated.")
