#!/usr/bin/env python3
"""
TraceDAE: dual autoencoder on per-window service trace graphs.
Li et al., IEEE TNSM 2025. Structure AE (GAT) catches path anomalies;
attribute AE (MLP) catches latency anomalies. No parent-child span links needed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import TraceRecord, SERVICES, N_FEATURES

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
D_NODE   = 4                      # span_count, error_rate, log_mean_dur, log_p95_dur
# Slice offsets inside TraceRecord.values (44 per-service features first)
_SVC_BASE = {s: i * 4 for i, s in enumerate(SERVICES)}

def _build_adjacency(spans: list[dict]) -> np.ndarray:
    """
    Build binary adjacency matrix from consecutive service pairs within
    each trace_id, sorted by start_us. No parent-child links needed.
    """
    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for sp in spans:
        svc = sp.get("service", "")
        if svc in SVC_IDX:
            by_trace[sp["trace_id"]].append(sp)
    for trace_spans in by_trace.values():
        trace_spans.sort(key=lambda s: s["start_us"])
        for a, b in zip(trace_spans[:-1], trace_spans[1:]):
            si = SVC_IDX.get(a["service"])
            sj = SVC_IDX.get(b["service"])
            if si is not None and sj is not None:
                A[si, sj] = 1.0
    return A


def _node_features(record: TraceRecord) -> np.ndarray:
    """Extract (N_NODES, D_NODE) feature matrix from a TraceRecord."""
    X = np.zeros((N_NODES, D_NODE), dtype=np.float32)
    vals = record.values
    for svc, base in _SVC_BASE.items():
        i = SVC_IDX[svc]
        X[i, 0] = vals[base + 0]   # span_count
        X[i, 1] = vals[base + 1]   # error_rate
        X[i, 2] = vals[base + 2]   # log_mean_dur
        X[i, 3] = vals[base + 3]   # log_p95_dur
    return X


def _build_stg(record: TraceRecord
               ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X: (N,D), A: (N,N)) for one TraceRecord."""
    X = _node_features(record)
    A = _build_adjacency(record.spans) if record.spans else np.zeros(
        (N_NODES, N_NODES), dtype=np.float32)
    return X, A


class _GATLayer(nn.Module if _TORCH else object):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W  = nn.Linear(d_in, d_out, bias=False)
        self.a  = nn.Linear(2 * d_out, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
        unbatched = x.dim() == 2
        if unbatched:
            x   = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
        B, N, _ = x.shape
        h   = self.W(x)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e   = self.lrelu(
            self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        # Mask non-edges (keep self-loops always)
        eye  = torch.eye(N, device=adj.device).unsqueeze(0)
        mask = (adj > 0) | eye.bool()
        e    = e.masked_fill(~mask, -1e9)
        alpha = F.softmax(e, dim=2)
        out  = F.elu(torch.bmm(alpha, h))
        return out.squeeze(0) if unbatched else out


class _StructureAE(nn.Module if _TORCH else object):
    def __init__(self, d_node: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.gat1 = _GATLayer(d_node,   d_hidden)
        self.gat2 = _GATLayer(d_hidden, d_latent)

    def encode(self, x, adj):
        h = self.gat1(x, adj)
        return self.gat2(h, adj)     # Z₁

    def decode(self, z):
        # Inner-product: Â = sigmoid(Z Z^T)
        if z.dim() == 2:
            return torch.sigmoid(z @ z.T)
        return torch.sigmoid(torch.bmm(z, z.transpose(1, 2)))

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z), z


class _AttributeAE(nn.Module if _TORCH else object):
    def __init__(self, d_in: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_latent))
        self.dec = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in))

    def forward(self, x_flat):
        z = self.enc(x_flat)
        return self.dec(z), z


class _TraceDAENet(nn.Module if _TORCH else object):
    def __init__(self, d_node, d_gat_hidden, d_gat_latent,
                 d_mlp_hidden, d_mlp_latent, n_nodes):
        super().__init__()
        self.struct_ae = _StructureAE(d_node, d_gat_hidden, d_gat_latent)
        self.attr_ae   = _AttributeAE(n_nodes * d_node, d_mlp_hidden, d_mlp_latent)

    def forward(self, x, adj):
        A_hat, Z1 = self.struct_ae(x, adj)
        x_flat    = x.flatten(1) if x.dim() == 2 else x.flatten(1)
        X_hat, Z2 = self.attr_ae(x_flat)
        return A_hat, X_hat, Z1, Z2


def _struct_loss(A: "torch.Tensor", A_hat: "torch.Tensor",
                 theta: float) -> "torch.Tensor":
    weight = torch.where(A > 0,
                         torch.tensor(theta, device=A.device),
                         torch.ones_like(A))
    return (weight * (A - A_hat) ** 2).mean()


def _attr_loss(X: "torch.Tensor", X_hat: "torch.Tensor",
               eta: float) -> "torch.Tensor":
    # (N,D) unbatched → (N*D,) ;  (B,N,D) batched → (B,N*D,)
    x_flat  = X.reshape(-1) if X.dim() == 2 else X.flatten(1)
    xh_flat = X_hat
    weight  = torch.where(x_flat > 0,
                           torch.tensor(eta, device=X.device),
                           torch.ones_like(x_flat))
    return (weight * (x_flat - xh_flat) ** 2).mean()


class TraceDAEDetector:
    def __init__(self, alpha: float = 0.5, theta: float = 5.0, eta: float = 40.0,
                 d_gat_h: int = 32, d_gat_z: int = 16,
                 d_mlp_h: int = 64, d_mlp_z: int = 32,
                 threshold_k: float = 3.0):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.alpha       = alpha       # weight on structure vs attribute loss
        self.theta       = theta       # edge weight in struct loss
        self.eta         = eta         # non-zero attribute weight in attr loss
        self.d_gat_h     = d_gat_h
        self.d_gat_z     = d_gat_z
        self.d_mlp_h     = d_mlp_h
        self.d_mlp_z     = d_mlp_z
        self.threshold_k = threshold_k
        self._model:     Optional[_TraceDAENet] = None
        self._mu:        Optional[np.ndarray] = None
        self._std:       Optional[np.ndarray] = None
        self._threshold: float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensors(self, records: list[TraceRecord]
                    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        Xs, As, Ls = [], [], []
        for r in records:
            X, A = _build_stg(r)
            Xs.append(X); As.append(A); Ls.append(r.label)
        if not Xs:
            return (torch.zeros(0, N_NODES, D_NODE),
                    torch.zeros(0, N_NODES, N_NODES),
                    torch.zeros(0, dtype=torch.long))
        X_np = np.stack(Xs).astype(np.float32)
        if self._mu is not None:
            flat      = X_np.reshape(len(X_np), -1)
            flat_norm = (flat - self._mu) / np.where(self._std > 0, self._std, 1.0)
            X_np      = flat_norm.reshape(X_np.shape)
        return (torch.tensor(X_np,          dtype=torch.float32),
                torch.tensor(np.stack(As),  dtype=torch.float32),
                torch.tensor(Ls,            dtype=torch.long))

    def fit(self, records: list[TraceRecord],
            epochs: int = 50, batch_size: int = 32,
            lr: float = 1e-3) -> "TraceDAEDetector":
        all_feats = np.stack([_node_features(r) for r in records]).reshape(len(records), -1).astype(np.float32)
        self._mu  = all_feats.mean(axis=0)
        self._std = all_feats.std(axis=0)

        X, A, _ = self._to_tensors(records)
        if len(X) == 0:
            raise ValueError("No training records.")

        self._model = _TraceDAENet(
            d_node=D_NODE, d_gat_hidden=self.d_gat_h, d_gat_latent=self.d_gat_z,
            d_mlp_hidden=self.d_mlp_h, d_mlp_latent=self.d_mlp_z, n_nodes=N_NODES,
        ).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)

        loader = DataLoader(TensorDataset(X, A), batch_size=batch_size, shuffle=True)
        print(f"[TraceDAE] Training on {len(X):,} windows")
        self._model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for xb, ab in loader:
                xb, ab = xb.to(self._device), ab.to(self._device)
                A_hat, X_hat, _, _ = self._model(xb, ab)
                l_s  = _struct_loss(ab, A_hat, self.theta)
                l_a  = _attr_loss(xb, X_hat, self.eta)
                loss = self.alpha * l_s + (1 - self.alpha) * l_a
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(X):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std()) \
                          if len(scores) else 0.0
        print(f"[TraceDAE] Fitted, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[TraceRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, A, L = self._to_tensors(records)
        if len(X) == 0:
            return np.array([]), np.array([])
        self._model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = X[i: i + 256].to(self._device)
                ab = A[i: i + 256].to(self._device)
                A_hat, X_hat, _, _ = self._model(xb, ab)
                l_s  = (ab   - A_hat).pow(2).flatten(1).mean(1)
                l_a  = (xb.flatten(1) - X_hat).pow(2).mean(1)
                s    = (self.alpha * l_s + (1 - self.alpha) * l_a).cpu().numpy()
                all_scores.append(s)
        return np.concatenate(all_scores), L.numpy()

    def predict(self, records: list[TraceRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "alpha": self.alpha, "theta": self.theta, "eta": self.eta,
            "d_gat_h": self.d_gat_h, "d_gat_z": self.d_gat_z,
            "d_mlp_h": self.d_mlp_h, "d_mlp_z": self.d_mlp_z,
            "threshold_k": self.threshold_k, "threshold": self._threshold,
            "mu": self._mu.tolist(), "std": self._std.tolist(),
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        print(f"[TraceDAE] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "TraceDAEDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(alpha=meta["alpha"], theta=meta["theta"], eta=meta["eta"],
                   d_gat_h=meta["d_gat_h"], d_gat_z=meta["d_gat_z"],
                   d_mlp_h=meta["d_mlp_h"], d_mlp_z=meta["d_mlp_z"],
                   threshold_k=meta["threshold_k"])
        obj._threshold = meta["threshold"]
        obj._mu  = np.array(meta["mu"],  dtype=np.float32)
        obj._std = np.array(meta["std"], dtype=np.float32)
        obj._model = _TraceDAENet(
            d_node=D_NODE, d_gat_hidden=obj.d_gat_h, d_gat_latent=obj.d_gat_z,
            d_mlp_hidden=obj.d_mlp_h, d_mlp_latent=obj.d_mlp_z, n_nodes=N_NODES)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = TraceDAEDetector(alpha=0.1, theta=5.0, eta=40.0,
                             d_gat_h=32, d_gat_z=16, d_mlp_h=64, d_mlp_z=32)
    model.fit(data["train"], epochs=50)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    print(f"[TraceDAE] P={prec:.3f}  R={rec:.3f}  "
          f"F1={2*prec*rec/max(prec+rec,1e-9):.3f}")