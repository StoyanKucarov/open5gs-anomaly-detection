#!/usr/bin/env python3
"""
Robust PCA trace anomaly detection (Candès et al., JACM 2011).
Inexact ALM decomposition M = L + S; score = reconstruction error vs. normal subspace.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import TraceRecord

THRESHOLD_K = 3.0


def _svt(M: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding: returns U diag(max(sig-tau,0)) Vt."""
    U, sig, Vt = np.linalg.svd(M, full_matrices=False)
    return (U * np.maximum(sig - tau, 0.0)) @ Vt


def _rpca_ialm(M: np.ndarray, lam: float | None = None,
               tol: float = 1e-6, max_iter: int = 500
               ) -> tuple[np.ndarray, np.ndarray]:
    """Inexact ALM RPCA (Lin et al. 2010).  Returns (L, S) where M = L + S."""
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_fro = np.linalg.norm(M, "fro")
    norm_2   = np.linalg.norm(M, 2)

    mu     = 1.25 / max(norm_2, 1e-10)
    mu_bar = mu * 1e7
    rho    = 1.5

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)

    for _ in range(max_iter):
        L = _svt(M - S + Y / mu, 1.0 / mu)

        T = M - L + Y / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0.0)

        res = M - L - S
        Y   = Y + mu * res
        mu  = min(rho * mu, mu_bar)

        if np.linalg.norm(res, "fro") / max(norm_fro, 1e-10) < tol:
            break

    return L, S


class TraceRPCADetector:
    def __init__(self, n_components: int = 10, threshold_k: float = THRESHOLD_K):
        self.n_components = n_components
        self.threshold_k  = threshold_k
        self._mean:       Optional[np.ndarray] = None
        self._std:        Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._threshold:  float = float("inf")

    def _to_matrix(self, records: list[TraceRecord]
                   ) -> tuple[np.ndarray, np.ndarray]:
        X      = np.stack([r.values for r in records]).astype(np.float64)
        labels = np.array([r.label  for r in records], dtype=int)
        return X, labels

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / np.where(self._std > 0, self._std, 1.0)

    def _recon_error(self, X_norm: np.ndarray) -> np.ndarray:
        V    = self._components
        proj = X_norm @ V.T @ V
        return ((X_norm - proj) ** 2).sum(axis=1)

    def fit(self, records: list[TraceRecord]) -> "TraceRPCADetector":
        X, _ = self._to_matrix(records)
        if len(X) == 0:
            raise ValueError("No training records.")

        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        X_norm     = self._normalise(X)

        print(f"[TraceRPCA] Running RPCA on {X_norm.shape} matrix ...")
        L, _ = _rpca_ialm(X_norm)

        n_comp = max(1, min(self.n_components, min(L.shape) - 1))
        _, _, Vt = np.linalg.svd(L, full_matrices=False)
        self._components = Vt[:n_comp]

        scores          = self._recon_error(X_norm)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std())
        print(f"[TraceRPCA] Fitted: {len(X):,} windows, "
              f"{n_comp} components, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[TraceRecord]
              ) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, labels = self._to_matrix(records)
        return self._recon_error(self._normalise(X)), labels

    def predict(self, records: list[TraceRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    m     = TraceRPCADetector(n_components=10)
    m.fit(data["train"])
    preds, scores, labels = m.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    print(f"[TraceRPCA] P={prec:.3f}  R={rec:.3f}  "
          f"F1={2*prec*rec/max(prec+rec,1e-9):.3f}")
