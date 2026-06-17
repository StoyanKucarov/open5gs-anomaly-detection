#!/usr/bin/env python3
"""
PCA-based metric anomaly detection; z-score normalised reconstruction error.
Alves et al. 2026 confirm PCA matches OmniAnomaly under fair evaluation protocols.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import MetricRecord, FEATURE_NAMES, N_FEATURES

THRESHOLD_K = 3.0


class MetricPCADetector:
    def __init__(self, n_components: int = 10, threshold_k: float = THRESHOLD_K):
        self.n_components  = n_components
        self.threshold_k   = threshold_k
        self._mean:       Optional[np.ndarray] = None
        self._std:        Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._threshold:  float = float("inf")

    def _to_matrix(self, records: list[MetricRecord]) -> tuple[np.ndarray, np.ndarray]:
        X      = np.stack([r.values for r in records]).astype(np.float32)
        labels = np.array([r.label for r in records], dtype=int)
        return X, labels

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / np.where(self._std > 0, self._std, 1.0)

    def _reconstruction_error(self, X_norm: np.ndarray) -> np.ndarray:
        proj     = X_norm @ self._components.T @ self._components
        residual = X_norm - proj
        return (residual ** 2).sum(axis=1)

    def fit(self, records: list[MetricRecord]) -> "MetricPCADetector":
        X, _ = self._to_matrix(records)
        if len(X) == 0:
            raise ValueError("No training records.")

        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        X_norm     = self._normalise(X)

        n_comp = min(self.n_components, X_norm.shape[0] - 1, X_norm.shape[1])
        n_comp = max(n_comp, 1)
        _, _, Vt = np.linalg.svd(X_norm, full_matrices=False)
        self._components = Vt[:n_comp]

        scores = self._reconstruction_error(X_norm)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std())
        print(f"[MetricPCA] Fitted: {len(X):,} records, "
              f"{n_comp} components, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[MetricRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X, labels = self._to_matrix(records)
        scores    = self._reconstruction_error(self._normalise(X))
        return scores, labels

    def predict(self, records: list[MetricRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        preds = (scores > self._threshold).astype(int)
        return preds, scores, labels

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 mean=self._mean, std=self._std,
                 components=self._components,
                 threshold=np.array([self._threshold]),
                 n_components=np.array([self.n_components]),
                 threshold_k=np.array([self.threshold_k]))
        print(f"[MetricPCA] Saved -> {path}.npz")

    @classmethod
    def load(cls, path: Path) -> "MetricPCADetector":
        d   = np.load(str(path) + ".npz", allow_pickle=False)
        obj = cls(n_components=int(d["n_components"][0]),
                  threshold_k=float(d["threshold_k"][0]))
        obj._mean       = d["mean"]
        obj._std        = d["std"]
        obj._components = d["components"]
        obj._threshold  = float(d["threshold"][0])
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = MetricPCADetector(n_components=10)
    model.fit(data["train"])
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[MetricPCA] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
