#!/usr/bin/env python3
"""
Isolation Forest on time-bucketed log features for 5G Core anomaly detection.
Features: log rate, error rate, novel template fraction, heartbeat deficit,
and per-NF log/error rates. Heartbeat templates identified by low inter-arrival CV.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LogRecord

ALL_NFS = [
    "amf", "ausf", "bsf", "mongodb", "nrf", "nssf", "pcf",
    "scp", "sepp", "smf", "udm", "udr", "ueransim-gnb", "ueransim-ues", "upf",
]

_ERROR_RE = re.compile(r"\b(ERROR|WARNING|error|warning|WARN|ERRO)\b")


def _is_error(line: str) -> bool:
    return bool(_ERROR_RE.search(line))


class FeatureModelDetector:
    def __init__(self,
                 bucket_size: int = 30,
                 heartbeat_cv_threshold: float = 0.5,
                 heartbeat_min_rate: float = 0.05,
                 n_estimators: int = 100,
                 contamination: float = 0.05):
        self.bucket_size   = bucket_size
        self.hb_cv_th      = heartbeat_cv_threshold
        self.hb_min_rate   = heartbeat_min_rate
        self.n_estimators  = n_estimators
        self.contamination = contamination

        self._known_templates: set[int] = set()
        self._heartbeat_templates: dict[int, float] = {}  # tid -> expected events/bucket
        self._feature_names: list[str] = []
        self._model = None
        self._threshold: float = 0.0

    def _bucket_records(self, records: list[LogRecord]
                        ) -> dict[int, list[LogRecord]]:
        buckets: dict[int, list[LogRecord]] = defaultdict(list)
        for r in records:
            if r.timestamp_ns <= 0:
                continue
            bid = int(r.timestamp_ns // 1_000_000_000) // self.bucket_size
            buckets[bid].append(r)
        return buckets

    def _extract_features(self, bucket_recs: list[LogRecord]) -> np.ndarray:
        n  = max(len(bucket_recs), 1)
        dt = float(self.bucket_size)

        total_log_rate = len(bucket_recs) / dt

        error_recs = [r for r in bucket_recs if _is_error(r.line)]
        error_rate  = len(error_recs) / n
        error_count = float(len(error_recs))

        tids = {r.template_id for r in bucket_recs}
        novel_rate = len(tids - self._known_templates) / max(len(tids), 1)

        if self._heartbeat_templates:
            tid_counts = Counter(r.template_id for r in bucket_recs)
            deficit = expected = 0.0
            for tid, exp in self._heartbeat_templates.items():
                actual   = float(tid_counts.get(tid, 0))
                deficit  += max(exp - actual, 0.0)
                expected += exp
            heartbeat_deficit = deficit / max(expected, 1.0)
        else:
            heartbeat_deficit = 0.0

        nf_counts = Counter(r.app for r in bucket_recs)
        nf_errors = Counter(r.app for r in error_recs)
        nf_feats: list[float] = []
        for nf in ALL_NFS:
            cnt = nf_counts.get(nf, 0)
            err = nf_errors.get(nf, 0)
            nf_feats.append(cnt / dt)
            nf_feats.append(err / max(cnt, 1))

        return np.array(
            [total_log_rate, error_rate, error_count, novel_rate, heartbeat_deficit]
            + nf_feats,
            dtype=np.float32,
        )

    def _build_feature_matrix(self, records: list[LogRecord]
                              ) -> tuple[np.ndarray, np.ndarray]:
        buckets = self._bucket_records(records)
        n_feat  = 5 + 2 * len(ALL_NFS)
        if not buckets:
            return np.zeros((0, n_feat), dtype=np.float32), np.zeros(0, dtype=int)

        rows, labels = [], []
        for bid in sorted(buckets):
            recs  = buckets[bid]
            feat  = self._extract_features(recs)
            label = int(any(r.label == 1 for r in recs))
            rows.append(feat)
            labels.append(label)

        return np.array(rows, dtype=np.float32), np.array(labels, dtype=int)

    def _identify_heartbeat_templates(self, records: list[LogRecord]) -> None:
        """
        Tag templates as periodic if their intra-experiment inter-arrival time
        has low CV and meets a minimum rate. Inter-arrival times are computed
        per (slug, tid) pair to avoid cross-experiment gaps inflating variance.
        """
        by_slug_tid: dict[tuple[str, int], list[float]] = defaultdict(list)
        slug_ts: dict[str, list[float]] = defaultdict(list)
        for r in records:
            if r.timestamp_ns > 0:
                ts = r.timestamp_ns / 1e9
                by_slug_tid[(r.slug, r.template_id)].append(ts)
                slug_ts[r.slug].append(ts)

        slug_durations = {
            slug: max(ts) - min(ts)
            for slug, ts in slug_ts.items()
            if len(ts) >= 2
        }
        total_duration = max(sum(slug_durations.values()), 1.0)

        tid_gaps: dict[int, list[float]] = defaultdict(list)
        tid_count: dict[int, int] = defaultdict(int)
        for (slug, tid), times in by_slug_tid.items():
            times_s = sorted(times)
            tid_count[tid] += len(times_s)
            if len(times_s) >= 2:
                tid_gaps[tid].extend(np.diff(times_s).tolist())

        hb: dict[int, float] = {}
        for tid, gaps in tid_gaps.items():
            rate = tid_count[tid] / total_duration
            if rate < self.hb_min_rate or len(gaps) < 4:
                continue

            gaps_arr = np.array(gaps, dtype=np.float64)

            # Drop outlier gaps (e.g. restarts within one slug) using IQR filter.
            q75, q25 = np.percentile(gaps_arr, [75, 25])
            iqr = q75 - q25
            gaps_arr = gaps_arr[gaps_arr <= q75 + 3.0 * iqr]
            if len(gaps_arr) < 4:
                continue

            mean_gap = float(gaps_arr.mean())
            if mean_gap <= 0:
                continue
            cv = float(gaps_arr.std() / mean_gap)
            if cv <= self.hb_cv_th:
                hb[tid] = rate * self.bucket_size  # expected events per bucket

        self._heartbeat_templates = hb
        print(f"[FeatureModel] Heartbeat templates identified: {len(hb)}")

    def fit(self, records: list[LogRecord]) -> "FeatureModelDetector":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise RuntimeError("scikit-learn required: pip install scikit-learn")

        self._known_templates = {r.template_id for r in records}
        self._identify_heartbeat_templates(records)
        self._feature_names = (
            ["total_log_rate", "error_rate", "error_count",
             "novel_rate", "heartbeat_deficit"]
            + [f"nf_{nf}_log_rate"   for nf in ALL_NFS]
            + [f"nf_{nf}_error_rate" for nf in ALL_NFS]
        )

        X, _ = self._build_feature_matrix(records)
        if len(X) == 0:
            raise ValueError("No time buckets built from training records.")

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        self._model.fit(X)

        raw = -self._model.score_samples(X)
        self._threshold = float(raw.mean() + 3.0 * raw.std())
        print(f"[FeatureModel] Fitted: {len(X)} buckets, "
              f"{len(self._known_templates)} known templates, "
              f"threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[LogRecord]) -> tuple[np.ndarray, np.ndarray]:
        X, labels = self._build_feature_matrix(records)
        if len(X) == 0:
            return np.array([]), np.array([])
        return -self._model.score_samples(X), labels

    def predict(self, records: list[LogRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        preds = (scores > self._threshold).astype(int)
        return preds, scores, labels

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "bucket_size":         self.bucket_size,
            "hb_cv_th":            self.hb_cv_th,
            "hb_min_rate":         self.hb_min_rate,
            "n_estimators":        self.n_estimators,
            "contamination":       self.contamination,
            "threshold":           self._threshold,
            "known_templates":     list(self._known_templates),
            "heartbeat_templates": {str(k): v for k, v in self._heartbeat_templates.items()},
            "feature_names":       self._feature_names,
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        with open(str(path) + "_model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        print(f"[FeatureModel] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "FeatureModelDetector":
        import pickle
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj = cls(
            bucket_size=meta["bucket_size"],
            heartbeat_cv_threshold=meta["hb_cv_th"],
            heartbeat_min_rate=meta["hb_min_rate"],
            n_estimators=meta["n_estimators"],
            contamination=meta["contamination"],
        )
        obj._threshold           = meta["threshold"]
        obj._known_templates     = set(meta["known_templates"])
        obj._heartbeat_templates = {int(k): v for k, v in meta["heartbeat_templates"].items()}
        obj._feature_names       = meta["feature_names"]
        with open(str(path) + "_model.pkl", "rb") as f:
            obj._model = pickle.load(f)
        return obj


if __name__ == "__main__":
    from data_loader import load_all

    data  = load_all()
    model = FeatureModelDetector(bucket_size=30)
    model.fit(data["train"])

    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n[FeatureModel] Test results — time buckets")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
