#!/usr/bin/env python3
"""
models/metrics/evaluate.py

Trains all metrics anomaly detection models on pre-phase Prometheus data,
evaluates per fault, and writes results to disk.

Outputs (models/metrics/out/)
------------------------------
  eval_results.json    — full results, readable by models/logs/plot_results.py
  eval_per_fault.csv   — tabular form

Visualize with:
  python ../logs/plot_results.py --results out/eval_results.json

Usage
-----
  python evaluate.py [--data PATH]

  --data PATH    experiment run directory (default: C-fault-detection)
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "analysis"))

from data_loader import load_all, load_multi, base_slug, MetricRecord, _DEFAULT_DATA
from lib import EXPERIMENTS

_DATA_ROOT = ROOT / "data" / "experiments"
_ALL_RUNS: list[Path] = [
    _DATA_ROOT / "C-fault-detection",
    _DATA_ROOT / "C-fault-detection-rerun",
    _DATA_ROOT / "C-fault-detection-4-clean" / "C-fault-detection",
    _DATA_ROOT / "C-fault-detection5",
]


# identical to logs/evaluate.py so plot_results.py works unchanged
def compute_metrics(preds: np.ndarray, scores: np.ndarray,
                    labels: np.ndarray) -> dict:
    if len(labels) == 0 or labels.sum() == 0:
        prevalence = int(labels.sum()) / max(len(labels), 1)
        return dict(precision=0.0, recall=0.0, f1=0.0,
                    best_recall=0.0, avg_precision=prevalence, auroc=0.5,
                    tp=0, fp=0, fn=0, tn=int((labels == 0).sum()),
                    n_windows=len(labels), n_anomalous=0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-9)
    auroc = _auroc(scores, labels)
    ap    = _average_precision(scores, labels)
    _, best_rec = _best_recall(scores, labels)
    return dict(precision=prec, recall=rec, f1=f1,
                best_recall=best_rec, avg_precision=ap, auroc=auroc,
                tp=tp, fp=fp, fn=fn, tn=tn,
                n_windows=len(labels), n_anomalous=int(labels.sum()))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    order    = np.argsort(-scores)
    labels_s = labels[order]
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(labels_s) / n_pos
    fpr = np.cumsum(1 - labels_s) / n_neg
    return float(np.trapz(tpr, fpr))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0 or labels.sum() == 0:
        return float(labels.sum() / max(len(labels), 1))
    order       = np.argsort(-scores)
    labs_sorted = labels[order]
    n_pos       = int(labels.sum())
    tp_cum      = np.cumsum(labs_sorted)
    prec        = tp_cum / np.arange(1, len(labs_sorted) + 1)
    rec         = tp_cum / n_pos
    rec  = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def _best_recall(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Recall at the threshold that maximises F1 (optimal operating point)."""
    if len(scores) == 0 or labels.sum() == 0:
        return 0.0, 0.0
    thresholds = np.unique(scores)
    if len(thresholds) > 500:
        thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 500)))
    best_f1, best_rec = 0.0, 0.0
    for t in thresholds:
        p  = (scores >= t).astype(int)
        tp = int(((p == 1) & (labels == 1)).sum())
        fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_rec = f1, rec
    return best_f1, best_rec


def run_models(train: list[MetricRecord]) -> dict:
    """Train all unsupervised models on the combined pre-phase corpus."""
    models = {}

    from pca_model import MetricPCADetector
    print("\n== MetricPCA ==")
    t0 = time.time()
    m  = MetricPCADetector(n_components=10)
    m.fit(train)
    models["MetricPCA"] = (m, time.time() - t0)

    from usad_model import USADDetector
    print("\n== USAD ==")
    t0 = time.time()
    m  = USADDetector(window=5)
    m.fit(train, epochs=50)
    models["USAD"] = (m, time.time() - t0)

    from tranad_model import TranADDetector
    print("\n== TranAD ==")
    t0 = time.time()
    m  = TranADDetector(window=10)
    m.fit(train, epochs=20)
    models["TranAD"] = (m, time.time() - t0)

    from omnianomaly_model import OmniAnomalyDetector
    print("\n== OmniAnomaly ==")
    t0 = time.time()
    m  = OmniAnomalyDetector(window=20, latent_dim=8, hidden_dim=64, n_layers=2)
    m.fit(train, epochs=20)
    models["OmniAnomaly"] = (m, time.time() - t0)

    from anomaly_transformer_model import AnomalyTransformerDetector
    print("\n== AnomalyTransformer ==")
    t0 = time.time()
    m  = AnomalyTransformerDetector(window=20, d_model=64, n_heads=4, n_layers=3)
    m.fit(train, epochs=30)
    models["AnomalyTransformer"] = (m, time.time() - t0)

    return models


def evaluate_per_fault(models: dict, test: list[MetricRecord]) -> list[dict]:
    slug_to_meta = {s: (ft, fc) for s, ft, _nf, fc in EXPERIMENTS}
    results      = []

    for slug in sorted({base_slug(r.slug) for r in test}):
        fault_recs = [r for r in test if base_slug(r.slug) == slug]
        ft, fc     = slug_to_meta.get(slug, ("unknown", "unknown"))
        n_anom     = sum(r.label for r in fault_recs)
        print(f"  {slug}: {len(fault_recs):,} records, {n_anom:,} anomalous")

        for model_name, (model, train_time) in models.items():
            preds, scores, labels = model.predict(fault_recs)
            m = compute_metrics(preds, scores, labels)
            results.append({
                "slug":        slug,
                "fault_type":  ft,
                "fault_class": fc,
                "model":       model_name,
                "train_time":  round(train_time, 3),
                **m,
            })

    return results


def write_json(results: list[dict], model_names: list[str],
               meta: dict, path: Path) -> None:
    payload = {"meta": meta, "model_names": model_names, "results": results}
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {path}")


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved -> {path}")


def _apply_split(all_data: dict, split_n: int,
                 data_dir: Path) -> tuple[list, list, list[str], list[str]]:
    available = [s for s, _, _, _ in EXPERIMENTS
                 if (data_dir / s).is_dir()]
    if split_n < 1 or split_n >= len(available):
        raise ValueError(
            f"--split must be between 1 and {len(available) - 1}, got {split_n}"
        )
    train_slugs = set(available[:split_n])
    test_slugs  = set(available[split_n:])
    train = [r for r in all_data["train"] if r.slug in train_slugs]
    test  = [r for r in all_data["test"]  if r.slug in test_slugs]
    return train, test, sorted(train_slugs), sorted(test_slugs)


# robustness perturbations (applied to test set only — model trains on clean data)
_METRIC_DROPOUT_GROUPS = {
    "http":       lambda n: "http" in n,
    "cpu":        lambda n: n.startswith("cpu_"),
    "memory":     lambda n: n.startswith("mem_"),
    "network":    lambda n: n.startswith("net_"),
    "5g_control": lambda n: any(n.startswith(p)
                                for p in ("amf_", "pfcp_", "smf_", "upf_", "gtp_")),
}


def _apply_dropout(records: list, group: str) -> None:
    from data_loader import FEATURE_NAMES
    if group not in _METRIC_DROPOUT_GROUPS:
        raise ValueError(f"--dropout must be one of: {sorted(_METRIC_DROPOUT_GROUPS)}")
    pred = _METRIC_DROPOUT_GROUPS[group]
    idxs = [i for i, n in enumerate(FEATURE_NAMES) if pred(n)]
    if not idxs:
        print(f"[dropout] WARNING: no features matched group '{group}'")
        return
    print(f"[dropout] Zeroing {len(idxs)} '{group}' features on {len(records):,} test records.")
    for r in records:
        r.values[idxs] = 0.0


def _apply_noise(records: list, std: float) -> None:
    rng = np.random.default_rng(42)
    for r in records:
        r.values = r.values + rng.standard_normal(r.values.shape).astype(np.float32) * std
    print(f"[noise] Added N(0, {std:.3f}) to {len(records):,} test feature vectors.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--out",       type=str, default=None,
                    help="Override output JSON path.")
    ap.add_argument("--multi-run", action="store_true",
                    help="Combine all four standard run directories.")
    ap.add_argument("--dropout", type=str, default=None, metavar="GROUP",
                    help="Zero out a metric feature group on the test set: "
                         "http|cpu|memory|network|5g_control  (model trains on clean data)")
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Std of Gaussian noise added to test feature vectors "
                         "(applied before each model scores; 0 = no noise).")
    ap.add_argument(
        "--split", type=int, default=None, metavar="N",
        help=(
            "Held-out experiment split: train on pre-phase of the first N "
            "experiments, test on during+post of the remaining ones. "
            "Example: --split 11  (train on 01–11, test on 12–22)"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.multi_run:
        print("Loading metrics from all runs ...")
        for p in _ALL_RUNS:
            print(f"  {'OK' if p.is_dir() else '--'} {p}")
        all_data = load_multi(_ALL_RUNS)
    else:
        print(f"Loading metrics from {args.data} ...")
        all_data = load_all(args.data)

    if args.split is not None:
        train, test, train_slugs, test_slugs = _apply_split(
            all_data, args.split, args.data)
        print(f"\n  Held-out split: train on first {args.split} experiments, "
              f"test on remaining {len(test_slugs)}")
        print(f"  Train experiments ({len(train_slugs)}): "
              + ", ".join(s.split("-")[0] for s in train_slugs))
        print(f"  Test  experiments ({len(test_slugs)}):  "
              + ", ".join(s.split("-")[0] for s in test_slugs))
        out_json = OUT / "eval_results_heldout.json"
        out_csv  = OUT / "eval_per_fault_heldout.csv"
    else:
        train, test = all_data["train"], all_data["test"]
        train_slugs, test_slugs = [], []
        out_json = OUT / "eval_results.json"
        out_csv  = OUT / "eval_per_fault.csv"

    if args.dropout:
        _apply_dropout(test, args.dropout)
    if args.noise_std > 0:
        _apply_noise(test, args.noise_std)

    n_anom = sum(r.label for r in test)
    print(f"  Train: {len(train):,} records (all normal)")
    print(f"  Test:  {len(test):,} records  "
          f"({n_anom:,} anomalous [{100*n_anom/max(len(test),1):.1f}%])")

    print("\n=== Training models ===")
    models = run_models(train)
    model_names = list(models.keys())

    print("\n=== Per-fault evaluation ===")
    results = evaluate_per_fault(models, test)

    meta = {
        "data":        str(args.data),
        "split":       args.split,
        "train_slugs": list(train_slugs),
        "test_slugs":  list(test_slugs),
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "n_train":     len(train),
        "n_test":      len(test),
        "n_anomalous": n_anom,
        "modality":    "metrics",
        "dropout":     args.dropout,
        "noise_std":   args.noise_std,
    }
    if args.out:
        out_json = Path(args.out)
    elif args.dropout or args.noise_std > 0:
        stem = out_json.stem
        if args.dropout:
            stem += f"_dropout_{args.dropout}"
        if args.noise_std > 0:
            stem += f"_noise_{str(args.noise_std).replace('.', 'p')}"
        out_json = out_json.with_name(stem + ".json")
    out_csv = out_json.with_suffix(".csv")
    write_json(results, model_names, meta, out_json)
    write_csv(results, out_csv)

    print(f"\nResults saved to {OUT}/")
    print("Visualize: python ../logs/plot_results.py --results out/eval_results.json")


if __name__ == "__main__":
    main()
