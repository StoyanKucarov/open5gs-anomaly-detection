#!/usr/bin/env python3
"""
models/traces/evaluate.py

Trains all trace anomaly detection models on pre-phase Jaeger data,
evaluates per fault, and writes results to disk.

Outputs (models/traces/out/)
------------------------------
  eval_results.json    — full results, readable by plot_results.py
  eval_per_fault.csv   — tabular form

Visualize with:
  python plot_results.py --results out/eval_results.json

Usage
-----
  python evaluate.py [--data PATH]

  --data PATH    experiment root (default: C-fault-detection)
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

from data_loader import load_all, load_multi, base_slug, TraceRecord, _DEFAULT_DATA
from lib import EXPERIMENTS

_DATA_ROOT = ROOT / "data" / "experiments"
_ALL_RUNS: list[Path] = [
    _DATA_ROOT / "C-fault-detection",
    _DATA_ROOT / "C-fault-detection-rerun",
    _DATA_ROOT / "C-fault-detection-4-clean" / "C-fault-detection",
    _DATA_ROOT / "C-fault-detection5",
]


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
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=prec, recall=rec, f1=f1,
                best_recall=_best_recall(scores, labels)[1],
                avg_precision=_average_precision(scores, labels),
                auroc=_auroc(scores, labels),
                tp=tp, fp=fp, fn=fn, tn=tn,
                n_windows=len(labels), n_anomalous=int(labels.sum()))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    order    = np.argsort(-scores)
    labels_s = labels[order]
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
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
    if len(scores) == 0 or labels.sum() == 0:
        return 0.0, 0.0
    thresholds = np.unique(scores)
    if len(thresholds) > 500:
        thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 500)))
    best_f1, best_rec = 0.0, 0.0
    for t in thresholds:
        p    = (scores >= t).astype(int)
        tp   = int(((p == 1) & (labels == 1)).sum())
        fp   = int(((p == 1) & (labels == 0)).sum())
        fn   = int(((p == 0) & (labels == 1)).sum())
        pr_v = tp / max(tp + fp, 1)
        rc_v = tp / max(tp + fn, 1)
        f1_v = 2 * pr_v * rc_v / max(pr_v + rc_v, 1e-9)
        if f1_v > best_f1:
            best_f1, best_rec = f1_v, rc_v
    return best_f1, best_rec


def run_models(train: list[TraceRecord]) -> dict:
    models: dict = {}

    from rpca_model import TraceRPCADetector
    print("\n== TraceRPCA ==")
    t0 = time.time()
    m  = TraceRPCADetector(n_components=10)
    m.fit(train)
    models["TraceRPCA"] = (m, time.time() - t0)

    from trace_anomaly_model import TraceAnomalyDetector
    print("\n== TraceAnomaly ==")
    t0 = time.time()
    m  = TraceAnomalyDetector(n_layers=6, hidden_dim=128)
    m.fit(train, epochs=80)
    models["TraceAnomaly"] = (m, time.time() - t0)

    from galmad_model import GALMADDetector
    print("\n== GAL-MAD ==")
    t0 = time.time()
    m  = GALMADDetector(seq_len=8, d_gat=16, d_lstm=32)
    m.fit(train, epochs=60)
    models["GAL-MAD"] = (m, time.time() - t0)

    from tracedae_model import TraceDAEDetector
    print("\n== TraceDAE ==")
    t0 = time.time()
    m  = TraceDAEDetector(alpha=0.1, theta=5.0, eta=40.0,
                          d_gat_h=32, d_gat_z=16,
                          d_mlp_h=64, d_mlp_z=32)
    m.fit(train, epochs=50)
    models["TraceDAE"] = (m, time.time() - t0)

    from tracesieve_model import TraceSieveDetector
    print("\n== TraceSieve ==")
    t0 = time.time()
    m  = TraceSieveDetector(hidden_dim=64, latent_dim=16, gan_epochs=50,
                            vgae_epochs=100)
    m.fit(train)
    models["TraceSieve"] = (m, time.time() - t0)

    return models


def evaluate_per_fault(models: dict, test: list[TraceRecord]) -> list[dict]:
    slug_to_meta = {s: (ft, fc) for s, ft, _nf, fc in EXPERIMENTS}
    results: list[dict] = []

    for slug in sorted({base_slug(r.slug) for r in test}):
        fault_recs = [r for r in test if base_slug(r.slug) == slug]
        ft, fc     = slug_to_meta.get(slug, ("unknown", "unknown"))
        n_anom     = sum(r.label for r in fault_recs)
        print(f"  {slug}: {len(fault_recs):,} windows, {n_anom:,} anomalous")

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
    path.write_text(json.dumps(
        {"meta": meta, "model_names": model_names, "results": results},
        indent=2))
    print(f"Saved -> {path}")


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved -> {path}")


# robustness perturbations (applied to test set only)
_TRACE_DROPOUT_GROUPS = {
    "span_count": lambda n: n.endswith("_span_count"),
    "error_rate": lambda n: n.endswith("_error_rate"),
    "latency":    lambda n: n.endswith("_log_mean_dur") or n.endswith("_log_p95_dur"),
    "global":     lambda n: n.startswith("g_"),
}


def _apply_dropout(records: list, group: str) -> None:
    from data_loader import FEATURE_NAMES
    if group not in _TRACE_DROPOUT_GROUPS:
        raise ValueError(f"--dropout must be one of: {sorted(_TRACE_DROPOUT_GROUPS)}")
    pred = _TRACE_DROPOUT_GROUPS[group]
    idxs = [i for i, n in enumerate(FEATURE_NAMES) if pred(n)]
    if not idxs:
        print(f"[dropout] WARNING: no features matched group '{group}'")
        return
    print(f"[dropout] Zeroing {len(idxs)} '{group}' dims on {len(records):,} test windows.")
    for r in records:
        r.values[idxs] = 0.0


def _apply_noise(records: list, std: float) -> None:
    rng = np.random.default_rng(42)
    for r in records:
        r.values = r.values + rng.standard_normal(r.values.shape).astype(np.float32) * std
    print(f"[noise] Added N(0, {std:.3f}) to {len(records):,} test windows.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--out",       type=str, default=None,
                    help="Override output JSON path.")
    ap.add_argument("--multi-run", action="store_true",
                    help="Combine all four standard run directories.")
    ap.add_argument("--dropout", type=str, default=None, metavar="GROUP",
                    help="Zero out a trace feature group on the test set: "
                         "span_count|error_rate|latency|global  (model trains on clean data)")
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Std of Gaussian noise added to test feature vectors "
                         "(0 = no noise).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.multi_run:
        print("Loading traces from all runs ...")
        for p in _ALL_RUNS:
            print(f"  {'OK' if p.is_dir() else '--'} {p}")
        data = load_multi(_ALL_RUNS)
    else:
        print(f"Loading traces from {args.data} ...")
        data = load_all(args.data)
    train, test = data["train"], data["test"]

    if args.dropout:
        _apply_dropout(test, args.dropout)
    if args.noise_std > 0:
        _apply_noise(test, args.noise_std)

    n_anom      = sum(r.label for r in test)
    print(f"  Train: {len(train):,} windows (pre-phase, all normal)")
    print(f"  Test:  {len(test):,} windows  "
          f"({n_anom:,} anomalous [{100*n_anom/max(len(test),1):.1f}%])")

    print("\n=== Training models ===")
    models      = run_models(train)
    model_names = list(models.keys())

    print("\n=== Per-fault evaluation ===")
    results = evaluate_per_fault(models, test)

    meta = {
        "data":        str(args.data),
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "n_train":     len(train),
        "n_test":      len(test),
        "n_anomalous": n_anom,
        "modality":    "traces",
        "dropout":     args.dropout,
        "noise_std":   args.noise_std,
    }
    if args.out:
        out_json = Path(args.out)
    else:
        stem = "eval_results"
        if args.dropout:
            stem += f"_dropout_{args.dropout}"
        if args.noise_std > 0:
            stem += f"_noise_{str(args.noise_std).replace('.', 'p')}"
        out_json = OUT / f"{stem}.json"
    out_csv = out_json.with_suffix(".csv")
    write_json(results, model_names, meta, out_json)
    write_csv(results, out_csv)

    print(f"\nResults saved to {OUT}/")
    print("Visualize: python plot_results.py --results out/eval_results.json")


if __name__ == "__main__":
    main()
