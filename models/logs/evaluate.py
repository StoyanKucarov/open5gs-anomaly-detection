#!/usr/bin/env python3
"""
models/logs/evaluate.py

Trains all log anomaly detection models on the combined pre-phase data from
all available experiments, evaluates each model per fault, and writes results
to disk.  No plotting — run plot_results.py to generate visualizations.

Outputs (models/logs/out/)
--------------------------
  eval_results.json    — full results (metrics + metadata), input for plot_results.py
  eval_per_fault.csv   — same data in tabular form

Usage
-----
  python evaluate.py [--data PATH] [--epochs N] [--skip-lstm]

  --data PATH    experiment run directory (default: C-fault-detection)
  --epochs N     LSTM training epochs (default 20)
  --skip-lstm    run Logs2Graphs + FeatureModel only (no LSTM)
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

from data_loader import load_all, load_multi, base_slug, LogRecord, _DEFAULT_DATA
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
        n_pos = int(labels.sum()) if len(labels) > 0 else 0
        prevalence = n_pos / max(len(labels), 1)
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


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Area under the precision-recall curve (Average Precision).

    Threshold-free and cannot be gamed by 'flag everything': a model with no
    ranking ability has AP ≈ prevalence (fraction of positives), not 1.0.
    A perfect model has AP = 1.0.
    """
    if len(scores) == 0 or labels.sum() == 0:
        return float(labels.sum() / max(len(labels), 1))
    order      = np.argsort(-scores)
    labs_sorted = labels[order]
    n_pos      = int(labels.sum())
    tp_cum     = np.cumsum(labs_sorted)
    prec       = tp_cum / np.arange(1, len(labs_sorted) + 1)
    rec        = tp_cum / n_pos
    # Prepend (recall=0, precision=1) sentinel so the first step is counted.
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


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    order    = np.argsort(-scores)
    labels_s = labels[order]
    n_pos    = labels.sum()
    n_neg    = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(labels_s) / n_pos
    fpr = np.cumsum(1 - labels_s) / n_neg
    return float(np.trapz(tpr, fpr))


def run_models(train: list[LogRecord], epochs: int, skip_lstm: bool) -> dict:
    """Train all models. Returns {name: (fitted_model, train_time_s)}."""
    models = {}

    from logbert_model import LogBERTDetector
    print("\n== LogBERT ==")
    t0 = time.time()
    m = LogBERTDetector(window=10, step=1, top_k=9,
                        d_model=64, n_heads=4, n_layers=2)
    m.fit(train, epochs=epochs, batch_size=512)
    models["LogBERT"] = (m, time.time() - t0)

    from logs2graphs_model import Logs2GraphsDetector
    print("\n== Logs2Graphs ==")
    t0 = time.time()
    m = Logs2GraphsDetector(window_ns=30_000_000_000,
                            d_emb=32, d_hidden=64)
    m.fit(train, epochs=30)
    models["Logs2Graphs"] = (m, time.time() - t0)

    from feature_model import FeatureModelDetector
    print("\n== FeatureModel ==")
    t0 = time.time()
    m = FeatureModelDetector(bucket_size=10, heartbeat_cv_threshold=0.5,
                             heartbeat_min_rate=0.05, n_estimators=100)
    m.fit(train)
    models["FeatureModel"] = (m, time.time() - t0)

    if not skip_lstm:
        from deeplog_model import DeepLogDetector
        print("\n== DeepLog ==")
        t0 = time.time()
        m = DeepLogDetector(window=10, step=1, top_k=9,
                            embed_dim=32, hidden_dim=64, n_layers=2)
        m.fit(train, epochs=epochs, batch_size=512)
        models["DeepLog"] = (m, time.time() - t0)

        from logrobust_model import LogRobustDetector
        print("\n== LogRobust ==")
        t0 = time.time()
        m = LogRobustDetector(window=10, step=1,
                              word_embed_dim=32, lstm_hidden=64, n_layers=2)
        m.fit(train, epochs=epochs, batch_size=256)
        models["LogRobust"] = (m, time.time() - t0)

    return models


def evaluate_per_fault(models: dict, test: list[LogRecord]) -> list[dict]:
    """
    Evaluate every model per fault, averaging AUROC/AP across runs.

    When a fault has records from multiple runs (e.g. primary + __r3 + __r4),
    each run is scored independently and AUROC/AP are averaged across runs.
    Precision/recall/F1 are computed over the pooled predictions (the threshold
    is shared, so pooling is the correct aggregation for those metrics).
    Single-run evaluation is unchanged.
    """
    slug_to_meta = {s: (ft, fc) for s, ft, _nf, fc in EXPERIMENTS}
    results = []

    for base in sorted({base_slug(r.slug) for r in test}):
        run_slugs  = sorted({r.slug for r in test if base_slug(r.slug) == base})
        all_recs   = [r for r in test if base_slug(r.slug) == base]
        ft, fc     = slug_to_meta.get(base, ("unknown", "unknown"))
        n_anom     = sum(r.label for r in all_recs)
        print(f"  {base}: {len(all_recs):,} records across "
              f"{len(run_slugs)} run(s), {n_anom} anomalous lines")

        for model_name, (model, train_time) in models.items():
            if len(run_slugs) > 1:
                # Score each run independently; average the threshold-free metrics.
                run_aurocs, run_aps = [], []
                pool_p, pool_s, pool_l = [], [], []
                for rs in run_slugs:
                    recs = [r for r in test if r.slug == rs]
                    p, s, l = model.predict(recs)
                    if len(p):
                        pool_p.append(p); pool_s.append(s); pool_l.append(l)
                    if len(s) and l.sum() > 0 and (l == 0).sum() > 0:
                        run_aurocs.append(_auroc(s, l))
                        run_aps.append(_average_precision(s, l))

                all_p = np.concatenate(pool_p) if pool_p else np.array([])
                all_s = np.concatenate(pool_s) if pool_s else np.array([])
                all_l = np.concatenate(pool_l) if pool_l else np.array([])
                m = compute_metrics(all_p, all_s, all_l)
                if run_aurocs:
                    m["auroc"]         = round(float(np.mean(run_aurocs)), 6)
                    m["avg_precision"] = round(float(np.mean(run_aps)), 6)
                m["auroc_std"] = round(float(np.std(run_aurocs)), 6) if len(run_aurocs) > 1 else 0.0
                m["n_runs"]    = len(run_aurocs)
            else:
                p, s, l = model.predict(all_recs)
                m = compute_metrics(p, s, l)
                m["auroc_std"] = 0.0
                m["n_runs"]    = 1

            results.append({
                "slug":        base,
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


def _apply_log_noise(records: list, frac: float) -> None:
    """Replace `frac` fraction of test template IDs with random in-vocabulary IDs."""
    vocab = list({r.template_id for r in records})
    if len(vocab) < 2:
        return
    rng = np.random.default_rng(42)
    n_replaced = 0
    for r in records:
        if rng.random() < frac:
            r.template_id = int(rng.choice(vocab))
            n_replaced += 1
    print(f"[noise] Replaced {n_replaced:,}/{len(records):,} "
          f"({100*n_replaced/max(len(records),1):.1f}%) template IDs "
          f"with random in-vocab IDs (frac={frac}).")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--epochs",     type=int,  default=20)
    ap.add_argument("--skip-lstm",  action="store_true")
    ap.add_argument("--out",        type=str, default=None,
                    help="Override output JSON path (CSV path derived automatically).")
    ap.add_argument("--multi-run",  action="store_true",
                    help="Combine all four standard run directories "
                         "(C-fault-detection, -rerun, -4-clean, 5). "
                         "More training data per fault; evaluation reports "
                         "per base-fault AUROC across all runs.")
    ap.add_argument("--noise-frac", type=float, default=0.0,
                    help="Fraction of test log template IDs to replace with random "
                         "in-vocabulary IDs (0 = no noise, 1 = full random). "
                         "Simulates noisy or missing log lines.")
    ap.add_argument(
        "--split", type=int, default=None, metavar="N",
        help=(
            "Held-out experiment split: train on the pre-phase of the first N "
            "experiments, test on during+post of the remaining ones. "
            "Outputs go to eval_results_heldout.json / eval_per_fault_heldout.csv. "
            "All experiments are parsed first so template IDs are stable. "
            "Example: --split 11  (train on 01–11, test on 12–22)"
        ),
    )
    return ap.parse_args()


def _apply_split(all_data: dict, split_n: int,
                 data_dir: Path) -> tuple[list, list, list[str], list[str]]:
    """
    Partition pre-loaded records into held-out train/test sets.

    Returns (train_records, test_records, train_slugs, test_slugs).
    All records have already been parsed through the shared Drain instance,
    so template IDs are globally consistent.
    """
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


def main():
    args = parse_args()

    if args.multi_run:
        print(f"Loading data from all runs ...")
        for p in _ALL_RUNS:
            print(f"  {'OK' if p.is_dir() else '--'} {p}")
        all_data = load_multi(_ALL_RUNS)
    else:
        print(f"Loading data from {args.data} ...")
        all_data = load_all(args.data)

    if args.split is not None:
        train, test, train_slugs, test_slugs = _apply_split(
            all_data, args.split, args.data
        )
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

    if args.noise_frac > 0:
        _apply_log_noise(test, args.noise_frac)

    n_anom = sum(r.label for r in test)
    print(f"  Train: {len(train):,} records (all normal)")
    print(f"  Test:  {len(test):,} records  "
          f"({n_anom:,} anomalous [{100*n_anom/max(len(test),1):.1f}%])")

    print("\n=== Training models ===")
    models = run_models(train, args.epochs, args.skip_lstm)
    model_names = list(models.keys())

    print("\n=== Per-fault evaluation ===")
    results = evaluate_per_fault(models, test)

    meta = {
        "data":        str(args.data),
        "epochs":      args.epochs,
        "skip_lstm":   args.skip_lstm,
        "split":       args.split,
        "train_slugs": list(train_slugs),
        "test_slugs":  list(test_slugs),
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "n_train":     len(train),
        "n_test":      len(test),
        "n_anomalous_lines": n_anom,
        "noise_frac":  args.noise_frac,
    }
    if args.out:
        out_json = Path(args.out)
    elif args.noise_frac > 0:
        stem = out_json.stem + f"_noise_{str(args.noise_frac).replace('.', 'p')}"
        out_json = out_json.with_name(stem + ".json")
    out_csv = out_json.with_suffix(".csv")
    write_json(results, model_names, meta, out_json)
    write_csv(results, out_csv)

    print(f"\nResults saved to {OUT}/")
    print("Run  python plot_results.py  to generate visualizations.")


if __name__ == "__main__":
    main()
