#!/usr/bin/env python3
"""
visualizations/robustness/run_sweep.py

Runner for Approach 2 (feature dropout) and Approach 3 (noise injection).

Calls each modality's evaluate.py with the relevant flags and saves results
to the modality's out/ directory.  Only runs what is missing — safe to re-run.

Usage
-----
  # Run everything (noise sweep + dropout sweep)
  python visualizations/robustness/run_sweep.py

  # Noise sweep only
  python visualizations/robustness/run_sweep.py --noise-only

  # Dropout sweep only
  python visualizations/robustness/run_sweep.py --dropout-only

  # Skip slow LSTM models in logs modality (much faster, still covers FeatureModel + LogCluster)
  python visualizations/robustness/run_sweep.py --skip-lstm

  # Noise sweep for logs only
  python visualizations/robustness/run_sweep.py --noise-only --modality logs

  # Dry run (print commands without executing)
  python visualizations/robustness/run_sweep.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
MODELS  = ROOT / "models"

# Noise levels to sweep (σ for metrics/traces; fraction for logs)
NOISE_LEVELS = [0.05, 0.10, 0.25, 0.50, 1.00]

DROPOUT_GROUPS = {
    "metrics": ["http", "cpu", "memory", "network", "5g_control"],
    "traces":  ["span_count", "error_rate", "latency", "global"],
}


def run(cmd: list[str], dry: bool) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if not dry:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[WARN] Command exited with code {result.returncode}")


def noise_path(modality: str, level: float) -> Path:
    stem = f"eval_results_noise_{str(level).replace('.', 'p')}.json"
    return MODELS / modality / "out" / stem


def dropout_path(modality: str, group: str) -> Path:
    return MODELS / modality / "out" / f"eval_results_dropout_{group}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-only",   action="store_true")
    ap.add_argument("--dropout-only", action="store_true")
    ap.add_argument("--modality",     choices=["logs", "metrics", "traces"],
                    help="Restrict sweep to a single modality")
    ap.add_argument("--skip-lstm",    action="store_true",
                    help="Pass --skip-lstm to logs evaluate.py (faster)")
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--force",        action="store_true",
                    help="Re-run even if output file already exists")
    args = ap.parse_args()

    py = sys.executable
    run_noise   = not args.dropout_only
    run_dropout = not args.noise_only

    if run_noise:
        print("\n" + "="*60)
        print("APPROACH 3 — Noise injection sweep")
        print("="*60)

        only = args.modality

        # Logs (template ID substitution noise)
        if only in (None, "logs"):
            eval_logs = MODELS / "logs" / "evaluate.py"
            for level in NOISE_LEVELS:
                out = noise_path("logs", level)
                if out.exists() and not args.force:
                    print(f"  [skip] {out.name} already exists")
                    continue
                cmd = [py, str(eval_logs), "--noise-frac", str(level)]
                if args.skip_lstm:
                    cmd.append("--skip-lstm")
                run(cmd, args.dry_run)

        # Metrics (Gaussian feature noise)
        if only in (None, "metrics"):
            eval_metrics = MODELS / "metrics" / "evaluate.py"
            for level in NOISE_LEVELS:
                out = noise_path("metrics", level)
                if out.exists() and not args.force:
                    print(f"  [skip] {out.name} already exists")
                    continue
                run([py, str(eval_metrics), "--noise-std", str(level)], args.dry_run)

        # Traces (Gaussian feature noise)
        if only in (None, "traces"):
            eval_traces = MODELS / "traces" / "evaluate.py"
            for level in NOISE_LEVELS:
                out = noise_path("traces", level)
                if out.exists() and not args.force:
                    print(f"  [skip] {out.name} already exists")
                    continue
                run([py, str(eval_traces), "--noise-std", str(level)], args.dry_run)

    if run_dropout:
        print("\n" + "="*60)
        print("APPROACH 2 — Feature group dropout")
        print("="*60)

        only = args.modality

        if only in (None, "metrics"):
            eval_metrics = MODELS / "metrics" / "evaluate.py"
            for group in DROPOUT_GROUPS["metrics"]:
                out = dropout_path("metrics", group)
                if out.exists() and not args.force:
                    print(f"  [skip] {out.name} already exists")
                    continue
                run([py, str(eval_metrics), "--dropout", group], args.dry_run)

        if only in (None, "traces"):
            eval_traces = MODELS / "traces" / "evaluate.py"
            for group in DROPOUT_GROUPS["traces"]:
                out = dropout_path("traces", group)
                if out.exists() and not args.force:
                    print(f"  [skip] {out.name} already exists")
                    continue
                run([py, str(eval_traces), "--dropout", group], args.dry_run)

    print("\n\nSweep complete.  Generate figures:")
    print("  python visualizations/robustness/02_noise_robustness.py")
    print("  python visualizations/robustness/03_feature_dropout.py")


if __name__ == "__main__":
    main()
