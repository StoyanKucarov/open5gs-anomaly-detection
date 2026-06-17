#!/usr/bin/env python3
"""
Run all log visualisation scripts in order, forwarding --data and --out.

Usage:
  python run_all.py [--data PATH] [--out PATH]

Examples:
  python run_all.py
  python run_all.py --data ../../data/experiments/C-fault-detection-rerun
  python run_all.py --data ../../data/experiments/C-fault-detection-rerun --out out/rerun
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_log_feature_heatmap.py",
    "02_fault_clustering.py",
    "03_error_template_distribution.py",
    "04_anomaly_timeline.py",
]

HERE = Path(__file__).parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Experiment run directory (passed through to each script)")
    ap.add_argument("--out",  type=str, default=None,
                    help="Output directory (passed through to each script)")
    return ap.parse_args()


def main():
    args = parse_args()
    extra = []
    if args.data:
        extra += ["--data", args.data]
    if args.out:
        extra += ["--out", args.out]

    for s in SCRIPTS:
        print(f"\n{'='*60}\n{s}\n{'='*60}")
        cmd = [sys.executable, str(HERE / s)] + extra
        r   = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(f"[FAILED] {s}")


if __name__ == "__main__":
    main()
