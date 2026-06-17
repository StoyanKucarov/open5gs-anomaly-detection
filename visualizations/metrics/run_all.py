#!/usr/bin/env python3
"""
Run all metrics visualisation scripts in order.

Usage:
  python run_all.py [--data PATH] [--out PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_metric_delta_heatmap.py",
    "02_metric_timelines.py",
    "03_fault_class_profiles.py",
]

HERE = Path(__file__).parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out",  type=str, default=None)
    return ap.parse_args()


def main():
    args  = parse_args()
    extra = []
    if args.data:
        extra += ["--data", args.data]
    if args.out:
        extra += ["--out", args.out]

    for s in SCRIPTS:
        print(f"\n{'='*60}\n{s}\n{'='*60}")
        r = subprocess.run([sys.executable, str(HERE / s)] + extra, check=False)
        if r.returncode != 0:
            print(f"[FAILED] {s}")


if __name__ == "__main__":
    main()
