"""
analysis/run_all_analysis.py

Master runner — executes all analysis modules in order and prints timing.

Usage:
    cd analysis
    pip install -r requirements.txt
    python run_all_analysis.py

Outputs land in:
    analysis/figures/   (*.pdf)
    analysis/tables/    (*.csv, *.txt)
"""

import time
import traceback
from pathlib import Path

import rq1a_overhead_comparison
import rq1b_granularity
import rq1c_scalability
import fault_detection_coverage
import observability_gap

MODULES = [
    ("RQ1.1a — Overhead comparison",       rq1a_overhead_comparison),
    ("RQ1.1b — Granularity / cost-value",  rq1b_granularity),
    ("RQ1.1c — Scalability",               rq1c_scalability),
    ("RQ1.2  — Fault detection coverage",  fault_detection_coverage),
    ("RQ1.3  — Observability gap",         observability_gap),
]


def main() -> None:
    from config import FIGURES_DIR, TABLES_DIR  # type: ignore[import]
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    results: list[tuple[str, float, str]] = []

    for label, module in MODULES:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        status = "OK"
        try:
            module.run()
        except Exception:
            status = "FAILED"
            traceback.print_exc()
        elapsed = time.perf_counter() - t0
        results.append((label, elapsed, status))
        print(f"  → {status}  ({elapsed:.1f}s)")

    total = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for label, elapsed, status in results:
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {label:<40}  {elapsed:5.1f}s  {status}")
    print(f"\n  Total: {total:.1f}s")
    failed = [l for l, _, s in results if s != "OK"]
    if failed:
        print(f"\n  FAILED modules ({len(failed)}):")
        for l in failed:
            print(f"    - {l}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
