#!/usr/bin/env python3
"""
analysis/signal_correlation.py

Two analyses on the fault atlas:

A) Signal co-occurrence matrix (Jaccard similarity):
   High score = signals always fire together (redundant)
   Low score  = signals fire independently (complementary)

B) Signal discriminability:
   Which signals are most specific to one fault class?
   Metric: information entropy of fault_class distribution when signal fires.
   Low entropy = signal is specific to few fault classes (high discriminability).

Requires fault_atlas.csv to exist (run fault_atlas.py first).

Outputs:
  <out>/signal_correlation.csv   — NxN Jaccard matrix
  <out>/discriminability.csv     — per-signal specificity scores
  stdout                         — discriminability ranking

Usage:
    python3 analysis/signal_correlation.py \
        [--out reproduce/data/analysis]
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from lib import ALL_SIGNALS, SIGNAL_LAYERS, entropy


def load_atlas(atlas_path: Path) -> list:
    with open(atlas_path, newline="") as f:
        return list(csv.DictReader(f))


def jaccard(a: list, b: list) -> float:
    """Jaccard similarity between two binary lists."""
    assert len(a) == len(b)
    intersection = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(
        Path(__file__).parent.parent / "data/analysis"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    atlas_path = out_dir / "fault_atlas.csv"
    if not atlas_path.exists():
        sys.exit(f"fault_atlas.csv not found at {atlas_path} — run fault_atlas.py first")

    rows = load_atlas(atlas_path)
    signals = [s for s in ALL_SIGNALS if s in rows[0]]

    # Binary vectors per signal
    vectors = {s: [int(r[s]) for r in rows] for s in signals}
    fault_classes = [r["fault_class"] for r in rows]

    # --- A) Jaccard co-occurrence matrix ---
    corr_path = out_dir / "signal_correlation.csv"
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"] + signals)
        for s1 in signals:
            row = [s1]
            for s2 in signals:
                row.append(round(jaccard(vectors[s1], vectors[s2]), 3))
            w.writerow(row)
    print(f"[out] {corr_path}")

    # Print top correlated pairs
    pairs = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i+1:]:
            j = jaccard(vectors[s1], vectors[s2])
            pairs.append((j, s1, s2))
    pairs.sort(reverse=True)

    print("\nTop 10 most correlated signal pairs (Jaccard):")
    for j, s1, s2 in pairs[:10]:
        print(f"  {j:.3f}  {s1:<25} ↔  {s2}")

    print("\nBottom 10 most independent signal pairs (Jaccard):")
    # filter to pairs where both fire at least once
    bottom = [(j, s1, s2) for j, s1, s2 in pairs
              if any(vectors[s1]) and any(vectors[s2])]
    for j, s1, s2 in bottom[-10:]:
        print(f"  {j:.3f}  {s1:<25} ↔  {s2}")

    # --- B) Signal discriminability ---
    disc_rows = []
    for s in signals:
        fired_indices = [i for i, v in enumerate(vectors[s]) if v]
        n_fired = len(fired_indices)
        if n_fired == 0:
            disc_rows.append({
                "signal": s,
                "layer": SIGNAL_LAYERS.get(s, "unknown"),
                "fires_in_n_faults": 0,
                "fault_classes_when_fired": "",
                "class_counts": "{}",
                "entropy": None,
                "specificity_score": None,
            })
            continue

        class_counts: dict = defaultdict(int)
        for i in fired_indices:
            class_counts[fault_classes[i]] += 1

        h = entropy(dict(class_counts))
        # Specificity: 1 - normalized entropy (0 = fires for all classes equally, 1 = fires for one class only)
        max_h = math.log2(len(set(fault_classes))) if len(set(fault_classes)) > 1 else 1
        specificity = round(1 - h / max_h, 3) if max_h > 0 else 1.0

        disc_rows.append({
            "signal": s,
            "layer": SIGNAL_LAYERS.get(s, "unknown"),
            "fires_in_n_faults": n_fired,
            "fault_classes_when_fired": ", ".join(sorted(set(fault_classes[i] for i in fired_indices))),
            "class_counts": str(dict(sorted(class_counts.items()))),
            "entropy": round(h, 3),
            "specificity_score": specificity,
        })

    # Sort by specificity descending
    disc_rows.sort(key=lambda r: (r["specificity_score"] is None, -(r["specificity_score"] or 0)))

    disc_path = out_dir / "discriminability.csv"
    with open(disc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(disc_rows[0].keys()))
        w.writeheader()
        w.writerows(disc_rows)
    print(f"[out] {disc_path}")

    print(f"\n{'Signal':<25} {'Layer':<16} {'Fires':>5}  {'Specificity':>11}  Fault classes")
    print("-" * 100)
    for r in disc_rows:
        if r["fires_in_n_faults"] == 0:
            continue
        spec = f"{r['specificity_score']:.3f}" if r['specificity_score'] is not None else "  n/a"
        print(f"  {r['signal']:<23} {r['layer']:<16} {r['fires_in_n_faults']:>5}  {spec:>11}  {r['fault_classes_when_fired']}")


if __name__ == "__main__":
    main()
