#!/usr/bin/env python3
"""
models/logs/data_loader.py

Unified log data loader for all 4 anomaly detection models.

Labelling strategy
------------------
Labels are assigned at the phase level:

  during-phase lines  -> label = 1  (anomalous)
  pre / post lines    -> label = 0  (normal)

Window label = 1 if ANY line in the window has label=1, else 0.

Public functions
----------------
load_experiment(slug, data_dir)
    -> list of LogRecord for one experiment (all phases).

load_all(data_dir)
    -> {"train": [LogRecord, ...], "test": [LogRecord, ...]}

    Training set  = all pre-phase logs from all experiments (label=0 only).
    Test set      = during-phase + post-phase from all experiments
                    (during error lines = 1, everything else = 0).

load_sequences(records, window, step)
    -> list of (sequence_of_template_ids, label)

LogRecord fields
----------------
  slug          experiment slug
  fault_class   resource_exhaustion | component_failure | ...
  phase         pre | during | post
  timestamp_ns  int (nanoseconds since epoch)
  app           NF short name (amf, smf, upf, ...)
  pod           k8s pod name
  line          raw log line (ANSI stripped)
  template_id   int assigned by the shared LogParser
  template      template string
  label         0 = normal, 1 = anomalous (error line during fault window)
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from lib import EXPERIMENTS

from log_parser import LogParser, strip_ansi

_DEFAULT_DATA = ROOT / "data" / "experiments" / "C-fault-detection"

# Shared parser — global so template IDs are consistent across all experiments
# regardless of which data directory is used.
_PARSER = LogParser(depth=4, similarity_threshold=0.5, max_children=128)


@dataclass
class LogRecord:
    slug: str
    fault_type: str
    fault_class: str
    phase: Literal["pre", "during", "post"]
    timestamp_ns: int
    app: str
    pod: str
    line: str
    template_id: int
    template: str
    label: int  # 0 = normal, 1 = anomalous (error line in fault window)


def _error_keys(exp_dir: Path, phase: str) -> set[tuple[str, str]]:
    """
    Return a set of (timestamp_ns_str, pod) for every line in errors.csv.
    Used to label individual lines in all.csv without re-parsing line content.
    """
    path = exp_dir / "loki" / phase / "errors.csv"
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("timestamp_ns", ""), row.get("pod", "")))
    return keys


def _load_phase(exp_dir: Path, phase: str, slug: str,
                fault_type: str, fault_class: str) -> list[LogRecord]:
    path = exp_dir / "loki" / phase / "all.csv"
    if not path.exists():
        return []

    phase_label = 1 if phase == "during" else 0

    records: list[LogRecord] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            raw   = row.get("line", "")
            clean = strip_ansi(raw).strip()
            if not clean:
                continue
            tid, tmpl = _PARSER.parse(clean)
            try:
                ts = int(row.get("timestamp_ns", 0))
            except ValueError:
                ts = 0

            records.append(LogRecord(
                slug=slug,
                fault_type=fault_type,
                fault_class=fault_class,
                phase=phase,
                timestamp_ns=ts,
                app=strip_ansi(row.get("app", "")),
                pod=strip_ansi(row.get("pod", "")),
                line=clean,
                template_id=tid,
                template=tmpl,
                label=phase_label,
            ))
    return records


def load_experiment(slug: str,
                    data_dir: Path | None = None) -> list[LogRecord]:
    """Load all three phases for one experiment slug."""
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    meta = next(
        ((ft, fc) for s, ft, _nf, fc in EXPERIMENTS if s == slug),
        ("unknown", "unknown"),
    )
    fault_type, fault_class = meta
    exp_dir = data_dir / slug
    records: list[LogRecord] = []
    records.extend(_load_phase(exp_dir, "pre",    slug, fault_type, fault_class))
    records.extend(_load_phase(exp_dir, "during", slug, fault_type, fault_class))
    records.extend(_load_phase(exp_dir, "post",   slug, fault_type, fault_class))
    records.sort(key=lambda r: r.timestamp_ns)
    return records


def load_all(data_dir: Path | None = None) -> dict[str, list[LogRecord]]:
    """
    Returns {"train": [...], "test": [...]}.

    Train = pre-phase from all available experiments (all label=0).
    Test  = during + post from all available experiments
            (during error lines = label=1, everything else = 0).

    Only slugs that exist in data_dir are included, so this works with
    both C-fault-detection (22 faults) and C-fault-detection-rerun (14).

    After all experiments are parsed, deduplicates template IDs: Drain's
    greedy sequential parsing can assign different IDs to identical template
    strings depending on parse order.  Records are updated in-place so the
    LSTM embedding table is shared across experiments correctly.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    train: list[LogRecord] = []
    test:  list[LogRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        exp_dir = data_dir / slug
        if not exp_dir.is_dir():
            continue
        train.extend(_load_phase(exp_dir, "pre",    slug, ft, fc))
        test.extend( _load_phase(exp_dir, "during", slug, ft, fc))
        test.extend( _load_phase(exp_dir, "post",   slug, ft, fc))

    # Merge template IDs that have identical strings but different IDs due to
    # Drain's parse-order sensitivity.  Must happen after all experiments are
    # loaded so every template string has been seen.
    remap = _PARSER.dedup_templates()
    if remap:
        n_changed = 0
        for r in train:
            new = remap.get(r.template_id)
            if new is not None:
                r.template_id = new
                n_changed += 1
        for r in test:
            new = remap.get(r.template_id)
            if new is not None:
                r.template_id = new
                n_changed += 1
        print(f"[data_loader] Template dedup: {len(remap):,} duplicate IDs merged, "
              f"{n_changed:,} records remapped")

    train.sort(key=lambda r: r.timestamp_ns)
    test.sort( key=lambda r: r.timestamp_ns)
    return {"train": train, "test": test}


def base_slug(slug: str) -> str:
    """Strip run tag from a slug, e.g. '01-cpu-stress-amf__r3' → '01-cpu-stress-amf'."""
    return slug.split("__r")[0]


def load_multi(data_dirs: list[Path]) -> dict[str, list[LogRecord]]:
    """
    Load from [run1, run2, run3, run4], preferring run2 over run1 per fault.

    For each fault:
      - If run2 has it  →  use run2 (primary, no tag) + run3 (__r3) + run4 (__r4)
      - If run2 missing →  use run1 (primary, no tag) + run3 (__r3) + run4 (__r4)

    Run1 is the original (potentially incorrect) data; run2 is the rerun that
    supersedes it wherever available; run3 and run4 are clean runs that always apply.

    All directories are parsed through the shared Drain instance before
    deduplication so template IDs are globally stable.
    """
    run1, run2, run3, run4 = (Path(d) for d in data_dirs[:4])
    train: list[LogRecord] = []
    test:  list[LogRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        primary = run2 if (run2 / slug).is_dir() else run1
        sources = []
        if (primary / slug).is_dir():
            sources.append((primary / slug, ""))      # primary: no tag
        if run3.is_dir() and (run3 / slug).is_dir():
            sources.append((run3 / slug, "__r3"))
        if run4.is_dir() and (run4 / slug).is_dir():
            sources.append((run4 / slug, "__r4"))

        for exp_dir, tag in sources:
            tagged = slug + tag
            train.extend(_load_phase(exp_dir, "pre",    tagged, ft, fc))
            test.extend( _load_phase(exp_dir, "during", tagged, ft, fc))
            test.extend( _load_phase(exp_dir, "post",   tagged, ft, fc))

    remap = _PARSER.dedup_templates()
    if remap:
        n_changed = 0
        for r in (*train, *test):
            new = remap.get(r.template_id)
            if new is not None:
                r.template_id = new
                n_changed += 1
        if n_changed:
            print(f"[data_loader] Template dedup: {len(remap):,} IDs merged, "
                  f"{n_changed:,} records remapped")

    train.sort(key=lambda r: r.timestamp_ns)
    test.sort( key=lambda r: r.timestamp_ns)

    n_base   = len({base_slug(r.slug) for r in train})
    n_tagged = len({r.slug for r in train})
    r2_used  = sum(1 for slug, *_ in EXPERIMENTS if (run2 / slug).is_dir())
    r1_used  = 22 - r2_used
    print(f"[data_loader] Multi-run: {n_base} faults — "
          f"{r2_used} from run2+run3+run4, {r1_used} from run1+run3+run4 "
          f"({n_tagged} run×fault combos), "
          f"{len(train):,} train / {len(test):,} test records")
    return {"train": train, "test": test}


def load_sequences(records: list[LogRecord],
                   window: int = 10,
                   step:   int = 1,
                   group_by_slug: bool = True,
                   ) -> list[tuple[list[int], int]]:
    """
    Sliding-window sequence extraction.

    Returns list of (template_id_sequence, label).
    Label = 1 if ANY record in the window is anomalous (label=1), else 0.
    group_by_slug=True restarts windows at experiment boundaries.
    """
    if group_by_slug:
        groups: dict[str, list[LogRecord]] = {}
        for r in records:
            groups.setdefault(r.slug, []).append(r)
        sequences: list[tuple[list[int], int]] = []
        for recs in groups.values():
            sequences.extend(_sliding_windows(recs, window, step))
        return sequences
    return _sliding_windows(records, window, step)


def _sliding_windows(records: list[LogRecord],
                     window: int, step: int,
                     ) -> list[tuple[list[int], int]]:
    out = []
    for i in range(0, len(records) - window + 1, step):
        w = records[i: i + window]
        ids   = [r.template_id for r in w]
        label = int(any(r.label == 1 for r in w))
        out.append((ids, label))
    return out


def get_parser() -> LogParser:
    return _PARSER


def vocab_size() -> int:
    return _PARSER._next_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    args = ap.parse_args()

    print(f"Loading from {args.data} ...")
    data = load_all(args.data)
    train, test = data["train"], data["test"]
    n_anom = sum(r.label for r in test)
    n_norm = sum(1 - r.label for r in test)
    print(f"Train records : {len(train):,}  (all label=0)")
    print(f"Test  records : {len(test):,}  "
          f"(anomalous={n_anom:,} [{100*n_anom/max(len(test),1):.1f}%], "
          f"normal={n_norm:,})")
    print(f"Template vocab: {vocab_size()} templates")
    seqs = load_sequences(train + test, window=10, step=1)
    n_anom_w = sum(l for _, l in seqs)
    print(f"Sequences (w=10, s=1): {len(seqs):,}  "
          f"anomalous windows={n_anom_w:,} [{100*n_anom_w/max(len(seqs),1):.1f}%]")
