#!/usr/bin/env python3
"""
models/traces/data_loader.py

Loads Jaeger spans_flat.csv into TraceRecord objects for anomaly detection.

Each TraceRecord represents one 30-second window within one experiment phase.
`values` holds a 48-dim feature vector; `spans` holds the raw span dicts for
RCA-style models that need per-span access.

Feature layout (FEATURE_NAMES order):
  For each service in SERVICES (11 NFs, sorted alphabetically):
    {svc}_span_count, {svc}_error_rate, {svc}_log_mean_dur, {svc}_log_p95_dur
  Global (last 4):
    g_trace_count, g_log_mean_trace_dur, g_error_rate, g_span_count

Labelling: during-phase windows = 1, pre/post = 0.

Public API
----------
load_all(data_dir)         -> {"train": [TraceRecord, ...], "test": [TraceRecord, ...]}
load_experiment(slug, ...) -> list[TraceRecord]
"""

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from lib import EXPERIMENTS

_DEFAULT_DATA = ROOT / "data" / "experiments" / "C-fault-detection"

WINDOW_S  = 30
WINDOW_US = WINDOW_S * 1_000_000

SERVICES = sorted(["amf", "ausf", "bsf", "nrf", "nssf",
                    "pcf", "scp", "sepp", "smf", "udr", "udm"])

_PER_SVC = ["span_count", "error_rate", "log_mean_dur", "log_p95_dur"]
FEATURE_NAMES = (
    [f"{s}_{f}" for s in SERVICES for f in _PER_SVC]
    + ["g_trace_count", "g_log_mean_trace_dur", "g_error_rate", "g_span_count"]
)
N_FEATURES = len(FEATURE_NAMES)   # 11*4 + 4 = 48


@dataclass
class TraceRecord:
    slug:        str
    fault_type:  str
    fault_class: str
    phase:       Literal["pre", "during", "post"]
    window_id:   int
    window_us:   int
    label:       int
    values:      np.ndarray
    spans:       list[dict] = field(default_factory=list, repr=False)


def _load_spans(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "trace_id":    row["trace_id"],
                    "span_id":     row["span_id"],
                    "service":     row["service"],
                    "operation":   row["operation"],
                    "start_us":    int(row["start_us"]),
                    "duration_us": int(row["duration_us"]),
                    "error":       int(row["error"]),
                })
            except (KeyError, ValueError):
                pass
    return rows


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = max(0, int(np.ceil(0.95 * len(s))) - 1)
    return s[i]


def _window_features(spans: list[dict]) -> np.ndarray:
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    svc_spans: dict[str, list[dict]] = {s: [] for s in SERVICES}
    for sp in spans:
        svc = sp["service"]
        if svc in svc_spans:
            svc_spans[svc].append(sp)

    for i, svc in enumerate(SERVICES):
        sp_list = svc_spans[svc]
        base = i * 4
        if not sp_list:
            continue
        durs = [float(sp["duration_us"]) for sp in sp_list]
        feat[base + 0] = float(len(sp_list))
        feat[base + 1] = sum(sp["error"] for sp in sp_list) / len(sp_list)
        feat[base + 2] = float(np.log1p(sum(durs) / len(durs)))
        feat[base + 3] = float(np.log1p(_p95(durs)))

    base = len(SERVICES) * 4
    if spans:
        trace_durs: dict[str, int] = {}
        for sp in spans:
            tid = sp["trace_id"]
            trace_durs[tid] = trace_durs.get(tid, 0) + sp["duration_us"]
        tdurs = list(trace_durs.values())
        feat[base + 0] = float(len(tdurs))
        feat[base + 1] = float(np.log1p(sum(tdurs) / max(len(tdurs), 1)))
        feat[base + 2] = sum(sp["error"] for sp in spans) / len(spans)
        feat[base + 3] = float(len(spans))

    return feat


def _load_phase(exp_dir: Path, phase: str,
                phase_start_s: int, phase_end_s: int,
                slug: str, fault_type: str, fault_class: str
                ) -> list[TraceRecord]:
    spans_path  = exp_dir / "jaeger" / phase / "spans_flat.csv"
    all_spans   = _load_spans(spans_path)
    phase_label = 1 if phase == "during" else 0

    if not all_spans and not spans_path.exists():
        return []

    phase_start_us = phase_start_s * 1_000_000
    phase_end_us   = phase_end_s   * 1_000_000
    n_windows      = max(1, (phase_end_us - phase_start_us) // WINDOW_US)

    phase_spans = [sp for sp in all_spans
                   if phase_start_us <= sp["start_us"] < phase_end_us]

    records = []
    for w in range(n_windows):
        w_start   = phase_start_us + w * WINDOW_US
        w_end     = w_start + WINDOW_US
        win_spans = [sp for sp in phase_spans if w_start <= sp["start_us"] < w_end]
        records.append(TraceRecord(
            slug=slug, fault_type=fault_type, fault_class=fault_class,
            phase=phase, window_id=w, window_us=w_start,
            label=phase_label, values=_window_features(win_spans),
            spans=win_spans,
        ))

    return records


def load_experiment(slug: str, data_dir: Path | None = None) -> list[TraceRecord]:
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    meta     = next(((ft, fc) for s, ft, _nf, fc in EXPERIMENTS if s == slug),
                    ("unknown", "unknown"))
    fault_type, fault_class = meta
    exp_dir  = data_dir / slug
    tl_path  = exp_dir / "timeline.json"
    if not tl_path.exists():
        return []
    tl = json.loads(tl_path.read_text())

    records: list[TraceRecord] = []
    for phase, tl_key in [("pre", "pre"), ("during", "fault"), ("post", "post")]:
        records.extend(_load_phase(
            exp_dir, phase, tl[tl_key]["start"], tl[tl_key]["end"],
            slug, fault_type, fault_class))
    return records


def base_slug(slug: str) -> str:
    """Strip run tag, e.g. '01-cpu-stress-amf__r3' → '01-cpu-stress-amf'."""
    return slug.split("__r")[0]


def load_all(data_dir: Path | None = None) -> dict[str, list[TraceRecord]]:
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    train: list[TraceRecord] = []
    test:  list[TraceRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        exp_dir = data_dir / slug
        if not (exp_dir / "jaeger").is_dir():
            continue
        tl_path = exp_dir / "timeline.json"
        if not tl_path.exists():
            continue
        tl = json.loads(tl_path.read_text())

        train.extend(_load_phase(exp_dir, "pre",
                                  tl["pre"]["start"], tl["pre"]["end"],
                                  slug, ft, fc))
        test.extend(_load_phase(exp_dir, "during",
                                 tl["fault"]["start"], tl["fault"]["end"],
                                 slug, ft, fc))
        test.extend(_load_phase(exp_dir, "post",
                                 tl["post"]["start"], tl["post"]["end"],
                                 slug, ft, fc))

    return {"train": train, "test": test}


def load_multi(data_dirs: list[Path]) -> dict[str, list[TraceRecord]]:
    """
    Load from [run1, run2, run3, run4], preferring run2 over run1 per fault.

    For each fault:
      - If run2 has it  →  use run2 (primary, no tag) + run3 (__r3) + run4 (__r4)
      - If run2 missing →  use run1 (primary, no tag) + run3 (__r3) + run4 (__r4)
    """
    run1, run2, run3, run4 = (Path(d) for d in data_dirs[:4])
    train: list[TraceRecord] = []
    test:  list[TraceRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        primary = run2 if (run2 / slug).is_dir() else run1
        sources = []
        if (primary / slug / "jaeger").is_dir():
            tl = json.loads((primary / slug / "timeline.json").read_text())
            sources.append((primary / slug, "", tl))
        if run3.is_dir() and (run3 / slug / "jaeger").is_dir():
            tl = json.loads((run3 / slug / "timeline.json").read_text())
            sources.append((run3 / slug, "__r3", tl))
        if run4.is_dir() and (run4 / slug / "jaeger").is_dir():
            tl = json.loads((run4 / slug / "timeline.json").read_text())
            sources.append((run4 / slug, "__r4", tl))

        for exp_dir, tag, tl in sources:
            tagged = slug + tag
            train.extend(_load_phase(exp_dir, "pre",
                                      tl["pre"]["start"], tl["pre"]["end"],
                                      tagged, ft, fc))
            test.extend(_load_phase(exp_dir, "during",
                                     tl["fault"]["start"], tl["fault"]["end"],
                                     tagged, ft, fc))
            test.extend(_load_phase(exp_dir, "post",
                                     tl["post"]["start"], tl["post"]["end"],
                                     tagged, ft, fc))

    n_base   = len({base_slug(r.slug) for r in train})
    n_tagged = len({r.slug for r in train})
    r2_used  = sum(1 for slug, *_ in EXPERIMENTS if (run2 / slug).is_dir())
    print(f"[data_loader] Multi-run: {n_base} faults — "
          f"{r2_used} from run2+run3+run4, {22-r2_used} from run1+run3+run4 "
          f"({n_tagged} run×fault combos), "
          f"{len(train):,} train / {len(test):,} test windows")
    return {"train": train, "test": test}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    args = ap.parse_args()
    print(f"Loading from {args.data} ...")
    data  = load_all(args.data)
    train, test = data["train"], data["test"]
    n_anom = sum(r.label for r in test)
    print(f"Train: {len(train):,} windows (pre-phase, normal)")
    print(f"Test:  {len(test):,} windows  "
          f"(anomalous={n_anom:,} [{100*n_anom/max(len(test),1):.1f}%])")
    print(f"Features ({N_FEATURES}): {FEATURE_NAMES[:6]} ...")
