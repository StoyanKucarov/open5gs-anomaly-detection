#!/usr/bin/env python3
"""
visualizations/logs/_log_features.py

Shared feature extraction used by the visualisation scripts.
Not a standalone script — imported by 01_, 02_, 03_, 04_.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "experiments" / "C-fault-detection"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ALL_APPS = [
    "amf","ausf","bsf","mongodb","nrf","nssf","pcf",
    "scp","sepp","smf","udm","udr","ueransim-gnb","ueransim-ues","upf",
]


def strip_ansi(line: str) -> str:
    return ANSI.sub("", line)


def load_loki(exp_dir: Path, phase: str, fname: str) -> list[dict]:
    p = exp_dir / "loki" / phase / fname
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_timeline(exp_dir: Path) -> dict:
    p = exp_dir / "timeline.json"
    return json.loads(p.read_text()) if p.exists() else {}


def simple_template(line: str) -> str:
    """Lightweight log template: strip ANSI, then tokenise away variables."""
    line = strip_ansi(line)
    line = re.sub(r"^\d{2}/\d{2} [\d:.]+:\s*\[\w+\]\s*\w+:\s*", "", line)
    line = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", line)
    line = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", line)
    line = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", line)
    line = re.sub(r"imsi-\d+", "<IMSI>", line)
    line = re.sub(r"\b\d+\b", "<N>", line)
    line = re.sub(r"/[^\s\]]+", "<PATH>", line)
    return line.strip()


def bucket_counts(rows: list[dict], bucket_s: int = 30) -> list[int]:
    buckets: dict[int, int] = defaultdict(int)
    for r in rows:
        try:
            t = int(r["timestamp_ns"]) // 1_000_000_000
            buckets[t // bucket_s] += 1
        except (KeyError, ValueError):
            pass
    return list(buckets.values()) if buckets else [0]


def extract_features(slug: str, fault_class: str,
                     data_dir: Path | None = None) -> dict:
    exp_dir = (Path(data_dir) if data_dir else DEFAULT_DATA) / slug

    pre_rows = load_loki(exp_dir, "pre",    "all.csv")
    dur_rows = load_loki(exp_dir, "during", "all.csv")
    pre_err  = load_loki(exp_dir, "pre",    "errors.csv")
    dur_err  = load_loki(exp_dir, "during", "errors.csv")

    pre_total  = max(len(pre_rows), 1)
    dur_total  = max(len(dur_rows), 1)
    pre_errors = len(pre_err)
    dur_errors = len(dur_err)

    log_volume_ratio  = dur_total / pre_total
    error_rate_pre    = pre_errors / pre_total
    error_rate_during = dur_errors / dur_total
    error_rate_delta  = error_rate_during - error_rate_pre

    app_err_counts = Counter(strip_ansi(r.get("app", "")) for r in dur_err)
    per_app = {a: app_err_counts.get(a, 0) / max(dur_errors, 1) for a in ALL_APPS}

    pre_b     = bucket_counts(pre_err)
    dur_b     = bucket_counts(dur_err)
    med_pre   = float(np.median(pre_b)) if pre_b else 1.0
    max_dur   = max(dur_b) if dur_b else 0
    error_spike = min(max_dur / max(med_pre, 1.0), 100.0)

    pre_templates = {simple_template(r["line"]) for r in pre_rows if r.get("line")}
    dur_templates = {simple_template(r["line"]) for r in dur_rows if r.get("line")}
    template_diversity  = len(dur_templates) / max(len(pre_templates), 1)
    novel_template_rate = len(dur_templates - pre_templates) / max(len(dur_templates), 1)

    feat: dict = {
        "log_volume_ratio":    log_volume_ratio,
        "error_rate_during":   error_rate_during,
        "error_rate_delta":    error_rate_delta,
        "error_spike":         error_spike,
        "template_diversity":  template_diversity,
        "novel_template_rate": novel_template_rate,
    }
    for a in ALL_APPS:
        feat[f"app_{a}"] = per_app[a]
    return feat


def available_experiments(data_dir: Path) -> list[tuple]:
    """Return the EXPERIMENTS entries whose slug directory exists in data_dir."""
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from lib import EXPERIMENTS
    return [(s, ft, nf, fc) for s, ft, nf, fc in EXPERIMENTS
            if (data_dir / s).is_dir()]
