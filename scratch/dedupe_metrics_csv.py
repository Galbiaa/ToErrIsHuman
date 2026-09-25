"""One-off cleanup: remove duplicated rows left by append-based metric CSVs.

Keeps the LAST occurrence per key (most recent run). Read-only on the CSVs
listed; prints before/after counts.
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "outputs"

JOBS = [
    {
        "path": BASE / "baseline" / "baseline_q_ir_metrics.csv",
        "keys": ["split", "model"],
    },
    {
        "path": BASE / "compare" / "weighted" / "metrics_point.csv",
        "keys": ["model", "threshold_note"],
    },
    {
        "path": BASE / "compare" / "unweighted" / "metrics_point.csv",
        "keys": ["model", "threshold_note"],
    },
]

for job in JOBS:
    p = job["path"]
    if not p.is_file():
        print(f"SKIP (missing): {p}")
        continue
    df = pd.read_csv(p)
    before = len(df)
    key_cols = [c for c in job["keys"] if c in df.columns]
    df["__n"] = range(len(df))  # preserve order, keep last
    ded = df.drop_duplicates(subset=key_cols, keep="last").drop(columns="__n")
    ded = ded.sort_values(by=key_cols if key_cols else df.columns[:1])
    ded.to_csv(p, index=False)
    print(f"{p.name}: {before} -> {len(ded)} rows (kept last per {key_cols})")