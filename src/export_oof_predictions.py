"""Export clean OOF prediction files for the report.

Request (richieste-1.txt #4): row-level OOF predictions for q_ir and D with at
least case_id, rater_id, y_true, predicted_probability, fold.

Files produced:
  outputs/baseline/oof_q_ir_predictions.csv            (from decision_level_features.csv)
  outputs/D/oof_D_predictions.csv                      (weighted, from oof_predictions.csv)
  outputs/D/oof_D_predictions_unweighted.csv           (unweighted, if present)

The same functions are imported by build_baseline.py and train_D.py so future
runs regenerate these files automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import load_config, project_path
from reproducibility import bootstrap_run_reproducibility, set_global_seed

EXPORT_COLS = ["id", "case_id", "rater_id", "y_true", "predicted_probability", "fold"]


def _validate_export(df: pd.DataFrame, *, source: str) -> None:
    if df["id"].duplicated().any():
        raise ValueError(f"{source}: duplicate id rows in OOF export")
    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{source}: y_true must be binary 0/1")
    if not np.isfinite(df["predicted_probability"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: non-finite predicted_probability")


def _prepare_export(df: pd.DataFrame, *, prob_col: str, source: str) -> pd.DataFrame:
    required = {"id", "case_id", "rater_id", "fold", "error-rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if prob_col not in df.columns:
        raise ValueError(f"{source}: missing probability column {prob_col!r}")

    out = pd.DataFrame(
        {
            "id": df["id"],
            "case_id": df["case_id"],
            "rater_id": df["rater_id"],
            "y_true": df["error-rating"],
            "predicted_probability": df[prob_col],
            "fold": df["fold"],
        }
    ).sort_values(["case_id", "rater_id"]).reset_index(drop=True)
    for c in ("case_id", "rater_id", "fold"):
        out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["predicted_probability"] = pd.to_numeric(
        out["predicted_probability"], errors="raise"
    ).astype(float)
    _validate_export(out, source=source)
    return out[EXPORT_COLS]


def export_q_ir_oof(baseline_decision_csv: str | Path, out_csv: str | Path) -> Path:
    """q_ir OOF from the decision-level baseline table."""
    df = pd.read_csv(baseline_decision_csv)
    out = _prepare_export(df, prob_col="q_ir", source=str(baseline_decision_csv))
    out.to_csv(out_csv, index=False)
    return Path(out_csv)


def export_d_oof(d_oof_csv: str | Path, out_csv: str | Path) -> Path:
    """D OOF (weighted or unweighted) with predicted probability = d_probability."""
    df = pd.read_csv(d_oof_csv)
    out = _prepare_export(df, prob_col="d_probability", source=str(d_oof_csv))
    out.to_csv(out_csv, index=False)
    return Path(out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export clean OOF prediction files (q_ir, D weighted, D unweighted)."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    baseline_dir = project_path(cfg["outputs"]["baseline_dir"], base_dir)
    d_dir = project_path(cfg["outputs"]["d_dir"], base_dir)

    base_csv = baseline_dir / "decision_level_features.csv"
    if not base_csv.is_file():
        raise FileNotFoundError(f"Missing {base_csv}")
    q_out = export_q_ir_oof(base_csv, baseline_dir / "oof_q_ir_predictions.csv")
    print(f"q_ir OOF -> {q_out}")

    for name, src_name, out_name in (
        ("weighted", "oof_predictions.csv", "oof_D_predictions.csv"),
        ("unweighted", "oof_predictions_unweighted.csv", "oof_D_predictions_unweighted.csv"),
    ):
        src = d_dir / src_name
        if not src.is_file():
            print(f"SKIP {name}: {src} not found")
            continue
        dst = export_d_oof(src, d_dir / out_name)
        rows = pd.read_csv(dst)
        print(f"D {name} OOF -> {dst} ({len(rows)} rows)")


if __name__ == "__main__":
    main()