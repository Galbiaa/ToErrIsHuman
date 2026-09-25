"""Step 2: document scale_pos_weight per fold, without retraining.

scale_pos_weight is computed in train_D.py as
    n_pos = max(#error-rating==1 rows in outer-train, 1)
    n_neg = #error-rating==0 rows in outer-train
    scale_pos_weight = n_neg / n_pos

The outer-train decision rows for fold k are exactly the rows whose case is in
a different fold.  decision_level_features.csv (built from USER + OOF) carries
the per-case fold, so we reconstruct the counts and cross-check them against
the scale_pos_weight actually stored in each saved scenario_D_fold_*.joblib.

Writes outputs/D/scale_pos_weight_per_fold.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import load_config, project_path
from reproducibility import bootstrap_run_reproducibility, set_global_seed


def reconstruct_fold_counts(
    table: pd.DataFrame,
    *,
    fold_column: str = "fold",
    label_column: str = "error-rating",
) -> pd.DataFrame:
    """Count positive/negative decision rows in each outer-train set.

    Outer-train for fold k = all rows whose case belongs to any fold != k.
    y_all_train in train_D is exactly these rows.
    """
    folds = sorted(int(f) for f in table[fold_column].unique())
    fs = pd.DataFrame(folds, columns=["fold"])
    fs["n_total"] = 0
    fs["n_positive"] = 0
    fs["n_negative"] = 0

    for k in folds:
        train_mask = table[fold_column].astype(int) != k
        y = table.loc[train_mask, label_column].astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        fs.loc[fs["fold"] == k, ["n_total", "n_positive", "n_negative"]] = [
            int(y.size), n_pos, n_neg
        ]
    fs["scale_pos_weight"] = (fs["n_negative"] / fs["n_positive"]).fillna(np.nan)
    return fs


def check_against_saved_models(
    counts: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """Cross-check reconstructed scale_pos_weight vs saved joblib dicts."""
    rows = []
    for _, r in counts.iterrows():
        k = int(r["fold"])
        path = model_dir / f"scenario_D_fold_{k}.joblib"
        if not path.is_file():
            rows.append({"fold": k, "stored_scale_pos_weight": np.nan, "matches_stored": False})
            continue
        payload = joblib.load(path)
        stored = payload.get("scale_pos_weight")
        recon = float(r["scale_pos_weight"])
        matches = stored is not None and np.isclose(float(stored), recon, rtol=1e-12, atol=1e-12)
        rows.append(
            {
                "fold": k,
                "stored_scale_pos_weight": (float(stored) if stored is not None else np.nan),
                "matches_stored": bool(matches),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Document scale_pos_weight per fold (reconstruction + cross-check)."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    baseline_csv = (
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
    )
    if not baseline_csv.is_file():
        raise FileNotFoundError(f"Missing {baseline_csv}")
    table = pd.read_csv(baseline_csv)
    need = {"case_id", "fold", "error-rating"}
    missing = need - set(table.columns)
    if missing:
        raise ValueError(f"decision_level_features.csv missing {sorted(missing)}")

    counts = reconstruct_fold_counts(table)
    model_dir = project_path(cfg["models"]["d_dir"], base_dir)
    check = check_against_saved_models(counts, model_dir)

    report = counts.merge(check, on="fold", how="left")
    if not report["matches_stored"].all():
        bad = report.loc[~report["matches_stored"], ["fold", "scale_pos_weight", "stored_scale_pos_weight"]]
        raise RuntimeError(f"scale_pos_weight mismatch vs saved models:\n{bad.to_string(index=False)}")

    report_path = project_path(cfg["outputs"]["d_dir"], base_dir) / "scale_pos_weight_per_fold.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nscale_pos_weight per fold (weighted D):")
    print(report.to_string(index=False))
    print("\nFormula: scale_pos_weight = n_negative / n_positive")
    print("All folds match the values stored in the saved models.")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()