"""Build the human-error baseline from diagnostic OOF probabilities.

For each case probability p_i and each rater rating R:
  q_ir = p_i if R == 0 else (1 - p_i)   # P(Y != R | images)

The baseline score for error-rating is q_ir (same as ai_probability_rater_wrong).
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

from data import load_config, load_user_infos, project_path
from export_oof_predictions import export_q_ir_oof
from metrics import (
    CALIBRATION_INTERPRETATION_NOTE,
    append_metrics_row,
    binary_metrics_with_calibration,
    reset_metrics_file,
    save_calibration_plot,
    save_pr_curve,
    save_roc_curve,
)
from methodology_guards import tag_probability_source
from reproducibility import bootstrap_run_reproducibility, set_global_seed

AI_FEATURE_COLS = [
    "ai_probability_class_1",
    "ai_predicted_class",
    "rater_ai_disagreement",
    "ai_probability_rater_wrong",
    "ai_margin",
]


def compute_ai_decision_features(
    p_i: np.ndarray | pd.Series,
    rating: np.ndarray | pd.Series,
    *,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Five AI-derived decision features from case-level p_i and rater rating R.

    q_ir = P(Y != R | images) = p_i if R=0 else 1-p_i.
    ai_margin = |p_i - 0.5|.
    """
    p = np.asarray(p_i, dtype=float)
    r = np.asarray(rating, dtype=int)
    if p.shape != r.shape:
        raise ValueError(f"p_i and rating length mismatch: {p.shape} vs {r.shape}")
    if not np.isfinite(p).all():
        raise ValueError("p_i contains non-finite values")
    if not set(np.unique(r)).issubset({0, 1}):
        raise ValueError(f"rating must be binary 0/1, got {sorted(np.unique(r))}")

    ai_pred = (p >= decision_threshold).astype(int)
    q_ir = np.where(r == 0, p, 1.0 - p)
    disagreement = (r != ai_pred).astype(int)
    margin = np.abs(p - 0.5)

    return pd.DataFrame(
        {
            "ai_probability_class_1": p,
            "ai_predicted_class": ai_pred,
            "rater_ai_disagreement": disagreement,
            "ai_probability_rater_wrong": q_ir,
            "ai_margin": margin,
            "q_ir": q_ir,
        }
    )


def load_oof_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "ai_probability_class_1", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF file missing columns {sorted(missing)}: {path}")
    out = df.copy()
    out["case_id"] = pd.to_numeric(out["case_id"], errors="raise").astype(int)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    out["ai_probability_class_1"] = pd.to_numeric(
        out["ai_probability_class_1"], errors="raise"
    ).astype(float)
    if out["case_id"].duplicated().any():
        dups = out.loc[out["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate case_id in OOF predictions: {dups[:20]}")
    return out


def build_decision_level_table(
    *,
    user: pd.DataFrame,
    oof: pd.DataFrame,
    probability_source: str,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Expand one p_i per case onto all case–rater rows and attach AI features."""
    source = tag_probability_source(probability_source)

    keep_user = [
        "id",
        "case_id",
        "rater_id",
        "rating",
        "error-rating",
        "rating-confidence",
        "case-difficulty",
        "rater-expertise",
        # Excel globals kept for audit only; not used as CV features here.
        "rater-accuracy",
        "rater-confidence",
    ]
    missing = [c for c in keep_user if c not in user.columns]
    if missing:
        raise ValueError(f"USER table missing columns: {missing}")

    # Never carry TARGET into this table.
    if "TARGET" in user.columns or "target" in user.columns:
        raise ValueError("USER frame must not contain TARGET when building baseline features")

    oof_cols = oof[["case_id", "ai_probability_class_1", "fold"]].rename(
        columns={"ai_probability_class_1": "p_i"}
    )
    merged = user[keep_user].merge(oof_cols, on="case_id", how="inner", validate="many_to_one")

    if len(merged) != len(user):
        raise RuntimeError(
            f"Join OOF×USER lost rows: user={len(user)} merged={len(merged)}. "
            "Every USER case_id must appear in OOF predictions."
        )
    if merged["case_id"].nunique() != oof["case_id"].nunique():
        raise RuntimeError("Case-id coverage mismatch between OOF and USER after join")

    feats = compute_ai_decision_features(
        merged["p_i"].to_numpy(),
        merged["rating"].to_numpy(),
        decision_threshold=decision_threshold,
    )
    # Drop temporary p_i; ai_probability_class_1 is the public name.
    out = pd.concat([merged.drop(columns=["p_i"]), feats], axis=1)
    out["probability_source"] = source
    out["ai_decision_threshold"] = float(decision_threshold)

    # Column order: ids / label / AI features / context / provenance
    ordered = [
        "id",
        "case_id",
        "rater_id",
        "fold",
        "rating",
        "error-rating",
        "q_ir",
        *AI_FEATURE_COLS,
        "rating-confidence",
        "case-difficulty",
        "rater-expertise",
        "rater-accuracy",
        "rater-confidence",
        "probability_source",
        "ai_decision_threshold",
    ]
    return out[ordered].sort_values(["case_id", "rater_id"]).reset_index(drop=True)


def evaluate_q_ir_baseline(df: pd.DataFrame, out_dir: Path) -> dict:
    """Score q_ir against error-rating on all decision rows (descriptive; IC later)."""
    y = df["error-rating"].to_numpy(dtype=int)
    p = df["q_ir"].to_numpy(dtype=float)
    metrics = binary_metrics_with_calibration(y, p, threshold=0.5)
    flat = {"split": "all_decisions@0.5", "model": "q_ir", **metrics}
    append_metrics_row(out_dir / "baseline_q_ir_metrics.csv", flat)

    save_roc_curve(y, p, out_dir / "roc_q_ir.png", title="Baseline q_ir ROC (error-rating)")
    save_pr_curve(y, p, out_dir / "pr_q_ir.png", title="Baseline q_ir PR (error-rating)")
    save_calibration_plot(
        y, p, out_dir / "calibration_q_ir.png", title="Baseline q_ir calibration (error-rating)"
    )
    (out_dir / "calibration_note.txt").write_text(
        CALIBRATION_INTERPRETATION_NOTE + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build decision-level baseline features from OOF p_i (q_ir)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--oof",
        default=None,
        help="Path to OOF CSV (default: outputs/diagnostic/oof_predictions.csv)",
    )
    parser.add_argument(
        "--probability-source",
        default="nested_oof",
        help="Provenance tag for p_i (default nested_oof for outer OOF).",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold for ai_predicted_class (default 0.5).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    oof_path = (
        Path(args.oof)
        if args.oof
        else project_path(cfg["outputs"]["diagnostic_dir"], base_dir)
        / "oof_predictions.csv"
    )
    if not oof_path.is_file():
        raise FileNotFoundError(
            f"OOF predictions not found: {oof_path}. "
            "Train diagnostic first."
        )

    out_dir = project_path(cfg["outputs"]["baseline_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user = load_user_infos(cfg, base_dir=base_dir)
    oof = load_oof_predictions(oof_path)
    table = build_decision_level_table(
        user=user,
        oof=oof,
        probability_source=args.probability_source,
        decision_threshold=float(args.decision_threshold),
    )

    # Sanity: q_ir equals ai_probability_rater_wrong
    if not np.allclose(table["q_ir"], table["ai_probability_rater_wrong"]):
        raise RuntimeError("q_ir and ai_probability_rater_wrong diverged")

    out_csv = out_dir / "decision_level_features.csv"
    table.to_csv(out_csv, index=False)

    # Export clean OOF prediction file for the report (auto-refresh on each run).
    export_q_ir_oof(out_csv, out_dir / "oof_q_ir_predictions.csv")

    # Fresh per-run report table (append_metrics_row would otherwise accumulate rows).
    reset_metrics_file(out_dir / "baseline_q_ir_metrics.csv")
    metrics = evaluate_q_ir_baseline(table, out_dir)

    print(f"Wrote {out_csv} ({len(table)} rows, {table['case_id'].nunique()} cases)")
    print(f"probability_source={table['probability_source'].iloc[0]}")
    print(
        f"Baseline q_ir vs error-rating: "
        f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"prevalence={metrics['prevalence']:.4f} "
        f"majority_acc={metrics['majority_accuracy']:.4f}"
    )
    print(f"Plots under: {out_dir}")


if __name__ == "__main__":
    main()
