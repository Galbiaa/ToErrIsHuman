"""Error analysis for diagnostic AI, raters, baseline q_ir, and solution D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import load_config, load_user_infos, project_path
from metrics import binary_metrics_with_calibration
from reproducibility import bootstrap_run_reproducibility, set_global_seed


def _strip_target(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c == "TARGET" or str(c).upper() == "TARGET"]
    return df.drop(columns=drop) if drop else df


def load_analysis_table(cfg: dict, base_dir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
    )
    d_w = pd.read_csv(project_path(cfg["outputs"]["d_dir"], base_dir) / "oof_predictions.csv")
    d_u = project_path(cfg["outputs"]["d_dir"], base_dir) / "oof_predictions_unweighted.csv"
    oof_diag = pd.read_csv(
        project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    )

    user = _strip_target(load_user_infos(cfg, base_dir=base_dir))
    df = baseline[
        [
            "id",
            "case_id",
            "rater_id",
            "fold",
            "rating",
            "error-rating",
            "q_ir",
            "ai_probability_class_1",
            "ai_predicted_class",
            "rater_ai_disagreement",
            "rating-confidence",
            "case-difficulty",
            "rater-expertise",
        ]
    ].copy()
    df = df.merge(
        d_w[["id", "d_probability", "threshold_youden"]].rename(
            columns={"d_probability": "d_probability_weighted"}
        ),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if d_u.is_file():
        du = pd.read_csv(d_u)
        df = df.merge(
            du[["id", "d_probability"]].rename(
                columns={"d_probability": "d_probability_unweighted"}
            ),
            on="id",
            how="left",
            validate="one_to_one",
        )
    df = df.merge(
        oof_diag[["case_id", "target"]].rename(columns={"target": "TARGET"}),
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    # Predictions at 0.5
    df["q_pred"] = (df["q_ir"] >= 0.5).astype(int)
    df["d_pred_w"] = (df["d_probability_weighted"] >= 0.5).astype(int)
    if "d_probability_unweighted" in df.columns:
        df["d_pred_u"] = (df["d_probability_unweighted"] >= 0.5).astype(int)
    df["ai_correct"] = (df["ai_predicted_class"] == df["TARGET"]).astype(int)
    df["rater_correct"] = (1 - df["error-rating"]).astype(int)
    df["q_correct"] = (df["q_pred"] == df["error-rating"]).astype(int)
    df["d_correct_w"] = (df["d_pred_w"] == df["error-rating"]).astype(int)
    return df


def agreement_tables(df: pd.DataFrame) -> dict:
    """AI vs rater on TARGET.

    Decision-level counts (n≈5551) and case-level AI correctness (n≈427).
    """
    both_ok = int(((df["ai_correct"] == 1) & (df["rater_correct"] == 1)).sum())
    both_bad = int(((df["ai_correct"] == 0) & (df["rater_correct"] == 0)).sum())
    disagree = int((df["ai_correct"] != df["rater_correct"]).sum())
    ai_ok_rater_bad = int(((df["ai_correct"] == 1) & (df["rater_correct"] == 0)).sum())
    ai_bad_rater_ok = int(((df["ai_correct"] == 0) & (df["rater_correct"] == 1)).sum())

    cases = df.drop_duplicates("case_id")
    n_cases = int(len(cases))
    ai_cases_ok = int((cases["ai_correct"] == 1).sum())
    return {
        "unit_decision_rows": {
            "n": int(len(df)),
            "ai_and_rater_both_correct": both_ok,
            "ai_and_rater_both_wrong": both_bad,
            "ai_rater_disagreement_on_TARGET": disagree,
            "ai_correct_rater_wrong": ai_ok_rater_bad,
            "ai_wrong_rater_correct": ai_bad_rater_ok,
        },
        "unit_cases": {
            "n_cases": n_cases,
            "ai_correct_cases": ai_cases_ok,
            "ai_wrong_cases": n_cases - ai_cases_ok,
            "ai_case_accuracy": float(ai_cases_ok / n_cases) if n_cases else float("nan"),
        },
    }


def d_vs_baseline_corrections(df: pd.DataFrame, *, thr: float = 0.5) -> dict:
    """Where D fixes / breaks the image-only error baseline q_ir at a fixed threshold."""
    q_pred = (df["q_ir"] >= thr).astype(int)
    d_pred = (df["d_probability_weighted"] >= thr).astype(int)
    y = df["error-rating"].astype(int)
    q_ok = q_pred == y
    d_ok = d_pred == y
    q_wrong = ~q_ok
    d_wrong = ~d_ok
    return {
        "threshold": float(thr),
        "D_fixes_q_ir_errors": int((q_wrong & d_ok).sum()),
        "D_breaks_q_ir_correct": int((q_ok & d_wrong).sum()),
        "both_correct_on_error_rating": int((q_ok & d_ok).sum()),
        "both_wrong_on_error_rating": int((q_wrong & d_wrong).sum()),
        "net_D_minus_q_ir_correct": int((q_wrong & d_ok).sum()) - int((q_ok & d_wrong).sum()),
    }


def metrics_by_group(df: pd.DataFrame, group_col: str, p_col: str, thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        y = sub["error-rating"].to_numpy()
        p = sub[p_col].to_numpy()
        m = binary_metrics_with_calibration(y, p, threshold=thr)
        rows.append({"group_col": group_col, "group": g, "n": len(sub), **m})
    return pd.DataFrame(rows)


def fold_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, sub in df.groupby("fold"):
        y = sub["error-rating"].to_numpy()
        for name, col in (
            ("q_ir", "q_ir"),
            ("D_weighted", "d_probability_weighted"),
        ):
            m = binary_metrics_with_calibration(y, sub[col].to_numpy(), threshold=0.5)
            rows.append({"fold": int(fold), "model": name, **m})
    return pd.DataFrame(rows)


def weighting_comparison(df: pd.DataFrame) -> dict:
    if "d_probability_unweighted" not in df.columns:
        return {"note": "unweighted OOF not found"}
    y = df["error-rating"].to_numpy()
    mw = binary_metrics_with_calibration(y, df["d_probability_weighted"].to_numpy(), 0.5)
    mu = binary_metrics_with_calibration(y, df["d_probability_unweighted"].to_numpy(), 0.5)
    keys = ["auroc", "auprc", "brier", "recall", "specificity", "precision", "f1", "ece"]
    out = {"weighted": {k: mw[k] for k in keys}, "unweighted": {k: mu[k] for k in keys}}
    out["delta_weighted_minus_unweighted"] = {k: mw[k] - mu[k] for k in keys}
    return out


def what_d_improves(df: pd.DataFrame) -> dict:
    """Separate threshold-free gains from operating-point (threshold) effects."""
    y = df["error-rating"].to_numpy()
    pq = df["q_ir"].to_numpy()
    pd_ = df["d_probability_weighted"].to_numpy()
    thr_youden = float(df["threshold_youden"].mean())

    mq05 = binary_metrics_with_calibration(y, pq, 0.5)
    md05 = binary_metrics_with_calibration(y, pd_, 0.5)
    mq_y = binary_metrics_with_calibration(y, pq, thr_youden)
    md_y = binary_metrics_with_calibration(y, pd_, thr_youden)

    def _delta(md: dict, mq: dict) -> dict:
        return {
            "delta_auroc": md["auroc"] - mq["auroc"],
            "delta_auprc": md["auprc"] - mq["auprc"],
            "delta_brier": md["brier"] - mq["brier"],  # lower better
            "delta_log_loss": md["log_loss"] - mq["log_loss"],
            "delta_ece": md["ece"] - mq["ece"],
            "delta_precision": md["precision"] - mq["precision"],
            "delta_recall": md["recall"] - mq["recall"],
            "delta_specificity": md["specificity"] - mq["specificity"],
            "delta_f1": md["f1"] - mq["f1"],
        }

    d05 = _delta(md05, mq05)
    narrative = []
    # Threshold-free: AUROC/AUPRC/Brier/ECE do not depend on the hard cut (ECE/Brier are score-based)
    if d05["delta_auroc"] > 0:
        narrative.append("D improves discrimination (AUROC) vs q_ir.")
    else:
        narrative.append("D does not improve discrimination (AUROC) vs q_ir.")
    if d05["delta_auprc"] > 0:
        narrative.append("D improves AUPRC vs q_ir.")
    else:
        narrative.append("D does not improve AUPRC vs q_ir.")
    if d05["delta_brier"] < 0 or d05["delta_ece"] < 0:
        bits = []
        if d05["delta_brier"] < 0:
            bits.append("Brier")
        if d05["delta_ece"] < 0:
            bits.append("ECE")
        narrative.append("D improves calibration/scoring (" + ", ".join(bits) + " lower).")
    else:
        narrative.append("D does not improve Brier/ECE vs q_ir.")
    if d05["delta_precision"] > 0:
        narrative.append("At threshold 0.5, D has higher precision than q_ir.")
    if d05["delta_recall"] > 0:
        narrative.append("At threshold 0.5, D has higher recall than q_ir.")
    elif d05["delta_recall"] < 0:
        narrative.append("At threshold 0.5, D has lower recall than q_ir.")
    # If discrimination worse but hard metrics look better only at one cut -> threshold story
    disc_worse = d05["delta_auroc"] < 0 and d05["delta_auprc"] < 0
    cal_better = d05["delta_brier"] < 0 or d05["delta_ece"] < 0
    if disc_worse and cal_better:
        narrative.append(
            "Main gain vs q_ir is calibration/scoring, not ranking (discrimination)."
        )
    if disc_worse and not cal_better:
        narrative.append(
            "No clear ranking or calibration gain; check whether any hard-metric shift is only an operating-threshold effect."
        )

    return {
        "mean_D_fold_youden_threshold": thr_youden,
        "at_threshold_0.5": {
            "q_ir": {k: mq05[k] for k in ("auroc", "auprc", "brier", "ece", "precision", "recall", "specificity", "f1")},
            "D_weighted": {k: md05[k] for k in ("auroc", "auprc", "brier", "ece", "precision", "recall", "specificity", "f1")},
            "deltas_D_minus_q_ir": d05,
        },
        "at_mean_D_youden": {
            "threshold": thr_youden,
            "q_ir": {k: mq_y[k] for k in ("precision", "recall", "specificity", "f1", "accuracy")},
            "D_weighted": {k: md_y[k] for k in ("precision", "recall", "specificity", "f1", "accuracy")},
            "deltas_D_minus_q_ir": {
                k: md_y[k] - mq_y[k]
                for k in ("precision", "recall", "specificity", "f1", "accuracy")
            },
        },
        "corrections_at_0.5": d_vs_baseline_corrections(df, thr=0.5),
        "corrections_at_mean_youden": d_vs_baseline_corrections(df, thr=thr_youden),
        "reading": narrative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze errors: AI, raters, q_ir, D.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)
    set_global_seed(int(cfg.get("reproducibility", {}).get("seed", 42)))
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    out_dir = project_path(cfg["outputs"]["analysis_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_analysis_table(cfg, base_dir)
    df.to_csv(out_dir / "analysis_decision_table.csv", index=False)

    summary = {
        "agreement_ai_rater": agreement_tables(df),
        "d_vs_q_ir_corrections_at_0.5": d_vs_baseline_corrections(df, thr=0.5),
        "what_d_improves": what_d_improves(df),
        "weighting_comparison": weighting_comparison(df),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metrics_by_group(df, "rating-confidence", "d_probability_weighted").to_csv(
        out_dir / "metrics_by_rating_confidence_D.csv", index=False
    )
    metrics_by_group(df, "case-difficulty", "d_probability_weighted").to_csv(
        out_dir / "metrics_by_case_difficulty_D.csv", index=False
    )
    metrics_by_group(df, "rater-expertise", "d_probability_weighted").to_csv(
        out_dir / "metrics_by_expertise_D.csv", index=False
    )
    metrics_by_group(df, "rater_id", "d_probability_weighted").to_csv(
        out_dir / "metrics_by_rater_D.csv", index=False
    )
    # Error rate vs confidence / difficulty (descriptive)
    err_conf = (
        df.groupby("rating-confidence", as_index=False)["error-rating"]
        .mean()
        .rename(columns={"error-rating": "error_rate"})
    )
    err_conf.to_csv(out_dir / "error_rate_by_rating_confidence.csv", index=False)
    err_diff = (
        df.groupby("case-difficulty", as_index=False)["error-rating"]
        .mean()
        .rename(columns={"error-rating": "error_rate"})
    )
    err_diff.to_csv(out_dir / "error_rate_by_case_difficulty.csv", index=False)

    fold_stability(df).to_csv(out_dir / "fold_stability_metrics.csv", index=False)

    print(f"Wrote analysis under {out_dir}")
    print(json.dumps(summary["agreement_ai_rater"], indent=2))
    print(json.dumps(summary["d_vs_q_ir_corrections_at_0.5"], indent=2))
    print("Reading:", summary["what_d_improves"]["reading"])


if __name__ == "__main__":
    main()
