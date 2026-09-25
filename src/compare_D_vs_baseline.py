"""Compare baseline q_ir vs solution D on error-rating (paired, case bootstrap).

Same 5551 decisions, same target. Bootstrap resamples case_id (all 13 rows together).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import load_config, project_path
from metrics import (
    CALIBRATION_INTERPRETATION_NOTE,
    append_metrics_row,
    binary_metrics_with_calibration,
    brier_skill_score,
    calibration_slope_intercept,
    expected_calibration_error,
    reset_metrics_file,
    save_calibration_plot,
    save_pr_curve,
    save_roc_curve,
)
from methodology_guards import case_bootstrap_indices
from reproducibility import bootstrap_run_reproducibility, set_global_seed

DISCRIM_KEYS = ("auroc", "auprc", "brier", "log_loss")


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        return float("nan")


def discrimination_bundle(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": _safe_auroc(y, p),
        "auprc": _safe_auprc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": _safe_logloss(y, p),
    }


def load_aligned_scores(
    *,
    baseline_csv: Path,
    d_csv: Path,
) -> pd.DataFrame:
    base = pd.read_csv(baseline_csv)
    dpred = pd.read_csv(d_csv)
    need_b = {"id", "case_id", "error-rating", "q_ir"}
    need_d = {"id", "case_id", "error-rating", "d_probability"}
    if need_b - set(base.columns):
        raise ValueError(f"Baseline missing {need_b - set(base.columns)}")
    if need_d - set(dpred.columns):
        raise ValueError(f"D OOF missing {need_d - set(dpred.columns)}")

    merged = base[["id", "case_id", "rater_id", "fold", "error-rating", "q_ir"]].merge(
        dpred[
            [
                c
                for c in [
                    "id",
                    "d_probability",
                    "threshold_youden",
                    "run",
                    "probability_source_ai",
                ]
                if c in dpred.columns
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 5551:
        raise RuntimeError(f"Expected 5551 aligned rows, got {len(merged)}")
    if not np.array_equal(
        merged["error-rating"].to_numpy(),
        base.set_index("id").loc[merged["id"], "error-rating"].to_numpy(),
    ):
        # labels must match between sources
        pass
    # Ensure label consistency across files
    d_lab = dpred.set_index("id").loc[merged["id"], "error-rating"].to_numpy()
    if not np.array_equal(merged["error-rating"].to_numpy(), d_lab):
        raise RuntimeError("error-rating mismatch between baseline and D OOF")
    return merged.sort_values(["case_id", "rater_id"]).reset_index(drop=True)


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Decision-curve net benefit at a probability threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if not (0.0 < threshold < 1.0):
        return float("nan")
    pred = (p >= threshold).astype(int)
    n = len(y)
    if n == 0:
        return float("nan")
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    return (tp / n) - (fp / n) * (threshold / (1.0 - threshold))


def decision_curve_frame(
    y: np.ndarray,
    p_base: np.ndarray,
    p_d: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    prev = float(np.mean(y))
    for t in thresholds:
        t = float(t)
        nb_all = prev - (1.0 - prev) * (t / (1.0 - t)) if 0 < t < 1 else float("nan")
        rows.append(
            {
                "threshold": t,
                "nb_treat_all": nb_all,
                "nb_treat_none": 0.0,
                "nb_q_ir": net_benefit(y, p_base, t),
                "nb_D": net_benefit(y, p_d, t),
            }
        )
    return pd.DataFrame(rows)


def save_dca_plot(dca: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(dca["threshold"], dca["nb_D"], label="D")
    plt.plot(dca["threshold"], dca["nb_q_ir"], label="q_ir baseline")
    plt.plot(dca["threshold"], dca["nb_treat_all"], linestyle="--", label="Treat all")
    plt.plot(dca["threshold"], dca["nb_treat_none"], linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_overlay_roc(
    y: np.ndarray, p_base: np.ndarray, p_d: np.ndarray, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    for name, p in (("q_ir", p_base), ("D", p_d)):
        fpr, tpr, _ = roc_curve(y, p)
        auc = _safe_auroc(y, p)
        plt.plot(fpr, tpr, label=f"{name} AUROC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC: D vs q_ir (error-rating)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def bootstrap_paired_deltas(
    case_ids: np.ndarray,
    y: np.ndarray,
    p_base: np.ndarray,
    p_d: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: int,
    metric_fns: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> pd.DataFrame:
    """Point estimate + bootstrap IC95% for each metric and Δ = D − baseline."""
    point_rows = []
    boot_store: Dict[str, List[float]] = {}

    def _record(name: str, base_v: float, d_v: float) -> None:
        delta = d_v - base_v
        point_rows.append(
            {
                "metric": name,
                "baseline_q_ir": base_v,
                "D": d_v,
                "delta_D_minus_q_ir": delta,
            }
        )

    for name, fn in metric_fns.items():
        b = fn(y, p_base)
        d = fn(y, p_d)
        _record(name, b, d)
        boot_store[f"{name}__base"] = []
        boot_store[f"{name}__d"] = []
        boot_store[f"{name}__delta"] = []

    for idx in case_bootstrap_indices(
        case_ids, n_bootstrap=n_bootstrap, random_state=random_state
    ):
        yb, pb, pd_ = y[idx], p_base[idx], p_d[idx]
        for name, fn in metric_fns.items():
            bv = fn(yb, pb)
            dv = fn(yb, pd_)
            boot_store[f"{name}__base"].append(bv)
            boot_store[f"{name}__d"].append(dv)
            boot_store[f"{name}__delta"].append(dv - bv)

    out = pd.DataFrame(point_rows)
    for name in metric_fns:
        for col, key in (
            ("baseline_ci95_low", f"{name}__base"),
            ("baseline_ci95_high", f"{name}__base"),
            ("D_ci95_low", f"{name}__d"),
            ("D_ci95_high", f"{name}__d"),
            ("delta_ci95_low", f"{name}__delta"),
            ("delta_ci95_high", f"{name}__delta"),
        ):
            arr = np.asarray(boot_store[key], dtype=float)
            arr = arr[np.isfinite(arr)]
            if col.endswith("low"):
                out.loc[out["metric"] == name, col] = (
                    float(np.quantile(arr, 0.025)) if len(arr) else float("nan")
                )
            else:
                out.loc[out["metric"] == name, col] = (
                    float(np.quantile(arr, 0.975)) if len(arr) else float("nan")
                )
    return out


def run_comparison(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    label: str,
    n_bootstrap: int,
    seed: int,
) -> Dict:
    y = df["error-rating"].to_numpy(dtype=int)
    p_base = df["q_ir"].to_numpy(dtype=float)
    p_d = df["d_probability"].to_numpy(dtype=float)
    case_ids = df["case_id"].to_numpy(dtype=int)

    sub = out_dir / label
    sub.mkdir(parents=True, exist_ok=True)
    # Fresh per-run report table (append_metrics_row would otherwise accumulate rows).
    reset_metrics_file(sub / "metrics_point.csv")

    # Point metrics @0.5
    m_base_05 = binary_metrics_with_calibration(y, p_base, threshold=0.5)
    m_d_05 = binary_metrics_with_calibration(y, p_d, threshold=0.5)
    append_metrics_row(sub / "metrics_point.csv", {"model": "q_ir", "threshold": 0.5, **m_base_05})
    append_metrics_row(sub / "metrics_point.csv", {"model": "D", "threshold": 0.5, **m_d_05})

    # D at stored Youden (from validation during train_D); q_ir at same numeric thr for pairing
    if "threshold_youden" in df.columns:
        thr = df["threshold_youden"].to_numpy(dtype=float)
        # Row-wise threshold: metrics need a single thr — use per-fold mode via masking
        # Evaluate with vectorized preds
        pred_d = (p_d >= thr).astype(int)
        pred_q = (p_base >= thr).astype(int)
        # Report discrimination (thr-free) already above; for thr-dependent use mean thr summary
        mean_thr = float(np.nanmean(thr))
        m_base_thr = binary_metrics_with_calibration(y, p_base, threshold=mean_thr)
        m_d_thr = binary_metrics_with_calibration(y, p_d, threshold=mean_thr)
        append_metrics_row(
            sub / "metrics_point.csv",
            {
                "model": "q_ir",
                "threshold": mean_thr,
                "threshold_note": "mean_of_D_fold_youden_for_paired_op_point",
                **m_base_thr,
            },
        )
        append_metrics_row(
            sub / "metrics_point.csv",
            {
                "model": "D",
                "threshold": mean_thr,
                "threshold_note": "mean_of_D_fold_youden",
                **m_d_thr,
            },
        )
        # Exact row-wise operating point summary
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        def _spec(yt, yp):
            tn = int(((yp == 0) & (yt == 0)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            return float(tn / (tn + fp)) if (tn + fp) else float("nan")

        row_ops = {
            "D_rowwise_youden_accuracy": float(accuracy_score(y, pred_d)),
            "D_rowwise_youden_balanced_accuracy": float(balanced_accuracy_score(y, pred_d)),
            "D_rowwise_youden_precision": float(precision_score(y, pred_d, zero_division=0)),
            "D_rowwise_youden_recall": float(recall_score(y, pred_d, zero_division=0)),
            "D_rowwise_youden_f1": float(f1_score(y, pred_d, zero_division=0)),
            "D_rowwise_youden_specificity": _spec(y, pred_d),
            "q_ir_same_thr_accuracy": float(accuracy_score(y, pred_q)),
            "q_ir_same_thr_balanced_accuracy": float(balanced_accuracy_score(y, pred_q)),
            "q_ir_same_thr_precision": float(precision_score(y, pred_q, zero_division=0)),
            "q_ir_same_thr_recall": float(recall_score(y, pred_q, zero_division=0)),
            "q_ir_same_thr_f1": float(f1_score(y, pred_q, zero_division=0)),
            "q_ir_same_thr_specificity": _spec(y, pred_q),
        }
        (sub / "operating_point_rowwise_youden.json").write_text(
            json.dumps(row_ops, indent=2), encoding="utf-8"
        )

    metric_fns = {
        "auroc": _safe_auroc,
        "auprc": _safe_auprc,
        "brier": lambda yt, pr: float(brier_score_loss(yt, pr)),
        "bss": lambda yt, pr: float(brier_skill_score(yt, pr)),
        "log_loss": _safe_logloss,
        "calibration_slope": lambda yt, pr: float(
            calibration_slope_intercept(yt, pr)["calibration_slope"]
        ),
        "calibration_intercept": lambda yt, pr: float(
            calibration_slope_intercept(yt, pr)["calibration_intercept"]
        ),
        "ece": lambda yt, pr: float(expected_calibration_error(yt, pr)),
        "precision@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["precision"]
        ),
        "recall@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["recall"]
        ),
        "specificity@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["specificity"]
        ),
        "f1@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["f1"]
        ),
        "accuracy@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["accuracy"]
        ),
        "balanced_accuracy@0.5": lambda yt, pr: float(
            binary_metrics_with_calibration(yt, pr, threshold=0.5)["balanced_accuracy"]
        ),
    }
    print(f"[{label}] Bootstrap n={n_bootstrap} (case_id)...", flush=True)
    boot = bootstrap_paired_deltas(
        case_ids,
        y,
        p_base,
        p_d,
        n_bootstrap=n_bootstrap,
        random_state=seed,
        metric_fns=metric_fns,
    )
    boot.to_csv(sub / "bootstrap_paired_deltas.csv", index=False)

    # DCA
    thresholds = np.linspace(0.05, 0.80, 16)
    dca = decision_curve_frame(y, p_base, p_d, thresholds)
    dca.to_csv(sub / "decision_curve.csv", index=False)
    save_dca_plot(
        dca,
        sub / "decision_curve.png",
        title=f"Decision curve ({label}) — illustrative; not clinical policy",
    )
    (sub / "dca_note.txt").write_text(
        "Decision-curve net benefit depends on prevalence and on the chosen utility "
        "trade-off (threshold). Curves here are descriptive for research comparison "
        "on this dataset; they are not a recommendation for clinical decision rules.\n",
        encoding="utf-8",
    )

    # Plots
    save_overlay_roc(y, p_base, p_d, sub / "roc_overlay.png")
    save_roc_curve(y, p_base, sub / "roc_q_ir.png", title="q_ir ROC")
    save_roc_curve(y, p_d, sub / "roc_D.png", title="D ROC")
    save_pr_curve(y, p_base, sub / "pr_q_ir.png", title="q_ir PR")
    save_pr_curve(y, p_d, sub / "pr_D.png", title="D PR")
    save_calibration_plot(y, p_base, sub / "calibration_q_ir.png", title="q_ir calibration")
    save_calibration_plot(y, p_d, sub / "calibration_D.png", title="D calibration")
    (sub / "calibration_note.txt").write_text(
        CALIBRATION_INTERPRETATION_NOTE + "\n", encoding="utf-8"
    )

    summary = {
        "label": label,
        "n": int(len(df)),
        "n_cases": int(df["case_id"].nunique()),
        "prevalence": float(y.mean()),
        "auroc_q_ir": m_base_05["auroc"],
        "auroc_D": m_d_05["auroc"],
        "auprc_q_ir": m_base_05["auprc"],
        "auprc_D": m_d_05["auprc"],
        "delta_auroc": m_d_05["auroc"] - m_base_05["auroc"],
        "brier_q_ir": float(brier_score_loss(y, p_base)),
        "brier_D": float(brier_score_loss(y, p_d)),
        "bss_q_ir": float(brier_skill_score(y, p_base)),
        "bss_D": float(brier_skill_score(y, p_d)),
        "majority_accuracy": m_base_05["majority_accuracy"],
        "bootstrap_csv": str(sub / "bootstrap_paired_deltas.csv"),
    }
    (sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


UNIFIED_METRIC_KEYS = (
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
)

UNIFIED_METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "log_loss": "log_loss",
    "precision": "precision",
    "recall": "recall",
    "specificity": "specificity",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "F1",
}


def generate_unified_metrics_table(out_dir: Path, d_dir: Path) -> Optional[Path]:
    """Side-by-side pooled OOF metrics: D weighted vs D unweighted.

    Reads the pipeline's pooled metric CSVs (one row each, written by
    train_D.py for the whole OOF at threshold 0.5) and writes a long table
    with columns [metric, D_weighted, D_unweighted]. Missing variants are
    skipped; returns None if no pooled CSV is available.
    """
    variants = {
        "D_weighted": d_dir / "metrics_pooled_weighted.csv",
        "D_unweighted": d_dir / "metrics_pooled_unweighted.csv",
    }
    frames: Dict[str, pd.DataFrame] = {}
    for name, path in variants.items():
        if not path.is_file():
            print(f"[unified-table] SKIP {name}: {path} not found", flush=True)
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"[unified-table] SKIP {name}: {path} is empty", flush=True)
            continue
        missing = [k for k in UNIFIED_METRIC_KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{path}: pooled metrics missing columns {missing}")
        if len(df) > 1:
            print(
                f"[unified-table] WARN {path} has {len(df)} rows; using the last one",
                flush=True,
            )
            df = df.tail(1)
        frames[name] = df

    if not frames:
        print("[unified-table] No pooled D metric CSVs found; table not written", flush=True)
        return None

    rows = []
    for key in UNIFIED_METRIC_KEYS:
        row = {"metric": UNIFIED_METRIC_LABELS[key]}
        for name, df in frames.items():
            row[name] = float(df.iloc[0][key])
        rows.append(row)

    table = pd.DataFrame(rows)
    out_csv = out_dir / "unified_D_metrics_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"[unified-table] Wrote {out_csv} ({len(table)} metrics)", flush=True)
    return out_csv


def compare_nested_vs_fast(out_dir: Path, d_dir: Path) -> Optional[Path]:
    """Compare D nested AUROC vs D fast AUROC on the same OOF decisions.

    The fast protocol uses in-sample probabilities on outer-train (optimistic),
    while the nested protocol uses inner-CV OOF (unseen).  Both use out-of-sample
    outer-test probabilities.  We align both weighted OOF files by id and report
    discrimination metrics for each, so the report can quote the expected
    ~0.63 (nested) vs ~0.70 (fast) AUROC gap.
    """
    nested_csv = d_dir / "oof_predictions.csv"
    fast_csv = d_dir / "oof_predictions_fast.csv"

    if not nested_csv.is_file():
        print(f"[nested-vs-fast] SKIP: nested OOF not found: {nested_csv}", flush=True)
        return None
    if not fast_csv.is_file():
        print(f"[nested-vs-fast] SKIP: fast OOF not found: {fast_csv}", flush=True)
        return None

    nested = pd.read_csv(nested_csv)
    fast = pd.read_csv(fast_csv)

    # Both are the *weighted* OOF; align by decision id (one-to-one).
    merged = nested[["id", "case_id", "fold", "error-rating", "d_probability"]].merge(
        fast[["id", "d_probability"]].rename(
            columns={"d_probability": "d_probability_fast"}
        ),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(nested):
        raise RuntimeError(
            f"nested-vs-fast alignment mismatch: expected {len(nested)} rows, "
            f"got {len(merged)}"
        )

    y = merged["error-rating"].to_numpy()
    p_nested = merged["d_probability"].to_numpy()
    p_fast = merged["d_probability_fast"].to_numpy()

    b_nested = discrimination_bundle(y, p_nested)
    b_fast = discrimination_bundle(y, p_fast)

    row = {
        "n_decisions": int(len(y)),
        "n_cases": int(merged["case_id"].nunique()),
        "prevalence": float(np.mean(y)),
        "auroc_nested": b_nested["auroc"],
        "auroc_fast": b_fast["auroc"],
        "delta_auroc_fast_minus_nested": b_fast["auroc"] - b_nested["auroc"],
        "auprc_nested": b_nested["auprc"],
        "auprc_fast": b_fast["auprc"],
        "brier_nested": b_nested["brier"],
        "brier_fast": b_fast["brier"],
        "log_loss_nested": b_nested["log_loss"],
        "log_loss_fast": b_fast["log_loss"],
    }

    out_csv = out_dir / "nested_vs_fast_comparison.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(
        f"[nested-vs-fast] AUROC nested={row['auroc_nested']:.4f} "
        f"fast={row['auroc_fast']:.4f} "
        f"delta={row['delta_auroc_fast_minus_nested']:+.4f}",
        flush=True,
    )
    print(f"[nested-vs-fast] Wrote {out_csv}", flush=True)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare q_ir vs D with case bootstrap.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--d-oof",
        default=None,
        help="D OOF csv (default: outputs/D/oof_predictions.csv)",
    )
    parser.add_argument(
        "--also-unweighted",
        action="store_true",
        default=True,
        help="Also compare unweighted D OOF if present (default true).",
    )
    parser.add_argument("--no-unweighted", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    out_dir = project_path(cfg["outputs"]["compare_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = (
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
    )
    d_dir = project_path(cfg["outputs"]["d_dir"], base_dir)
    d_oof = Path(args.d_oof) if args.d_oof else d_dir / "oof_predictions.csv"

    runs = [("weighted", d_oof)]
    if not args.no_unweighted:
        uw = d_dir / "oof_predictions_unweighted.csv"
        if uw.is_file():
            runs.append(("unweighted", uw))

    all_summaries = []
    for label, path in runs:
        print(f"=== Compare q_ir vs D ({label}) ===", flush=True)
        df = load_aligned_scores(baseline_csv=baseline_csv, d_csv=path)
        summary = run_comparison(
            df,
            out_dir=out_dir,
            label=label,
            n_bootstrap=int(args.n_bootstrap),
            seed=seed,
        )
        all_summaries.append(summary)
        print(
            f"  AUROC q_ir={summary['auroc_q_ir']:.4f}  D={summary['auroc_D']:.4f}  "
            f"delta={summary['delta_auroc']:.4f}",
            flush=True,
        )

    (out_dir / "compare_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )

    # Step 5: side-by-side pooled metrics table for D weighted vs unweighted.
    generate_unified_metrics_table(out_dir, d_dir)

    # Step 6 (plan): D nested AUROC vs D fast AUROC (optimistic declared variant).
    compare_nested_vs_fast(out_dir, d_dir)

    print(f"Wrote results under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
