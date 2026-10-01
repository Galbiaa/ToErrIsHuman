"""Consolidated automated leakage / invariant guards for the two-stage pipeline.

Implements the explicit automated checks required by `Implementation Plan.md`
§18 (Leakage Guards) and the acceptance criteria of §24, using only the
committed artifacts (no re-training). Exits with status 1 if any check fails.

Usage:
    python -u src/verify_pipeline_guards.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_D_features import (  # noqa: E402
    build_d_feature_frame,
    d_feature_allowlist,
    extract_xy,
    make_d_preprocessor,
)
from data import load_config, load_ground_truth, load_user_infos, project_path  # noqa: E402
from generate_folds import load_fold_assignment  # noqa: E402
from methodology_guards import case_bootstrap_indices  # noqa: E402

EXPECTED_N_CASES = 427
EXPECTED_N_RATERS = 13
EXPECTED_N_ROWS = 5551


@dataclass
class Results:
    checks: list = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.checks.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    def info(self, name: str, detail: str = "") -> None:
        print(f"[INFO] {name}" + (f" — {detail}" if detail else ""))

    @property
    def all_ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def check_dataset_and_folds(cfg, base_dir, res):
    """§2, §3.2, §18.1, §24 — dataset counts, target definition, case disjointness."""
    gt = load_ground_truth(cfg, base_dir=base_dir)
    user = load_user_infos(cfg, base_dir=base_dir)
    folds = load_fold_assignment(
        project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    )

    res.check("§24 dataset: 427 cases", len(gt) == EXPECTED_N_CASES, f"got {len(gt)}")
    res.check("§24 dataset: 5551 decisions", len(user) == EXPECTED_N_ROWS, f"got {len(user)}")
    per = user.groupby("case_id").size()
    res.check("§24 dataset: 13 decisions per case", bool((per == EXPECTED_N_RATERS).all()))
    res.check("§24 dataset: 13 raters", user["rater_id"].nunique() == EXPECTED_N_RATERS)

    joined = user.merge(gt, on="case_id", how="left", validate="many_to_one")
    res.check(
        "§3.2 error-rating == 1(rating != TARGET)",
        bool((joined["error-rating"] == (joined["rating"] != joined["TARGET"]).astype(int)).all()),
    )

    res.check(
        "§24 folds: each case exactly once",
        folds["case_id"].nunique() == EXPECTED_N_CASES and not folds["case_id"].duplicated().any(),
    )
    ok = True
    for k in sorted(folds["fold"].unique()):
        tr = set(folds.loc[folds["fold"] != k, "case_id"])
        te = set(folds.loc[folds["fold"] == k, "case_id"])
        ok = ok and not (tr & te)
    res.check("§18.1 case disjointness (all folds)", ok)
    return gt, user, folds


def check_output1(cfg, base_dir, gt, res):
    """§3.1, §4 — image-only diagnostic: target TARGET, one outer-OOF per case."""
    oof = pd.read_csv(
        project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    )
    res.check(
        "§4.3 one outer-OOF prediction per case",
        len(oof) == EXPECTED_N_CASES and not oof["case_id"].duplicated().any(),
        f"rows={len(oof)}",
    )
    res.check("§3.1 diagnostic target is binary", set(oof["target"].unique()).issubset({0, 1}))
    p = oof["ai_probability_class_1"]
    res.check(
        "§4 probabilities finite and in [0,1]",
        bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()),
    )
    m = oof.merge(gt, on="case_id", how="left", validate="one_to_one")
    res.check("§3.1 OOF target == TARGET", bool((m["target"] == m["TARGET"]).all()))
    return oof


def check_output2(cfg, base_dir, oof, res):
    """§3.2, §5 — image-only baseline q_ir from outer-OOF p_i and the rating."""
    base = pd.read_csv(
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
    )
    res.check("§5.4 baseline table has 5551 rows", len(base) == EXPECTED_N_ROWS, f"got {len(base)}")

    p = base["ai_probability_class_1"].to_numpy()
    r = base["rating"].to_numpy()
    res.check("§5.1 q_ir = p_i if R=0 else 1-p_i", bool(np.allclose(base["q_ir"], np.where(r == 0, p, 1 - p))))
    res.check("§5.1 q_ir == ai_probability_rater_wrong", bool(np.allclose(base["q_ir"], base["ai_probability_rater_wrong"])))

    m = base.merge(
        oof[["case_id", "ai_probability_class_1"]].rename(columns={"ai_probability_class_1": "p_oof"}),
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    res.check("§5.2 baseline p_i == diagnostic outer-OOF", bool(np.allclose(m["ai_probability_class_1"], m["p_oof"])))

    exp = pd.read_csv(
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "oof_q_ir_predictions.csv"
    )
    res.check("§5.4 q_ir export has 5551 rows", len(exp) == EXPECTED_N_ROWS, f"got {len(exp)}")
    return base


def check_features(cfg, res):
    """§7, §18.5, Invariant 7 — D feature allowlist excludes TARGET / error-rating."""
    allow = d_feature_allowlist(cfg)
    res.check("§7 allowlist has exactly 11 features", len(allow) == 11, f"got {len(allow)}")
    res.check("§18.5 TARGET not in D allowlist", not any(f.upper() == "TARGET" for f in allow))
    res.check("§18.5 error-rating not in D allowlist", "error-rating" not in allow)
    return allow


def check_guards(cfg, base_dir, user, folds, allow, res):
    """§18.2–§18.8 — inner-OOF isolation, preprocessing isolation, fold alignment."""
    diag_dir = project_path(cfg["outputs"]["diagnostic_dir"], base_dir)
    d_dir = project_path(cfg["outputs"]["d_dir"], base_dir)
    model_dir = project_path(cfg["models"]["d_dir"], base_dir)
    diag_model_dir = project_path(cfg["models"]["diagnostic_dir"], base_dir)
    oof = pd.read_csv(diag_dir / "oof_predictions.csv")

    n_folds = int(cfg["validation"]["n_splits"])
    for k in range(1, n_folds + 1):
        train_cases = folds.loc[folds["fold"] != k, "case_id"].astype(int).tolist()
        test_cases = folds.loc[folds["fold"] == k, "case_id"].astype(int).tolist()

        # §18.2 inner-OOF isolation: cache must not contain outer-test cases
        inner = pd.read_csv(d_dir / f"nested_inner_oof_fold_{k}.csv")
        res.check(
            f"§18.2 inner-OOF fold {k} excludes test cases",
            len(set(inner["case_id"]) & set(test_cases)) == 0,
        )

        # §18.8 fold alignment: all fold-k artifacts exist
        aligned = all(
            [
                (diag_model_dir / f"diagnostic_fold_{k}.pt").is_file(),
                (d_dir / f"rater_profiles_foldsafe_fold_{k}.csv").is_file(),
                (model_dir / f"preprocessor_fold_{k}.joblib").is_file(),
                (model_dir / f"scenario_D_fold_{k}.joblib").is_file(),
            ]
        )
        res.check(f"§18.8 fold {k} artifacts aligned", aligned)

        payload = joblib.load(model_dir / f"scenario_D_fold_{k}.joblib")
        res.check(f"§18.8 D model fold {k} outer_fold", int(payload.get("outer_fold")) == k)
        res.check(
            f"§7 D model fold {k} feature_names == allowlist",
            list(payload.get("feature_names")) == allow,
        )

        # §18.7 preprocessing isolation: reproduce the fit on outer-train
        inner_f = inner[["case_id", "ai_probability_class_1"]]
        tf = build_d_feature_frame(
            user, inner_f, train_cases, config=cfg, probability_source="nested_oof", fold_col=None
        )
        tf = tf.loc[tf["case_id"].isin(set(train_cases))].reset_index(drop=True)
        X, _, _ = extract_xy(tf, cfg)
        pre_ref = make_d_preprocessor()
        pre_ref.fit(X)
        saved = joblib.load(model_dir / f"preprocessor_fold_{k}.joblib")
        same = (
            np.allclose(
                pre_ref.named_steps["imputer"].statistics_,
                saved.named_steps["imputer"].statistics_,
                equal_nan=True,
            )
            and np.allclose(pre_ref.named_steps["scaler"].mean_, saved.named_steps["scaler"].mean_)
            and np.allclose(pre_ref.named_steps["scaler"].scale_, saved.named_steps["scaler"].scale_)
        )
        res.check(f"§18.7 preprocessor fold {k} fit only on outer-train", same)

    # §18.6 test-label isolation (representative fold 1): test rater-accuracy
    # must come from outer-train cases only.
    k = 1
    train_cases = folds.loc[folds["fold"] != k, "case_id"].astype(int).tolist()
    test_cases = folds.loc[folds["fold"] == k, "case_id"].astype(int).tolist()
    test_probs = oof.loc[oof["case_id"].isin(set(test_cases))][["case_id", "ai_probability_class_1"]]
    test_feat = build_d_feature_frame(
        user, test_probs, train_cases, config=cfg, probability_source="nested_test", fold_col=None
    )
    test_feat = test_feat.loc[test_feat["case_id"].isin(set(test_cases))].reset_index(drop=True)
    agg = user[user["case_id"].isin(set(train_cases))].groupby("rater_id")["error-rating"].agg(["sum", "count"])
    exp_acc = test_feat.apply(
        lambda row: 1.0 - agg.loc[row["rater_id"], "sum"] / agg.loc[row["rater_id"], "count"], axis=1
    )
    res.check("§18.6 test rater-accuracy uses outer-train only", bool(np.allclose(test_feat["rater-accuracy"], exp_acc)))

    # §18.3 + §24: D pooled OOF coverage and outer-test provenance
    doof = pd.read_csv(d_dir / "oof_predictions.csv")
    res.check(
        "§24 D OOF 5551 rows, one per decision",
        len(doof) == EXPECTED_N_ROWS and doof["id"].nunique() == EXPECTED_N_ROWS,
    )
    srcs = set(doof["probability_source_ai"].unique())
    res.check("§18.3 outer-test p_i provenance = nested_test", srcs == {"nested_test"}, f"{sorted(srcs)}")


def check_evaluation(cfg, base_dir, res):
    """§19, §24, Invariant 8 — paired evaluation with case-level bootstrap."""
    comp = project_path(cfg["outputs"]["compare_dir"], base_dir)
    summary = json.loads((comp / "compare_summary.json").read_text(encoding="utf-8"))
    for s in summary:
        res.check(
            f"§24 {s['label']}: same 5551 decisions / 427 cases",
            int(s["n"]) == EXPECTED_N_ROWS and int(s["n_cases"]) == EXPECTED_N_CASES,
        )
    boot = pd.read_csv(comp / "weighted" / "bootstrap_paired_deltas.csv")
    res.check(
        "§19 bootstrap reports paired delta with 95% CI",
        {"delta_D_minus_q_ir", "delta_ci95_low", "delta_ci95_high"}.issubset(boot.columns),
    )

    # Invariant 8: the bootstrap resamples whole cases (each drawn case = 13*k rows)
    base = pd.read_csv(
        project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
    )
    case_ids = base["case_id"].to_numpy()
    idx = next(case_bootstrap_indices(case_ids, n_bootstrap=1, random_state=42))
    sizes = pd.Series(case_ids[idx]).value_counts().to_numpy()
    res.check(
        "§19 bootstrap resamples whole cases (13-row groups)",
        bool((sizes % EXPECTED_N_RATERS == 0).all()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidated leakage/invariant guards (§18/§24).")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    res = Results()
    gt, user, folds = check_dataset_and_folds(cfg, base_dir, res)
    oof = check_output1(cfg, base_dir, gt, res)
    check_output2(cfg, base_dir, oof, res)
    allow = check_features(cfg, res)
    check_guards(cfg, base_dir, user, folds, allow, res)
    check_evaluation(cfg, base_dir, res)

    # §20 reference benchmarks: reported for verification, not pass/fail.
    m = pd.read_csv(project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "metrics_pooled.csv")
    res.info("§20 diagnostic AUROC (benchmark ~0.759)", f"{float(m['auroc'].iloc[0]):.4f}")
    res.info("§20 diagnostic Brier (benchmark ~0.201)", f"{float(m['brier'].iloc[0]):.4f}")
    res.info("§26 Invariant 9", "no protocol tuning vs test - enforced by design (fixed seeds/protocol)")

    print()
    n_fail = sum(1 for _, ok, _ in res.checks if not ok)
    print(f"SUMMARY: {len(res.checks) - n_fail}/{len(res.checks)} checks passed")
    if not res.all_ok:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS - all leakage/invariant guards satisfied")


if __name__ == "__main__":
    main()
