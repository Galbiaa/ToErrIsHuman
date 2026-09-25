"""Train solution D with nested (default) or fast protocol.

Nested (scientific):
  A) inner-CV OOF p_i on outer-train cases
  B) outer-test p_i from diagnostic trained without those test cases
     
  C) fold-safe D features + preprocessor + XGBoost; threshold on inner val cases

Fast is available via --protocol fast (optimistic, declared):
  A') diagnostic trained on ALL outer-train
  B') outer-train p_i are IN-SAMPLE (fast_insample, declared optimism)
      outer-test p_i are out-of-sample (fast_test)
  C) same fold-safe D features + preprocessor + XGBoost as nested
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_D_features import (
    build_d_feature_frame,
    d_feature_allowlist,
    extract_xy,
    fit_transform_d_features,
    make_d_preprocessor,
    rater_profile_table,
    save_preprocessor,
)
from build_baseline import load_oof_predictions
from data import load_config, load_diagnostic_case_table, load_user_infos, project_path
from export_oof_predictions import export_d_oof
from generate_folds import load_fold_assignment
from metrics import append_metrics_row, binary_metrics_with_calibration, reset_metrics_file
from methodology_guards import assert_case_disjoint, select_threshold_on_validation
from reproducibility import bootstrap_run_reproducibility, set_global_seed, xgb_seed_for_fold
from tqdm import tqdm
from train_diagnostic import (
    get_device,
    logits_to_prob,
    make_loader,
    make_transforms,
    predict_logits,
    split_train_val,
    train_one_fold,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _strip_target(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c == "TARGET" or str(c).upper() == "TARGET"]
    return df.drop(columns=drop) if drop else df


def make_xgb(cfg: dict, *, fold: int, scale_pos_weight: Optional[float]) -> XGBClassifier:
    xcfg = cfg["solution_d"]["xgb"]
    params = dict(
        n_estimators=int(xcfg["n_estimators"]),
        max_depth=int(xcfg["max_depth"]),
        learning_rate=float(xcfg["learning_rate"]),
        subsample=float(xcfg["subsample"]),
        colsample_bytree=float(xcfg["colsample_bytree"]),
        objective=str(xcfg["objective"]),
        eval_metric=str(xcfg["eval_metric"]),
        random_state=xgb_seed_for_fold(int(xcfg["base_seed"]), fold),
        n_jobs=4,
        tree_method="hist",
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


def compute_inner_oof_probabilities(
    *,
    outer_train_cases: pd.DataFrame,
    cfg: dict,
    image_dir: Path,
    device: torch.device,
    outer_fold: int,
    cache_path: Path,
) -> pd.DataFrame:
    """A) Inner CV OOF diagnostic probabilities for outer-train cases."""
    if cache_path.is_file():
        _log(f"  [A] Loading cached inner OOF: {cache_path}")
        cached = pd.read_csv(cache_path)
        need = set(outer_train_cases["case_id"].astype(int))
        got = set(cached["case_id"].astype(int))
        if need == got and not cached["case_id"].duplicated().any():
            return cached
        _log("  [A] Cache mismatch — recomputing inner OOF")

    inner_n = int(cfg["solution_d"].get("inner_n_splits", 4))
    y = outer_train_cases["target"].to_numpy()
    skf = StratifiedKFold(
        n_splits=inner_n,
        shuffle=True,
        random_state=int(cfg["validation"]["random_state"]) + 100 * outer_fold,
    )
    rows: List[dict] = []
    case_ids = outer_train_cases["case_id"].to_numpy()

    for inner_i, (tr_idx, te_idx) in enumerate(skf.split(case_ids, y), start=1):
        inner_train = outer_train_cases.iloc[tr_idx].reset_index(drop=True)
        inner_test = outer_train_cases.iloc[te_idx].reset_index(drop=True)
        assert_case_disjoint(inner_train["case_id"], inner_test["case_id"])
        fit_df, es_df = split_train_val(
            inner_train,
            val_fraction=0.2,
            random_state=int(cfg["validation"]["random_state"]) + outer_fold * 10 + inner_i,
        )
        _log(
            f"  [A] outer={outer_fold} inner={inner_i}/{inner_n} "
            f"fit={len(fit_df)} es={len(es_df)} pred={len(inner_test)}"
        )
        model, meta = train_one_fold(
            train_df=fit_df,
            val_df=es_df,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
        )
        _log(f"      best_epoch={meta['best_epoch']} val_auroc={meta['best_val_auroc']:.4f}")
        loader = make_loader(
            inner_test,
            image_dir=image_dir,
            orientations=list(cfg["images"]["orientations"]),
            extension=cfg["images"].get("extension", "jpg"),
            transform=make_transforms(cfg, train=False),
            batch_size=int(cfg["diagnostic"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["diagnostic"].get("num_workers", 0)),
        )
        logits, _, cases = predict_logits(model, loader, device)
        probs = logits_to_prob(logits)
        for c, p in zip(cases.astype(int), probs.astype(float)):
            rows.append(
                {
                    "case_id": int(c),
                    "ai_probability_class_1": float(p),
                    "outer_fold": outer_fold,
                    "inner_fold": inner_i,
                    "probability_source": "nested_oof",
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    if out["case_id"].duplicated().any() or len(out) != len(outer_train_cases):
        raise RuntimeError("Inner OOF incomplete or duplicated")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    _log(f"  [A] Saved inner OOF cache: {cache_path}")
    return out


def outer_test_probabilities_from_diagnostic(
    oof_diagnostic: pd.DataFrame,
    test_case_ids: List[int],
    *,
    outer_fold: int,
) -> pd.DataFrame:

    test_set = set(int(x) for x in test_case_ids)
    sub = oof_diagnostic.loc[oof_diagnostic["case_id"].isin(test_set)].copy()
    if len(sub) != len(test_set):
        raise RuntimeError(
            f"Diagnostic OOF missing test cases for outer fold {outer_fold}: "
            f"need {len(test_set)} got {len(sub)}"
        )
    sub = sub[["case_id", "ai_probability_class_1"]].copy()
    sub["outer_fold"] = outer_fold
    sub["probability_source"] = "nested_test"
    return sub.reset_index(drop=True)


def compute_fast_probabilities(
    *,
    outer_train_df: pd.DataFrame,
    outer_test_df: pd.DataFrame,
    cfg: dict,
    image_dir: Path,
    device: torch.device,
    outer_fold: int,
    cache_path: Path,
) -> pd.DataFrame:
    """Fast A'/B' stage: diagnostic trained on ALL outer-train.

    Key difference from nested (see PLAN):
      nested: outer-train probabilities from inner CV OOF (model never saw the case)
      fast:   outer-train probabilities are IN-SAMPLE from a diagnostic trained on
              ALL outer-train (probability_source=fast_insample, declared optimism).
    Outer-test probabilities stay out-of-sample (fast_test). Cached per fold so
    re-runs / `--fold k` resumes do not retrain the diagnostic.
    """
    if cache_path.is_file():
        _log(f"  [fast] Loading cached fast probabilities: {cache_path}")
        cached = pd.read_csv(cache_path)
        if cached["case_id"].duplicated().any():
            _log("  [fast] Cache has duplicate case_ids — recomputing")
        else:
            tr_need = set(outer_train_df["case_id"].astype(int))
            te_need = set(outer_test_df["case_id"].astype(int))
            tr = set(cached.loc[cached["split"] == "train", "case_id"].astype(int))
            te = set(cached.loc[cached["split"] == "test", "case_id"].astype(int))
            if tr == tr_need and te == te_need:
                return cached
            _log("  [fast] Cache mismatch — recomputing")

    # Early-stopping split of outer-train (same scheme as nested inner training).
    fit_df, es_df = split_train_val(
        outer_train_df,
        val_fraction=0.2,
        random_state=int(cfg["validation"]["random_state"]) + outer_fold * 10,
    )
    _log(
        f"  [fast] outer={outer_fold} training diagnostic on ALL outer-train: "
        f"fit={len(fit_df)} es={len(es_df)} train_pred={len(outer_train_df)} "
        f"test_pred={len(outer_test_df)}"
    )
    model, meta = train_one_fold(
        train_df=fit_df,
        val_df=es_df,
        cfg=cfg,
        image_dir=image_dir,
        device=device,
    )
    _log(f"      best_epoch={meta['best_epoch']} val_auroc={meta['best_val_auroc']:.4f}")

    def _predict(df: pd.DataFrame, split: str, source: str) -> List[dict]:
        loader = make_loader(
            df,
            image_dir=image_dir,
            orientations=list(cfg["images"]["orientations"]),
            extension=cfg["images"].get("extension", "jpg"),
            transform=make_transforms(cfg, train=False),
            batch_size=int(cfg["diagnostic"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["diagnostic"].get("num_workers", 0)),
        )
        logits, _, cases = predict_logits(model, loader, device)
        probs = logits_to_prob(logits)
        return [
            {
                "case_id": int(c),
                "ai_probability_class_1": float(p),
                "split": split,
                "outer_fold": outer_fold,
                "probability_source": source,
            }
            for c, p in zip(cases.astype(int), probs.astype(float))
        ]

    rows = _predict(outer_train_df, "train", "fast_insample")
    rows += _predict(outer_test_df, "test", "fast_test")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    if out["case_id"].duplicated().any() or len(out) != len(outer_train_df) + len(outer_test_df):
        raise RuntimeError("Fast probabilities incomplete or duplicated")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    _log(f"  [fast] Saved fast probabilities cache: {cache_path}")
    return out


def split_cases_for_threshold(
    case_ids: np.ndarray,
    y_case: np.ndarray,
    *,
    val_fraction: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified case split for D threshold validation (by mean error proxy / TARGET)."""
    if len(case_ids) < 5:
        raise ValueError("Not enough cases for D threshold split")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=random_state
    )
    tr, va = next(splitter.split(case_ids, y_case))
    return case_ids[tr], case_ids[va]


def train_d_one_outer_fold(
    *,
    outer_fold: int,
    user: pd.DataFrame,
    cases: pd.DataFrame,
    folds: pd.DataFrame,
    oof_diagnostic: Optional[pd.DataFrame],
    cfg: dict,
    image_dir: Path,
    device: torch.device,
    out_dir: Path,
    model_dir: Path,
    protocol: str = "nested",
) -> Dict[str, pd.DataFrame]:
    test_cases = folds.loc[folds["fold"] == outer_fold, "case_id"].astype(int).tolist()
    train_cases = folds.loc[folds["fold"] != outer_fold, "case_id"].astype(int).tolist()
    assert_case_disjoint(train_cases, test_cases)

    outer_train_df = cases.loc[cases["case_id"].isin(train_cases)].reset_index(drop=True)
    outer_test_df = cases.loc[cases["case_id"].isin(test_cases)].reset_index(drop=True)
    _log(f"\n=== Outer fold {outer_fold}: train_cases={len(outer_train_df)} test_cases={len(outer_test_df)} ===")

    if protocol == "fast":
        cache = out_dir / f"fast_probs_fold_{outer_fold}.csv"
        fast_probs = compute_fast_probabilities(
            outer_train_df=outer_train_df,
            outer_test_df=outer_test_df,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
            outer_fold=outer_fold,
            cache_path=cache,
        )
        inner_oof = fast_probs.loc[fast_probs["split"] == "train"].reset_index(drop=True)
        test_probs = fast_probs.loc[fast_probs["split"] == "test"].reset_index(drop=True)
        train_source = "fast_insample"
        test_source = "fast_test"
    else:
        cache = out_dir / f"nested_inner_oof_fold_{outer_fold}.csv"
        inner_oof = compute_inner_oof_probabilities(
            outer_train_cases=outer_train_df,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
            outer_fold=outer_fold,
            cache_path=cache,
        )
        test_probs = outer_test_probabilities_from_diagnostic(
            oof_diagnostic, test_cases, outer_fold=outer_fold
        )
        train_source = "nested_oof"
        test_source = "nested_test"

    # C) Features — separate frames for provenance tags, same train_case_ids
    _log(f"  [C] Building fold-safe D features ({protocol})")
    train_feat = build_d_feature_frame(
        user,
        inner_oof.rename(columns={})[["case_id", "ai_probability_class_1"]],
        train_cases,
        config=cfg,
        probability_source=train_source,
        fold_col=None,
    )
    train_feat["fold"] = outer_fold
    train_feat["probability_source"] = train_source

    test_feat = build_d_feature_frame(
        user,
        test_probs[["case_id", "ai_probability_class_1"]],
        train_cases,
        config=cfg,
        probability_source=test_source,
        fold_col=None,
    )
    test_feat["fold"] = outer_fold
    test_feat["probability_source"] = test_source

    # Keep only rows whose cases are in the intended split (builder returns all USER rows)
    train_feat = train_feat.loc[train_feat["case_id"].isin(train_cases)].reset_index(drop=True)
    test_feat = test_feat.loc[test_feat["case_id"].isin(test_cases)].reset_index(drop=True)

    profiles = rater_profile_table(train_feat, train_cases)
    profiles.to_csv(out_dir / f"rater_profiles_foldsafe_fold_{outer_fold}.csv", index=False)

    # Threshold validation split on outer-train cases (stratified by TARGET)
    uniq_train_cases = outer_train_df["case_id"].to_numpy()
    uniq_y = outer_train_df["target"].to_numpy()
    fit_cases, thr_cases = split_cases_for_threshold(
        uniq_train_cases,
        uniq_y,
        val_fraction=0.2,
        random_state=int(cfg["validation"]["random_state"]) + outer_fold,
    )
    assert_case_disjoint(fit_cases, thr_cases)

    fit_rows = train_feat.loc[train_feat["case_id"].isin(set(fit_cases))]
    thr_rows = train_feat.loc[train_feat["case_id"].isin(set(thr_cases))]
    X_fit, y_fit, feat_names = extract_xy(fit_rows, cfg)
    X_thr, y_thr, _ = extract_xy(thr_rows, cfg)
    X_all_train, y_all_train, _ = extract_xy(train_feat, cfg)
    X_test, y_test, _ = extract_xy(test_feat, cfg)

    n_pos = max(int((y_all_train == 1).sum()), 1)
    n_neg = int((y_all_train == 0).sum())

    runs = []
    if bool(cfg["solution_d"].get("use_scale_pos_weight", True)):
        runs.append(("weighted", float(n_neg) / float(n_pos)))
    if bool(cfg["solution_d"].get("also_run_without_weighting", True)):
        runs.append(("unweighted", None))
    if not runs:
        runs.append(("weighted", None))

        # Document the effective scale_pos_weight (n_negative / n_positive) per fold.
    proto_suffix = "_fast" if protocol == "fast" else ""
    for run_name, spw in runs:
        if spw is None:
            continue
        _log(
            f"    scale_pos_weight[{run_name}] fold={outer_fold}: "
            f"n_pos={int(n_pos)} n_neg={int(n_neg)} spw={float(spw):.6f} "
            f"(n_negative/n_positive)"
        )
        append_metrics_row(
            out_dir / f"scale_pos_weight_per_fold{proto_suffix}.csv",
            {
                "fold": int(outer_fold),
                "run": str(run_name),
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
                "scale_pos_weight": float(spw),
                "formula": "n_negative / n_positive",
            },
        )

    results: Dict[str, pd.DataFrame] = {}
    for run_name, spw in runs:
        _log(f"  [C] XGBoost {run_name} (scale_pos_weight={spw})")
        # Model for threshold: fit on fit_cases only
        pre_thr = make_d_preprocessor()
        X_fit_t, X_thr_t, pre_thr = fit_transform_d_features(pre_thr, X_fit, X_thr)
        clf_thr = make_xgb(cfg, fold=outer_fold, scale_pos_weight=spw)
        clf_thr.fit(X_fit_t, y_fit)
        # Guard: predict_proba[:, 1] must be P(error-rating=1) (labels are {0, 1}).
        assert list(clf_thr.classes_) == [0, 1], f"class order: {clf_thr.classes_}"
        p_thr = clf_thr.predict_proba(X_thr_t)[:, 1]
        thr = select_threshold_on_validation(y_thr, p_thr, method="youden")

        # Final model: preprocessor + XGB on all outer-train
        pre = make_d_preprocessor()
        X_tr_t, X_te_t, pre = fit_transform_d_features(pre, X_all_train, X_test)
        clf = make_xgb(cfg, fold=outer_fold, scale_pos_weight=spw)
        clf.fit(X_tr_t, y_all_train)
        # Guard: predict_proba[:, 1] must be P(error-rating=1) (labels are {0, 1}).
        assert list(clf.classes_) == [0, 1], f"class order: {clf.classes_}"
        p_test = clf.predict_proba(X_te_t)[:, 1]

        suffix = proto_suffix + ("" if run_name == "weighted" else "_unweighted")
        pre_path = model_dir / f"preprocessor_fold_{outer_fold}{suffix}.joblib"
        model_path = model_dir / f"scenario_D_fold_{outer_fold}{suffix}.joblib"
        save_preprocessor(pre, pre_path)
        joblib.dump(
            {
                "model": clf,
                "feature_names": feat_names,
                "threshold_youden": float(thr),
                "scale_pos_weight": spw,
                "run_name": run_name,
                "outer_fold": outer_fold,
                "protocol": protocol,
                "train_probability_source": train_source,
                "test_probability_source": test_source,
            },
            model_path,
        )
        _log(f"      saved {model_path.name} thr={thr:.4f}")

        metrics = binary_metrics_with_calibration(y_test, p_test, threshold=float(thr))
        append_metrics_row(
            out_dir / f"metrics_per_fold_{run_name}{proto_suffix}.csv",
            {"fold": outer_fold, "run": run_name, "threshold": thr, **metrics},
        )

        pred = test_feat[["id", "case_id", "rater_id", "fold", "error-rating"]].copy()
        pred["d_probability"] = p_test
        pred["probability_source_ai"] = test_feat["probability_source"].to_numpy()
        pred["run"] = run_name
        pred["threshold_youden"] = float(thr)
        results[run_name] = pred

    return results


def run_nested(cfg: dict, base_dir: Path, *, only_fold: Optional[int] = None) -> None:
    out_dir = project_path(cfg["outputs"]["d_dir"], base_dir)
    model_dir = project_path(cfg["models"]["d_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["diagnostic"].get("device", "auto"))
    image_dir = project_path(cfg["data"]["image_dir"], base_dir)
    _log(f"Device: {device}")
    _log(f"D allowlist: {d_feature_allowlist(cfg)}")

    folds = load_fold_assignment(
        project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    )
    cases = load_diagnostic_case_table(cfg, base_dir=base_dir)
    cases = cases.merge(folds[["case_id", "fold"]], on="case_id", how="inner", validate="one_to_one")
    user = _strip_target(load_user_infos(cfg, base_dir=base_dir))
    oof_diagnostic = load_oof_predictions(
        project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    )

    n_splits = int(cfg["validation"]["n_splits"])
    fold_range = [only_fold] if only_fold is not None else list(range(1, n_splits + 1))

    # Idempotent per-run report tables: a full run starts from clean files.
    # Single-fold `--fold k` resumes keep the append semantics (do not wipe
    # rows already written by previous folds of the same run).
    if only_fold is None:
        for fname in (
            "scale_pos_weight_per_fold.csv",
            "metrics_per_fold_weighted.csv",
            "metrics_per_fold_unweighted.csv",
            "metrics_pooled_weighted.csv",
            "metrics_pooled_unweighted.csv",
        ):
            reset_metrics_file(out_dir / fname)
            _log(f"  [clean] removed previous {fname} (fresh per-run table)")

    collected: Dict[str, List[pd.DataFrame]] = {"weighted": [], "unweighted": []}
    fold_pbar = tqdm(
        fold_range,
        desc="D nested folds",
        unit="fold",
    )
    for fold in fold_pbar:
        fold_pbar.set_postfix({"fold": fold})
        fold_results = train_d_one_outer_fold(
            outer_fold=fold,
            user=user,
            cases=cases,
            folds=folds,
            oof_diagnostic=oof_diagnostic,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
            out_dir=out_dir,
            model_dir=model_dir,
        )
        for name, df in fold_results.items():
            collected.setdefault(name, []).append(df)

    if only_fold is not None:
        _log(f"Single-fold run done (fold={only_fold}). Re-run without --fold to pool OOF.")
        return

    for run_name, parts in collected.items():
        if not parts:
            continue
        oof = pd.concat(parts, ignore_index=True).sort_values(["case_id", "rater_id"])
        if len(oof) != len(user):
            raise RuntimeError(f"OOF {run_name} has {len(oof)} rows, expected {len(user)}")
        path = out_dir / (
            "oof_predictions.csv" if run_name == "weighted" else "oof_predictions_unweighted.csv"
        )
        oof.to_csv(path, index=False)

        # Export clean OOF prediction file for the report (auto-refresh on each run).
        export_name = (
            "oof_D_predictions.csv" if run_name == "weighted" else "oof_D_predictions_unweighted.csv"
        )
        export_d_oof(path, out_dir / export_name)
        pooled = binary_metrics_with_calibration(
            oof["error-rating"].to_numpy(),
            oof["d_probability"].to_numpy(),
            threshold=0.5,
        )
        append_metrics_row(
            out_dir / f"metrics_pooled_{run_name}.csv",
            {"split": "oof@0.5", "run": run_name, **pooled},
        )
        # Also at mean Youden from folds if available
        _log(
            f"Saved {path} ({len(oof)} rows) "
            f"AUROC@0.5={pooled['auroc']:.4f} AUPRC={pooled['auprc']:.4f}"
        )

    meta = {
        "protocol": "nested",
        "note": (
            "Outer-test p_i reused from diagnostic OOF (nested_test). "
            "Outer-train p_i from inner CV OOF (nested_oof)."
        ),
        "inner_n_splits": int(cfg["solution_d"].get("inner_n_splits", 4)),
    }
    (out_dir / "train_D_nested_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_fast(cfg: dict, base_dir: Path, *, only_fold: Optional[int] = None) -> None:
    out_dir = project_path(cfg["outputs"]["d_dir"], base_dir)
    model_dir = project_path(cfg["models"]["d_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["diagnostic"].get("device", "auto"))
    image_dir = project_path(cfg["data"]["image_dir"], base_dir)
    _log(f"Device: {device}")
    _log(f"D allowlist: {d_feature_allowlist(cfg)}")

    folds = load_fold_assignment(
        project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    )
    cases = load_diagnostic_case_table(cfg, base_dir=base_dir)
    cases = cases.merge(folds[["case_id", "fold"]], on="case_id", how="inner", validate="one_to_one")
    user = _strip_target(load_user_infos(cfg, base_dir=base_dir))
    oof_diagnostic = load_oof_predictions(
        project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    )

    n_splits = int(cfg["validation"]["n_splits"])
    fold_range = [only_fold] if only_fold is not None else list(range(1, n_splits + 1))

    # Idempotent per-run tables for the fast protocol (parallel to nested).
    if only_fold is None:
        for fname in (
            "scale_pos_weight_per_fold_fast.csv",
            "metrics_per_fold_weighted_fast.csv",
            "metrics_per_fold_unweighted_fast.csv",
            "metrics_pooled_weighted_fast.csv",
            "metrics_pooled_unweighted_fast.csv",
        ):
            reset_metrics_file(out_dir / fname)

    collected: Dict[str, List[pd.DataFrame]] = {}
    fold_pbar = tqdm(
        fold_range,
        desc="D fast folds",
        unit="fold",
    )
    for outer_fold in fold_pbar:
        fold_pbar.set_postfix({"fold": outer_fold})
        _log(f"=== Fast protocol: outer fold {outer_fold}/{n_splits} ===")
        fold_results = train_d_one_outer_fold(
            outer_fold=outer_fold,
            user=user,
            cases=cases,
            folds=folds,
            oof_diagnostic=oof_diagnostic,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
            out_dir=out_dir,
            model_dir=model_dir,
            protocol="fast",
        )
        for name, df in fold_results.items():
            collected.setdefault(name, []).append(df)
        tqdm.write(f"  [fast] outer={outer_fold} D fold {outer_fold} done")

    if only_fold is not None:
        _log(f"Single-fast-fold run done (fold={only_fold}). Re-run without --fold to pool OOF.")
        return

    for run_name, parts in collected.items():
        if not parts:
            continue
        oof = pd.concat(parts, ignore_index=True).sort_values(["case_id", "rater_id"])
        if len(oof) != len(user):
            raise RuntimeError(f"Fast OOF {run_name} has {len(oof)} rows, expected {len(user)}")
        path = out_dir / (
            "oof_predictions_fast.csv" if run_name == "weighted" else "oof_predictions_unweighted_fast.csv"
        )
        oof.to_csv(path, index=False)

        # Export clean OOF prediction file for the report (auto-refresh on each run).
        export_name = (
            "oof_D_predictions_fast.csv" if run_name == "weighted" else "oof_D_predictions_unweighted_fast.csv"
        )
        export_d_oof(path, out_dir / export_name)
        pooled = binary_metrics_with_calibration(
            oof["error-rating"].to_numpy(),
            oof["d_probability"].to_numpy(),
            threshold=0.5,
        )
        append_metrics_row(
            out_dir / f"metrics_pooled_{run_name}_fast.csv",
            {"split": "oof@0.5", "run": run_name, **pooled},
        )
        _log(
            f"  Saved {path} ({len(oof)} rows) "
            f"AUROC@0.5={pooled['auroc']:.4f} AUPRC={pooled['auprc']:.4f}"
        )

    meta = {
        "protocol": "fast",
        "note": (
            "Fast (optimistic, declared): outer-train probabilities are IN-SAMPLE from a diagnostic "
            "trained on ALL outer-train (probability_source=fast_insample). "
            "Outer-test probabilities are out-of-sample (probability_source=fast_test). "
            "D features, preprocessor, XGBoost, and threshold selection are the same as nested."
        ),
        "inner_n_splits": int(cfg["solution_d"].get("inner_n_splits", 4)),
    }
    (out_dir / "train_D_fast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train solution D (nested default).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--protocol",
        choices=["nested", "fast"],
        default=None,
        help="Override config solution_d.protocol",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single outer fold (1..n_splits), useful to resume",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    protocol = args.protocol or str(cfg["solution_d"].get("protocol", "nested"))
    _log(f"Protocol: {protocol}")
    if protocol == "nested":
        run_nested(cfg, base_dir, only_fold=args.fold)
    elif protocol == "fast":
        run_fast(cfg, base_dir, only_fold=args.fold)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


if __name__ == "__main__":
    main()
