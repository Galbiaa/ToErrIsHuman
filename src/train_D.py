"""Train solution D with nested (default) or fast protocol.

Nested (scientific):
  A) inner-CV OOF p_i on outer-train cases
  B) outer-test p_i from diagnostic trained without those test cases
     (reuse Phase-4 OOF test predictions = same protocol)
  C) fold-safe D features + preprocessor + XGBoost; threshold on inner val cases

Fast is available via --protocol fast (optimistic; not the main analysis).
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
from generate_folds import load_fold_assignment
from metrics import append_metrics_row, binary_metrics_with_calibration
from methodology_guards import assert_case_disjoint, select_threshold_on_validation
from reproducibility import bootstrap_run_reproducibility, set_global_seed, xgb_seed_for_fold
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


def outer_test_probabilities_from_phase4(
    oof_phase4: pd.DataFrame,
    test_case_ids: List[int],
    *,
    outer_fold: int,
) -> pd.DataFrame:
    """B) Reuse Phase-4 OOF for outer-test cases (model never saw those cases)."""
    test_set = set(int(x) for x in test_case_ids)
    sub = oof_phase4.loc[oof_phase4["case_id"].isin(test_set)].copy()
    if len(sub) != len(test_set):
        raise RuntimeError(
            f"Phase-4 OOF missing test cases for outer fold {outer_fold}: "
            f"need {len(test_set)} got {len(sub)}"
        )
    sub = sub[["case_id", "ai_probability_class_1"]].copy()
    sub["outer_fold"] = outer_fold
    sub["probability_source"] = "nested_test"
    return sub.reset_index(drop=True)


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
    oof_phase4: pd.DataFrame,
    cfg: dict,
    image_dir: Path,
    device: torch.device,
    out_dir: Path,
    model_dir: Path,
) -> Dict[str, pd.DataFrame]:
    test_cases = folds.loc[folds["fold"] == outer_fold, "case_id"].astype(int).tolist()
    train_cases = folds.loc[folds["fold"] != outer_fold, "case_id"].astype(int).tolist()
    assert_case_disjoint(train_cases, test_cases)

    outer_train_df = cases.loc[cases["case_id"].isin(train_cases)].reset_index(drop=True)
    outer_test_df = cases.loc[cases["case_id"].isin(test_cases)].reset_index(drop=True)
    _log(f"\n=== Outer fold {outer_fold}: train_cases={len(outer_train_df)} test_cases={len(outer_test_df)} ===")

    cache = out_dir / f"nested_inner_oof_fold_{outer_fold}.csv"
    inner_oof = compute_inner_oof_probabilities(
        outer_train_cases=outer_train_df,
        cfg=cfg,
        image_dir=image_dir,
        device=device,
        outer_fold=outer_fold,
        cache_path=cache,
    )
    test_probs = outer_test_probabilities_from_phase4(
        oof_phase4, test_cases, outer_fold=outer_fold
    )

    # C) Features — separate frames for provenance tags, same train_case_ids
    _log("  [C] Building fold-safe D features")
    train_feat = build_d_feature_frame(
        user,
        inner_oof.rename(columns={})[["case_id", "ai_probability_class_1"]],
        train_cases,
        config=cfg,
        probability_source="nested_oof",
        fold_col=None,
    )
    train_feat["fold"] = outer_fold
    train_feat["probability_source"] = "nested_oof"

    test_feat = build_d_feature_frame(
        user,
        test_probs[["case_id", "ai_probability_class_1"]],
        train_cases,
        config=cfg,
        probability_source="nested_test",
        fold_col=None,
    )
    test_feat["fold"] = outer_fold
    test_feat["probability_source"] = "nested_test"

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

    runs = []
    if bool(cfg["solution_d"].get("use_scale_pos_weight", True)):
        n_pos = max(int((y_all_train == 1).sum()), 1)
        n_neg = int((y_all_train == 0).sum())
        runs.append(("weighted", float(n_neg) / float(n_pos)))
    if bool(cfg["solution_d"].get("also_run_without_weighting", True)):
        runs.append(("unweighted", None))
    if not runs:
        runs.append(("weighted", None))

    results: Dict[str, pd.DataFrame] = {}
    for run_name, spw in runs:
        _log(f"  [C] XGBoost {run_name} (scale_pos_weight={spw})")
        # Model for threshold: fit on fit_cases only
        pre_thr = make_d_preprocessor()
        X_fit_t, X_thr_t, pre_thr = fit_transform_d_features(pre_thr, X_fit, X_thr)
        clf_thr = make_xgb(cfg, fold=outer_fold, scale_pos_weight=spw)
        clf_thr.fit(X_fit_t, y_fit)
        p_thr = clf_thr.predict_proba(X_thr_t)[:, 1]
        thr = select_threshold_on_validation(y_thr, p_thr, method="youden")

        # Final model: preprocessor + XGB on all outer-train
        pre = make_d_preprocessor()
        X_tr_t, X_te_t, pre = fit_transform_d_features(pre, X_all_train, X_test)
        clf = make_xgb(cfg, fold=outer_fold, scale_pos_weight=spw)
        clf.fit(X_tr_t, y_all_train)
        p_test = clf.predict_proba(X_te_t)[:, 1]

        suffix = "" if run_name == "weighted" else "_unweighted"
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
                "protocol": "nested",
            },
            model_path,
        )
        _log(f"      saved {model_path.name} thr={thr:.4f}")

        metrics = binary_metrics_with_calibration(y_test, p_test, threshold=float(thr))
        append_metrics_row(
            out_dir / f"metrics_per_fold_{run_name}.csv",
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
    oof_phase4 = load_oof_predictions(
        project_path(cfg["outputs"]["diagnostic_dir"], base_dir) / "oof_predictions.csv"
    )

    n_splits = int(cfg["validation"]["n_splits"])
    fold_range = [only_fold] if only_fold is not None else list(range(1, n_splits + 1))

    collected: Dict[str, List[pd.DataFrame]] = {"weighted": [], "unweighted": []}
    for fold in fold_range:
        fold_results = train_d_one_outer_fold(
            outer_fold=fold,
            user=user,
            cases=cases,
            folds=folds,
            oof_phase4=oof_phase4,
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
            "Outer-test p_i reused from Phase-4 OOF (nested_test). "
            "Outer-train p_i from inner CV OOF (nested_oof)."
        ),
        "inner_n_splits": int(cfg["solution_d"].get("inner_n_splits", 4)),
    }
    (out_dir / "train_D_nested_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_fast(cfg: dict, base_dir: Path) -> None:
    raise NotImplementedError(
        "Fast protocol is optional; this run requested nested. "
        "Re-run with --protocol fast when you explicitly want the optimistic replica."
    )


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
        run_fast(cfg, base_dir)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


if __name__ == "__main__":
    main()
