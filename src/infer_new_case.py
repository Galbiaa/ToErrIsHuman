"""Inference on a new (or held-out demo) case: fold-aligned ensemble.

For each fold k:
  diagnostic_fold_k -> p_i
  build D features (fold-safe rater stats from historical train cases)
  preprocessor_fold_k + scenario_D_fold_k -> P(error)
Average the five D probabilities. Components of the same fold stay paired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_D_features import build_d_feature_frame, d_feature_allowlist, extract_xy
from data import load_config, load_diagnostic_case_table, load_user_infos, project_path
from generate_folds import load_fold_assignment
from models import DiagnosticNet
from reproducibility import set_global_seed
from train_diagnostic import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_device,
    logits_to_prob,
    make_loader,
    make_transforms,
    predict_logits,
)


def _strip_target(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c == "TARGET" or str(c).upper() == "TARGET"]
    return df.drop(columns=drop) if drop else df


def load_diagnostic_from_checkpoint(path: Path, device: torch.device) -> DiagnosticNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", {})
    model = DiagnosticNet(
        pretrained=False,
        freeze_backbone=False,
        aggregation=arch.get("aggregation", "concat"),
        hidden_dim=int(arch.get("hidden_dim", 256)),
        dropout=float(ckpt.get("dropout", arch.get("dropout", 0.35))),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_case_probability(
    model: DiagnosticNet,
    case_row: pd.DataFrame,
    cfg: dict,
    image_dir: Path,
    device: torch.device,
) -> float:
    loader = make_loader(
        case_row,
        image_dir=image_dir,
        orientations=list(cfg["images"]["orientations"]),
        extension=cfg["images"].get("extension", "jpg"),
        transform=make_transforms(cfg, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    logits, _, _ = predict_logits(model, loader, device)
    return float(logits_to_prob(logits)[0])


def ensemble_infer_case(
    *,
    case_id: int,
    cfg: dict,
    base_dir: Path,
    rater_id: Optional[int] = None,
    train_case_ids: Optional[List[int]] = None,
) -> dict:
    device = get_device(cfg["diagnostic"].get("device", "auto"))
    image_dir = project_path(cfg["data"]["image_dir"], base_dir)
    model_diag_dir = project_path(cfg["models"]["diagnostic_dir"], base_dir)
    model_d_dir = project_path(cfg["models"]["d_dir"], base_dir)

    cases = load_diagnostic_case_table(cfg, base_dir=base_dir)
    case_row = cases.loc[cases["case_id"] == case_id].reset_index(drop=True)
    if len(case_row) != 1:
        raise ValueError(f"case_id={case_id} not found or duplicated in case table")

    user = _strip_target(load_user_infos(cfg, base_dir=base_dir))
    folds = load_fold_assignment(
        project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    )
    all_cases = folds["case_id"].astype(int).tolist()
    if train_case_ids is None:
        # Historical pool for rater stats: all cases except the queried one
        train_case_ids = [c for c in all_cases if c != int(case_id)]

    allow = d_feature_allowlist(cfg)
    n_splits = int(cfg["validation"]["n_splits"])
    per_fold = []
    d_probs = []

    for fold in range(1, n_splits + 1):
        diag_path = model_diag_dir / f"diagnostic_fold_{fold}.pt"
        pre_path = model_d_dir / f"preprocessor_fold_{fold}.joblib"
        d_path = model_d_dir / f"scenario_D_fold_{fold}.joblib"
        for p in (diag_path, pre_path, d_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        model = load_diagnostic_from_checkpoint(diag_path, device)
        p_i = predict_case_probability(model, case_row, cfg, image_dir, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        probs = pd.DataFrame(
            {"case_id": [int(case_id)], "ai_probability_class_1": [p_i]}
        )
        feat = build_d_feature_frame(
            user,
            probs,
            train_case_ids,
            config=cfg,
            probability_source="nested_test",
            fold_col=None,
        )
        feat = feat.loc[feat["case_id"] == int(case_id)].copy()
        if rater_id is not None:
            feat = feat.loc[feat["rater_id"] == int(rater_id)]
        if feat.empty:
            raise RuntimeError("No decision rows for this case/rater after feature build")

        X, _, names = extract_xy(feat, cfg)
        assert list(names) == allow
        pre = joblib.load(pre_path)
        payload = joblib.load(d_path)
        clf = payload["model"]
        thr = float(payload.get("threshold_youden", 0.5))
        X_t = pre.transform(np.asarray(X, dtype=float))
        p_err = clf.predict_proba(X_t)[:, 1]

        fold_rows = feat[["id", "rater_id", "rating", "error-rating"]].copy()
        fold_rows["fold"] = fold
        fold_rows["ai_probability_class_1"] = p_i
        fold_rows["d_probability"] = p_err
        fold_rows["threshold_youden"] = thr
        per_fold.append(fold_rows)
        d_probs.append(p_err)

    stacked = pd.concat(per_fold, ignore_index=True)
    # Ensemble: mean D probability across folds, per decision id
    ens = (
        stacked.groupby(["id", "rater_id"], as_index=False)
        .agg(
            d_probability_ensemble=("d_probability", "mean"),
            ai_probability_mean=("ai_probability_class_1", "mean"),
            rating=("rating", "first"),
            error_rating=("error-rating", "first"),
        )
    )
    return {
        "case_id": int(case_id),
        "n_folds": n_splits,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "per_fold_rows": len(stacked),
        "ensemble_decisions": ens.to_dict(orient="records"),
        "mean_d_probability_over_decisions": float(ens["d_probability_ensemble"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-aligned ensemble inference for one case."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--rater-id", type=int, default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path (default outputs/analysis/infer_case_{id}.json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)
    set_global_seed(int(cfg.get("reproducibility", {}).get("seed", 42)))

    result = ensemble_infer_case(
        case_id=int(args.case_id),
        cfg=cfg,
        base_dir=base_dir,
        rater_id=args.rater_id,
    )
    out = (
        Path(args.out)
        if args.out
        else project_path(cfg["outputs"]["analysis_dir"], base_dir)
        / f"infer_case_{args.case_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop huge if any; ens records are fine
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"case_id={result['case_id']}")
    print(f"mean_d_probability_over_decisions={result['mean_d_probability_over_decisions']:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
