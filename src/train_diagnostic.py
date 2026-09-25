"""Train the diagnostic model (outer CV, case-level).

Does not start training unless you run this script. Model selection uses
validation AUROC only; the test fold is scored once per outer fold.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import (
    DiagnosticDataset,
    load_config,
    load_diagnostic_case_table,
    project_path,
)
from generate_folds import load_fold_assignment
from metrics import (
    CALIBRATION_INTERPRETATION_NOTE,
    append_metrics_row,
    binary_metrics_with_calibration,
    binary_metrics_with_rowwise_threshold,
    reset_metrics_file,
    save_calibration_plot,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
    write_metrics_table,
)
from methodology_guards import assert_case_disjoint, select_threshold_on_validation
from models import DiagnosticNet
from reproducibility import bootstrap_run_reproducibility, set_global_seed
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device(spec: str = "auto") -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transforms(cfg: dict, *, train: bool) -> transforms.Compose:
    img_cfg = cfg["images"]
    resize = int(img_cfg.get("resize_size", 256))
    size = int(img_cfg.get("image_size", 224))
    rot = float(img_cfg.get("rotation_degrees", 5))
    if train:
        return transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.RandomCrop(size),
                transforms.RandomRotation(degrees=(-rot, rot)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_train_val(
    case_df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified case split inside the outer-train set (by TARGET)."""
    if len(case_df) < 5:
        raise ValueError("Not enough cases to split train/validation")
    y = case_df["target"].to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(case_df["case_id"], y))
    return (
        case_df.iloc[train_idx].reset_index(drop=True),
        case_df.iloc[val_idx].reset_index(drop=True),
    )


def make_loader(
    case_df: pd.DataFrame,
    *,
    image_dir: Path,
    orientations: List[str],
    extension: str,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = DiagnosticDataset(
        case_df,
        image_dir=image_dir,
        orientations=orientations,
        extension=extension,
        transform=transform,
        target_col="target",
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all, case_all = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        logits = model(images)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(batch["target"].numpy())
        case_all.append(batch["case_id"].numpy())
    return (
        np.concatenate(logits_all),
        np.concatenate(y_all).astype(int),
        np.concatenate(case_all).astype(int),
    )


def logits_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def epoch_auroc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import roc_auc_score

    logits, y, _ = predict_logits(model, loader, device)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, logits_to_prob(logits)))


def train_one_fold(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    image_dir: Path,
    device: torch.device,
) -> Tuple[DiagnosticNet, dict]:
    dcfg = cfg["diagnostic"]
    icfg = cfg["images"]
    batch_size = int(dcfg["batch_size"])
    num_workers = int(dcfg.get("num_workers", 0))
    orientations = list(icfg["orientations"])
    extension = icfg.get("extension", "jpg")

    train_loader = make_loader(
        train_df,
        image_dir=image_dir,
        orientations=orientations,
        extension=extension,
        transform=make_transforms(cfg, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_df,
        image_dir=image_dir,
        orientations=orientations,
        extension=extension,
        transform=make_transforms(cfg, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = DiagnosticNet(
        pretrained=bool(icfg.get("pretrained", True)),
        freeze_backbone=True,
        aggregation=icfg.get("embedding_aggregation", "concat"),
        hidden_dim=int(dcfg.get("hidden_dim", 256)),
        dropout=float(dcfg.get("dropout", 0.35)),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(dcfg["learning_rate"]),
        weight_decay=float(dcfg["weight_decay"]),
    )

    max_epochs = int(dcfg["num_epochs"])
    freeze_epochs = int(dcfg.get("freeze_epochs", 2))
    patience = int(dcfg.get("early_stopping_patience", 4))
    unfreeze_layer4 = bool(dcfg.get("unfreeze_layer4_only", True))

    best_state = None
    best_val = -np.inf
    best_epoch = 0
    wait = 0
    history: List[dict] = []

    epoch_pbar = tqdm(
        range(1, max_epochs + 1),
        desc="Epochs",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_pbar:
        if epoch == freeze_epochs + 1 and unfreeze_layer4:
            model.unfreeze_layer4_only()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(dcfg["learning_rate"]),
                weight_decay=float(dcfg["weight_decay"]),
            )

        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            images = batch["images"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(target)
            n_seen += len(target)

        train_loss = running / max(n_seen, 1)
        val_auroc = epoch_auroc(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc})

        epoch_pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "val_auc": f"{val_auroc:.4f}",
            "best": f"{best_val:.4f}",
        })

        improved = np.isfinite(val_auroc) and val_auroc > best_val + 1e-6
        if improved:
            best_val = float(val_auroc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max_epochs
        best_val = float(epoch_auroc(model, val_loader, device))

    model.load_state_dict(best_state)
    meta = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val,
        "history": history,
    }
    return model, meta


def save_checkpoint(
    path: Path,
    *,
    model: DiagnosticNet,
    cfg: dict,
    fold: int,
    seed: int,
    meta: dict,
    val_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": model.architecture_dict(),
        "image_size": int(cfg["images"]["image_size"]),
        "resize_size": int(cfg["images"].get("resize_size", 256)),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "dropout": float(cfg["diagnostic"]["dropout"]),
        "seed": int(seed),
        "fold": int(fold),
        "selected_epoch": int(meta["best_epoch"]),
        "validation_metrics": val_metrics,
        "best_val_auroc": float(meta["best_val_auroc"]),
    }
    torch.save(payload, path)


def export_oof_diagnostic_artifacts(
    out_dir: Path,
    oof: pd.DataFrame,
    fold_thresholds: Dict[int, float],
) -> None:
    """
    Pooled OOF metrics (@0.5 and row-wise validation Youden), confusion matrices, plots.
    Each case uses the Youden threshold from the outer fold where it was held out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {"case_id", "target", "ai_probability_class_1", "fold"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF frame missing columns: {sorted(missing)}")

    oof = oof.sort_values("case_id").reset_index(drop=True)
    thr_map = {int(k): float(v) for k, v in fold_thresholds.items()}
    row_thr = oof["fold"].astype(int).map(thr_map)
    if row_thr.isna().any():
        bad = oof.loc[row_thr.isna(), "fold"].unique().tolist()
        raise ValueError(f"Missing validation Youden threshold for fold(s): {bad}")

    y_oof = oof["target"].to_numpy(dtype=int)
    p_oof = oof["ai_probability_class_1"].to_numpy(dtype=float)
    thr_oof = row_thr.to_numpy(dtype=float)

    pooled_05 = binary_metrics_with_calibration(y_oof, p_oof, threshold=0.5)
    pooled_youden = binary_metrics_with_rowwise_threshold(y_oof, p_oof, thr_oof)
    write_metrics_table(
        out_dir / "metrics_pooled.csv",
        [
            {"split": "oof@0.5", **pooled_05},
            {"split": "oof@youden", **pooled_youden},
        ],
    )

    cm_05 = save_confusion_matrix_plot(
        y_oof,
        p_oof,
        out_dir / "confusion_matrix_oof@0.5.png",
        threshold=0.5,
        title="diagnostic OOF confusion matrix",
    )
    cm_youden = save_confusion_matrix_plot(
        y_oof,
        p_oof,
        out_dir / "confusion_matrix_oof@youden.png",
        thresholds=thr_oof,
        title="diagnostic OOF confusion matrix",
    )
    save_confusion_matrix_csv(
        [
            {
                "split": "oof@0.5",
                "threshold": 0.5,
                "tn": cm_05["tn"],
                "fp": cm_05["fp"],
                "fn": cm_05["fn"],
                "tp": cm_05["tp"],
                "accuracy": pooled_05["accuracy"],
                "balanced_accuracy": pooled_05["balanced_accuracy"],
            },
            {
                "split": "oof@youden",
                "threshold": pooled_youden["threshold"],
                "threshold_note": pooled_youden["threshold_note"],
                "tn": cm_youden["tn"],
                "fp": cm_youden["fp"],
                "fn": cm_youden["fn"],
                "tp": cm_youden["tp"],
                "accuracy": pooled_youden["accuracy"],
                "balanced_accuracy": pooled_youden["balanced_accuracy"],
            },
        ],
        out_dir / "confusion_matrix_oof.csv",
    )

    save_calibration_plot(
        y_oof,
        p_oof,
        out_dir / "calibration_oof.png",
        title="diagnostic OOF calibration",
    )
    save_roc_curve(
        y_oof,
        p_oof,
        out_dir / "roc_oof.png",
        title="diagnostic OOF ROC",
    )
    save_pr_curve(
        y_oof,
        p_oof,
        out_dir / "pr_oof.png",
        title="diagnostic OOF Precision-Recall",
    )
    note_path = out_dir / "calibration_note.txt"
    note_path.write_text(CALIBRATION_INTERPRETATION_NOTE + "\n", encoding="utf-8")

    print(f"Pooled OOF AUROC: {pooled_05['auroc']:.4f}")
    print(f"Pooled OOF accuracy @0.5: {pooled_05['accuracy']:.4f}")
    print(
        f"Pooled OOF accuracy @youden (row-wise, mean thr={pooled_youden['threshold']:.3f}): "
        f"{pooled_youden['accuracy']:.4f}"
    )
    print(
        f"Confusion matrices: {out_dir / 'confusion_matrix_oof@0.5.png'}, "
        f"{out_dir / 'confusion_matrix_oof@youden.png'}"
    )


def load_fold_youden_thresholds(metrics_per_fold_path: Path) -> Dict[int, float]:
    df = pd.read_csv(metrics_per_fold_path)
    if "fold" not in df.columns or "val_threshold_youden" not in df.columns:
        raise ValueError(f"{metrics_per_fold_path} must contain fold and val_threshold_youden")
    return {
        int(row.fold): float(row.val_threshold_youden)
        for row in df.itertuples(index=False)
    }


def run_export_oof_metrics(cfg: dict, base_dir: Path) -> None:
    """Regenerate pooled metrics and confusion-matrix artifacts from saved OOF CSVs."""
    out_dir = project_path(cfg["outputs"]["diagnostic_dir"], base_dir)
    oof_path = out_dir / "oof_predictions.csv"
    fold_metrics_path = out_dir / "metrics_per_fold.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
    if not fold_metrics_path.exists():
        raise FileNotFoundError(f"Missing per-fold metrics: {fold_metrics_path}")
    oof = pd.read_csv(oof_path)
    fold_thresholds = load_fold_youden_thresholds(fold_metrics_path)
    oof["threshold_youden"] = oof["fold"].astype(int).map(fold_thresholds)
    oof.to_csv(oof_path, index=False)
    export_oof_diagnostic_artifacts(out_dir, oof, fold_thresholds)
    print(f"Updated: {out_dir / 'metrics_pooled.csv'}")
    print(f"Updated: {out_dir / 'confusion_matrix_oof.csv'}")


def run_dry_run(cfg: dict, base_dir: Path, folds: pd.DataFrame, cases: pd.DataFrame) -> None:
    """Validate wiring without training."""
    n_splits = int(cfg["validation"]["n_splits"])
    print("DRY-RUN: no weights will be trained or written.")
    print(f"Cases in table: {len(cases)}; folds file cases: {folds['case_id'].nunique()}")
    if "fold" not in cases.columns:
        cases = cases.merge(folds[["case_id", "fold"]], on="case_id", how="inner", validate="one_to_one")
    if len(cases) != folds["case_id"].nunique():
        raise RuntimeError("Mismatch between diagnostic cases and fold assignment")
    for fold in range(1, n_splits + 1):
        test_df = cases.loc[cases["fold"] == fold].reset_index(drop=True)
        train_pool = cases.loc[cases["fold"] != fold].reset_index(drop=True)
        assert_case_disjoint(train_pool["case_id"], test_df["case_id"])
        tr, va = split_train_val(
            train_pool,
            val_fraction=0.2,
            random_state=int(cfg["validation"]["random_state"]) + fold,
        )
        assert_case_disjoint(tr["case_id"], va["case_id"])
        print(
            f"  fold {fold}: test={len(test_df)} train={len(tr)} val={len(va)} "
            f"target_prev test={test_df['target'].mean():.3f}"
        )
    device = get_device(cfg["diagnostic"].get("device", "auto"))
    model = DiagnosticNet(
        pretrained=False,
        freeze_backbone=True,
        aggregation=cfg["images"].get("embedding_aggregation", "concat"),
        hidden_dim=int(cfg["diagnostic"].get("hidden_dim", 256)),
        dropout=float(cfg["diagnostic"].get("dropout", 0.35)),
    )
    print(f"Device: {device}")
    print(f"Architecture: {json.dumps(model.architecture_dict(), indent=2)}")
    print("DRY-RUN complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diagnostic model (outer CV).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--export-oof-metrics",
        action="store_true",
        help="Regenerate pooled OOF metrics and confusion-matrix plots from saved CSVs (no training).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check folds/splits/architecture only; do not train.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg["diagnostic"].get("seed", cfg.get("reproducibility", {}).get("seed", 42)))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    folds_path = project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    folds = load_fold_assignment(folds_path)
    cases = load_diagnostic_case_table(cfg, base_dir=base_dir)
    cases = cases.merge(folds[["case_id", "fold"]], on="case_id", how="inner", validate="one_to_one")

    if args.export_oof_metrics:
        run_export_oof_metrics(cfg, base_dir)
        return

    if args.dry_run:
        run_dry_run(cfg, base_dir, folds, cases)
        return

    device = get_device(cfg["diagnostic"].get("device", "auto"))
    image_dir = project_path(cfg["data"]["image_dir"], base_dir)
    out_dir = project_path(cfg["outputs"]["diagnostic_dir"], base_dir)
    model_dir = project_path(cfg["models"]["diagnostic_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    # Fresh per-run report table (append_metrics_row would otherwise accumulate rows).
    reset_metrics_file(out_dir / "metrics_per_fold.csv")

    n_splits = int(cfg["validation"]["n_splits"])
    oof_rows: List[dict] = []

    fold_pbar = tqdm(
        range(1, n_splits + 1),
        desc="Outer folds",
        unit="fold",
    )
    for fold in fold_pbar:
        fold_pbar.set_postfix({"fold": fold})
        print(f"\n=== Outer fold {fold}/{n_splits} ===")
        test_df = cases.loc[cases["fold"] == fold].reset_index(drop=True)
        train_pool = cases.loc[cases["fold"] != fold].reset_index(drop=True)
        assert_case_disjoint(train_pool["case_id"], test_df["case_id"])

        train_df, val_df = split_train_val(
            train_pool,
            val_fraction=0.2,
            random_state=int(cfg["validation"]["random_state"]) + fold,
        )
        assert_case_disjoint(train_df["case_id"], val_df["case_id"])

        model, meta = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            cfg=cfg,
            image_dir=image_dir,
            device=device,
        )

        val_loader = make_loader(
            val_df,
            image_dir=image_dir,
            orientations=list(cfg["images"]["orientations"]),
            extension=cfg["images"].get("extension", "jpg"),
            transform=make_transforms(cfg, train=False),
            batch_size=int(cfg["diagnostic"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["diagnostic"].get("num_workers", 0)),
        )
        test_loader = make_loader(
            test_df,
            image_dir=image_dir,
            orientations=list(cfg["images"]["orientations"]),
            extension=cfg["images"].get("extension", "jpg"),
            transform=make_transforms(cfg, train=False),
            batch_size=int(cfg["diagnostic"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["diagnostic"].get("num_workers", 0)),
        )

        val_logits, y_val, _ = predict_logits(model, val_loader, device)
        test_logits, y_test, case_test = predict_logits(model, test_loader, device)
        p_val = logits_to_prob(val_logits)
        p_test = logits_to_prob(test_logits)

        thr_val = select_threshold_on_validation(y_val, p_val, method="youden")
        val_metrics = binary_metrics_with_calibration(y_val, p_val, threshold=thr_val)
        test_metrics_05 = binary_metrics_with_calibration(y_test, p_test, threshold=0.5)
        test_metrics_thr = binary_metrics_with_calibration(y_test, p_test, threshold=thr_val)

        ckpt_path = model_dir / f"diagnostic_fold_{fold}.pt"
        save_checkpoint(
            ckpt_path,
            model=model,
            cfg=cfg,
            fold=fold,
            seed=seed,
            meta=meta,
            val_metrics=val_metrics,
        )

        for cid, yt, pt in zip(case_test, y_test, p_test):
            oof_rows.append(
                {
                    "case_id": int(cid),
                    "target": int(yt),
                    "ai_probability_class_1": float(pt),
                    "fold": int(fold),
                    "threshold_youden": float(thr_val),
                }
            )

        fold_row = {
                "fold": fold,
                "best_epoch": meta["best_epoch"],
                "best_val_auroc": meta["best_val_auroc"],
                "val_threshold_youden": thr_val,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test05_{k}": v for k, v in test_metrics_05.items()},
                **{f"testThr_{k}": v for k, v in test_metrics_thr.items()},
            }
        append_metrics_row(out_dir / "metrics_per_fold.csv", fold_row)
        tqdm.write(
            f"fold {fold}: best_epoch={meta['best_epoch']} "
            f"val_auroc={meta['best_val_auroc']:.4f} test_auroc@0.5={test_metrics_05['auroc']:.4f}"
        )

    oof = pd.DataFrame(oof_rows).sort_values("case_id").reset_index(drop=True)
    if len(oof) != len(cases) or oof["case_id"].duplicated().any():
        raise RuntimeError("OOF predictions must cover each case exactly once")
    oof_path = out_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    fold_thresholds = {
        int(row["fold"]): float(row["val_threshold_youden"])
        for row in pd.read_csv(out_dir / "metrics_per_fold.csv").to_dict("records")
    }
    export_oof_diagnostic_artifacts(out_dir, oof, fold_thresholds)

    print(f"\nSaved OOF predictions: {oof_path}")
    print(f"Plots: {out_dir / 'roc_oof.png'}, {out_dir / 'pr_oof.png'}, {out_dir / 'calibration_oof.png'}")
    print(f"Checkpoints under: {model_dir}")


if __name__ == "__main__":
    main()
