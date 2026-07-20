from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import (
    ImageOnlyDataset,
    MultimodalDecisionDataset,
    attach_fold_column,
    iter_group_fold_splits,
    load_config,
    load_fold_assignments,
    load_training_dataframe,
    make_case_level_dataframe,
    prepare_fold_feature_frames,
    project_path,
    resolve_folds_path,
    validate_image_files,
)
from experiment_output import (
    build_oof_dataframe,
    evaluate_binary_fold,
    get_classification_threshold,
    init_experiment_directory,
    save_config_snapshot,
    save_oof_predictions,
    write_metrics_summary,
    write_run_metadata,
)
from metrics import append_metrics_row, soft_target_metrics
from models import ImageOnlyNet, MultimodalNet


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "image_only":
        return model(batch["images"])
    if mode == "multimodal":
        return model(batch["images"], batch["tabular"])
    raise ValueError(f"Unknown mode: {mode}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []
    targets: List[float] = []
    probs: List[float] = []
    case_ids: List[int] = []
    rater_ids: List[int] = []

    iterator = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for batch in iterator:
        batch = batch_to_device(batch, device)
        y = batch["target"].float()
        logits = forward_model(model, batch, mode)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()) * y.shape[0])
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(prob.tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
        case_ids.extend(batch["case_id"].detach().cpu().numpy().astype(int).tolist())
        if "rater_id" in batch:
            rater_ids.extend(batch["rater_id"].detach().cpu().numpy().astype(int).tolist())

    n = max(len(targets), 1)
    avg_loss = sum(losses) / n
    meta = {"case_id": np.asarray(case_ids, dtype=int)}
    if rater_ids:
        meta["rater_id"] = np.asarray(rater_ids, dtype=int)
    return avg_loss, np.asarray(targets, dtype=float), np.asarray(probs, dtype=float), meta


def prepare_multimodal_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: List[str],
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_df, test_df = prepare_fold_feature_frames(df, train_idx, test_idx, config)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def train_one_fold(
    mode: str,
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    base_dir: Path,
    output_root: Path,
    threshold: float,
    oof_frames: list[pd.DataFrame],
) -> None:
    data_cfg = cfg["data"]
    img_cfg = cfg["images"]
    train_cfg = cfg["training"]

    image_dir = project_path(data_cfg["image_dir"], base_dir=base_dir)
    transform = make_transforms(int(img_cfg["image_size"]))
    device = get_device(str(train_cfg.get("device", "auto")))

    print(f"Fold {fold}: using device {device}")

    if mode == "image_only":
        train_ds = ImageOnlyDataset(
            train_df,
            image_dir=image_dir,
            orientations=img_cfg["orientations"],
            extension=img_cfg.get("extension", "jpg"),
            transform=transform,
        )
        test_ds = ImageOnlyDataset(
            test_df,
            image_dir=image_dir,
            orientations=img_cfg["orientations"],
            extension=img_cfg.get("extension", "jpg"),
            transform=transform,
        )
        model = ImageOnlyNet(
            pretrained=bool(img_cfg.get("pretrained", True)),
            freeze_backbone=bool(img_cfg.get("freeze_backbone", True)),
            aggregation=img_cfg.get("embedding_aggregation", "concat"),
        )
        criterion = nn.BCEWithLogitsLoss()
    elif mode == "multimodal":
        feature_cols = data_cfg["numeric_features"]
        target_col = data_cfg["target_column"]
        train_ds = MultimodalDecisionDataset(
            train_df,
            image_dir=image_dir,
            orientations=img_cfg["orientations"],
            extension=img_cfg.get("extension", "jpg"),
            transform=transform,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        test_ds = MultimodalDecisionDataset(
            test_df,
            image_dir=image_dir,
            orientations=img_cfg["orientations"],
            extension=img_cfg.get("extension", "jpg"),
            transform=transform,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        model = MultimodalNet(
            n_tabular_features=len(feature_cols),
            pretrained=bool(img_cfg.get("pretrained", True)),
            freeze_backbone=bool(img_cfg.get("freeze_backbone", True)),
            aggregation=img_cfg.get("embedding_aggregation", "concat"),
        )
        y_train = train_df[target_col].to_numpy(dtype=float)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError("mode must be either 'image_only' or 'multimodal'")

    model = model.to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    model_dir = output_root / "models"
    pred_dir = output_root / "predictions"
    plot_dir = output_root / "plots"
    for p in (model_dir, pred_dir, plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_state = None
    patience = int(train_cfg.get("early_stopping_patience", 8))
    remaining_patience = patience
    history_rows = []

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, mode, optimizer=optimizer)
        val_loss, y_val, p_val, _ = run_epoch(model, test_loader, criterion, device, mode, optimizer=None)
        history_rows.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.4f} | validation_loss={val_loss:.4f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_dir / f"{mode}_fold{fold}.pt")
    pd.DataFrame(history_rows).to_csv(output_root / f"history_fold{fold}.csv", index=False)

    val_loss, y_test, p_test, meta = run_epoch(model, test_loader, criterion, device, mode, optimizer=None)

    plot_prefix = plot_dir / f"{mode}_fold{fold}"
    if mode == "image_only":
        metrics = soft_target_metrics(y_test, p_test)
        metrics["validation_loss"] = val_loss
        pred_df = pd.DataFrame(
            {
                "case_id": meta["case_id"],
                "target_mean_error": y_test,
                "predicted_probability_mean_error": p_test,
                "fold": fold,
            }
        )
        pred_df.to_csv(pred_dir / f"{mode}_fold{fold}_predictions.csv", index=False)
    else:
        metrics = evaluate_binary_fold(y_test.astype(int), p_test, plot_prefix=plot_prefix, threshold=threshold)
        metrics["validation_loss"] = val_loss
        oof_df = build_oof_dataframe(
            case_id=meta["case_id"],
            rater_id=meta["rater_id"],
            fold=fold,
            y_true=y_test.astype(int),
            y_probability=p_test,
            threshold=threshold,
        )
        oof_frames.append(oof_df)
        save_oof_predictions(oof_df, pred_dir / f"{mode}_fold{fold}_predictions.csv")

    append_metrics_row(output_root / "metrics.csv", {"fold": fold, "model": mode, **metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["image_only", "multimodal"], required=True)
    args = parser.parse_args()

    base_dir = Path(args.config).resolve().parent
    cfg = load_config(args.config)
    df = load_training_dataframe(cfg, base_dir=base_dir).reset_index(drop=True)
    n_splits = int(cfg["validation"]["n_splits"])
    fold_assignments = load_fold_assignments(resolve_folds_path(cfg, base_dir=base_dir))
    attach_fold_column(df, fold_assignments)

    output_root = project_path(cfg["outputs"]["root_dir"], base_dir=base_dir) / args.mode
    paths = init_experiment_directory(output_root)
    save_config_snapshot(args.config, output_root)
    write_run_metadata(output_root, cfg, experiment_name=args.mode)
    threshold = get_classification_threshold(cfg)

    fold_stats_path = project_path("outputs/folds/fold_stats.csv", base_dir=base_dir)
    if fold_stats_path.exists():
        import shutil

        shutil.copy2(fold_stats_path, output_root / "fold_stats.csv")

    metrics_path = output_root / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    oof_frames: list[pd.DataFrame] = []

    image_dir = project_path(cfg["data"]["image_dir"], base_dir=base_dir)
    _, missing = validate_image_files(
        case_ids=df["case_id"].unique(),
        image_dir=image_dir,
        orientations=cfg["images"]["orientations"],
        extension=cfg["images"].get("extension", "jpg"),
    )
    if missing:
        print(f"Missing {len(missing)} image files. First missing files:")
        for p in missing[:20]:
            print(f"  - {p}")
        raise SystemExit(1)

    if args.mode == "image_only":
        case_df = make_case_level_dataframe(df, target_col=cfg["data"]["target_column"])
        case_to_fold = fold_assignments.drop_duplicates("case_id").set_index("case_id")["fold"]
        case_df = case_df.copy()
        case_df["fold"] = case_df["case_id"].map(case_to_fold).astype(int)

        for fold in range(1, n_splits + 1):
            test_mask = case_df["fold"].values == fold
            train_case_df = case_df.loc[~test_mask].drop(columns=["fold"])
            test_case_df = case_df.loc[test_mask].drop(columns=["fold"])
            train_one_fold(
                args.mode, fold, train_case_df, test_case_df, cfg, base_dir, output_root, threshold, oof_frames
            )
    else:
        for fold, train_idx, test_idx in iter_group_fold_splits(df, fold_assignments, n_splits):
            train_df, test_df, scaler = prepare_multimodal_fold_data(
                df, train_idx, test_idx, feature_cols=cfg["data"]["numeric_features"], config=cfg
            )
            paths["models"].mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, paths["models"] / f"tabular_scaler_fold{fold}.joblib")
            train_one_fold(args.mode, fold, train_df, test_df, cfg, base_dir, output_root, threshold, oof_frames)

    if oof_frames:
        pooled_oof = pd.concat(oof_frames, ignore_index=True)
        save_oof_predictions(pooled_oof, paths["oof"] / "predictions.csv")
        metrics_df = pd.read_csv(metrics_path)
        write_metrics_summary(metrics_df, output_root / "metrics_summary.csv", group_cols=["model"])

    print(f"Saved outputs to: {output_root}")
    if args.mode == "multimodal":
        print(f"Classification threshold: {threshold}")


if __name__ == "__main__":
    main()
