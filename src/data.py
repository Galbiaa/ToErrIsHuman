from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ParsedId:
    case_id: int
    rater_id: int


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / p


def parse_decision_id(value: str, sep: str = "-") -> ParsedId:
    try:
        case_str, rater_str = str(value).split(sep)
        return ParsedId(case_id=int(case_str), rater_id=int(rater_str))
    except Exception as exc:
        raise ValueError(f"Cannot parse id '{value}' as <case_id>{sep}<rater_id>.") from exc


def load_training_dataframe(config: dict, base_dir: str | Path | None = None) -> pd.DataFrame:
    data_cfg = config["data"]
    path = project_path(data_cfg["training_file"], base_dir)
    df = pd.read_excel(path)

    required = [
        data_cfg["id_column"],
        *data_cfg["numeric_features"],
        data_cfg["target_column"],
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    sep = data_cfg.get("case_id_separator", "-")
    parsed = df[data_cfg["id_column"]].apply(lambda x: parse_decision_id(x, sep=sep))
    df = df.copy()
    df["case_id"] = parsed.apply(lambda x: x.case_id)
    df["rater_id"] = parsed.apply(lambda x: x.rater_id)

    for col in data_cfg["numeric_features"] + [data_cfg["target_column"]]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def get_image_paths(case_id: int, image_dir: str | Path, orientations: Iterable[str], extension: str = "jpg") -> List[Path]:
    image_dir = Path(image_dir)
    return [image_dir / f"case{case_id}{orientation}.{extension}" for orientation in orientations]


def validate_image_files(
    case_ids: Iterable[int], image_dir: str | Path, orientations: Iterable[str], extension: str = "jpg"
) -> Tuple[List[Path], List[Path]]:
    expected: List[Path] = []
    missing: List[Path] = []
    for case_id in sorted(set(int(x) for x in case_ids)):
        paths = get_image_paths(case_id, image_dir, orientations, extension)
        expected.extend(paths)
        missing.extend([p for p in paths if not p.exists()])
    return expected, missing


def resolve_folds_path(config: dict, base_dir: str | Path | None = None) -> Path:
    folds_rel = config.get("validation", {}).get("folds_file", "outputs/folds/fold_assignments.csv")
    return project_path(folds_rel, base_dir=base_dir)


def build_fold_assignments(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Assign each decision to a test fold using GroupKFold on case_id."""
    from sklearn.model_selection import GroupKFold

    groups = df["case_id"].to_numpy(dtype=int)
    case_to_fold: Dict[int, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    x_dummy = np.zeros(len(df))
    y_dummy = np.zeros(len(df))

    for fold, (_, test_idx) in enumerate(splitter.split(x_dummy, y_dummy, groups=groups), start=1):
        for idx in test_idx:
            case_to_fold[int(df.iloc[idx]["case_id"])] = fold

    assignments = df[["case_id", "rater_id"]].copy()
    assignments["fold"] = assignments["case_id"].map(case_to_fold).astype(int)
    return assignments.reset_index(drop=True)


def compute_fold_stats(df: pd.DataFrame, fold_assignments: pd.DataFrame, target_col: str) -> pd.DataFrame:
    merged = attach_fold_column(df, fold_assignments)
    rows = []
    for fold in sorted(merged["fold"].unique()):
        sub = merged[merged["fold"] == fold]
        rows.append(
            {
                "fold": int(fold),
                "n_cases": int(sub["case_id"].nunique()),
                "n_decisions": int(len(sub)),
                "n_errors": int(sub[target_col].sum()),
                "prevalence": float(sub[target_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def load_fold_assignments(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fold assignments not found: {path}")

    df = pd.read_csv(path)
    required = {"case_id", "rater_id", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fold assignments ({path}): {sorted(missing)}")

    df["case_id"] = df["case_id"].astype(int)
    df["rater_id"] = df["rater_id"].astype(int)
    df["fold"] = df["fold"].astype(int)
    return df.reset_index(drop=True)


def attach_fold_column(df: pd.DataFrame, fold_assignments: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["case_id", "rater_id"]
    fold_map = fold_assignments.set_index(key_cols)["fold"]
    out = df.reset_index(drop=True).copy()
    out["fold"] = out.set_index(key_cols).index.map(fold_map)

    if out["fold"].isna().any():
        raise ValueError("Fold assignments do not cover all decision rows in the dataframe.")

    out["fold"] = out["fold"].astype(int)
    return out


def iter_group_fold_splits(
    df: pd.DataFrame, fold_assignments: pd.DataFrame, n_splits: int
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold, train_idx, test_idx) using precomputed case-level fold assignments."""
    df_with_folds = attach_fold_column(df, fold_assignments)
    folds = df_with_folds["fold"].to_numpy(dtype=int)

    for fold in range(1, n_splits + 1):
        test_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Fold {fold} has an empty train or test split.")
        yield fold, train_idx, test_idx


def get_rater_aggregate_specs(config: dict) -> Dict[str, Dict[str, str]]:
    default = {
        "rater-accuracy": {"from_column": "error-rating", "method": "one_minus_mean"},
        "rater-confidence": {"from_column": "rating-confidence", "method": "mean"},
    }
    return config.get("data", {}).get("rater_aggregate_features", default)


def _aggregate_rater_series(values: pd.Series, method: str) -> float:
    if method == "mean":
        return float(values.mean())
    if method == "one_minus_mean":
        return float(1.0 - values.mean())
    raise ValueError(f"Unknown rater aggregate method: {method}")


def compute_rater_aggregates(train_df: pd.DataFrame, specs: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Compute rater-level aggregate features using training-fold decisions only."""
    rows = []
    for rater_id, group in train_df.groupby("rater_id"):
        row: Dict[str, float | int] = {"rater_id": int(rater_id)}
        for out_col, spec in specs.items():
            source_col = spec["from_column"]
            row[out_col] = _aggregate_rater_series(group[source_col], method=str(spec["method"]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("rater_id")


def apply_fold_rater_aggregates(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    specs: Dict[str, Dict[str, str]],
    missing_strategy: str = "train_global_mean",
) -> pd.DataFrame:
    """Replace rater aggregate columns using statistics computed on the training fold."""
    out = df.copy()
    aggregates = compute_rater_aggregates(train_df, specs)

    for out_col in specs:
        if missing_strategy != "train_global_mean":
            raise ValueError(f"Unsupported missing rater aggregate strategy: {missing_strategy}")

        fallback = float(aggregates[out_col].mean()) if len(aggregates) else 0.0
        mapped = out["rater_id"].map(aggregates[out_col]).astype(float)
        if mapped.isna().any():
            missing_raters = sorted(out.loc[mapped.isna(), "rater_id"].unique().tolist())
            print(
                f"  Warning: rater(s) {missing_raters} absent from training fold; "
                f"using train-global mean for '{out_col}'."
            )
            mapped = mapped.fillna(fallback)
        out[out_col] = mapped.to_numpy(dtype=float)

    return out


def prepare_fold_feature_frames(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test frames with leakage-safe rater aggregates for the current fold."""
    specs = get_rater_aggregate_specs(config)
    missing_strategy = config.get("data", {}).get("missing_rater_aggregate_strategy", "train_global_mean")
    train_raw = df.iloc[train_idx].copy()
    test_raw = df.iloc[test_idx].copy()
    train_df = apply_fold_rater_aggregates(train_raw, train_raw, specs, missing_strategy=missing_strategy)
    test_df = apply_fold_rater_aggregates(test_raw, train_raw, specs, missing_strategy=missing_strategy)
    return train_df, test_df


def make_case_level_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Aggregate decision-level rows into one row per case for image-only training."""
    grouped = (
        df.groupby("case_id", as_index=False)
        .agg(
            mean_error=(target_col, "mean"),
            n_ratings=(target_col, "size"),
            n_errors=(target_col, "sum"),
        )
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    return grouped


def pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class TripleImageMixin:
    """Load three orientations per case, with an in-memory cache keyed by case_id."""

    def _init_image_cache(self) -> None:
        if not hasattr(self, "_image_cache"):
            self._image_cache: Dict[int, torch.Tensor] = {}

    def _load_triple(self, case_id: int) -> torch.Tensor:
        tensors = []
        paths = get_image_paths(case_id, self.image_dir, self.orientations, self.extension)
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing image file: {path}")
            img = pil_loader(path)
            tensors.append(self.transform(img))
        return torch.stack(tensors, dim=0)  # [3, C, H, W]

    def _get_cached_triple(self, case_id: int) -> torch.Tensor:
        self._init_image_cache()
        cached = self._image_cache.get(case_id)
        if cached is None:
            cached = self._load_triple(case_id)
            self._image_cache[case_id] = cached
        return cached

    def preload_case_images(self, case_ids: Iterable[int]) -> None:
        """Warm the cache for the given case IDs (same tensors as on-demand loading)."""
        for case_id in sorted({int(c) for c in case_ids}):
            self._get_cached_triple(case_id)


class ImageOnlyCaseDataset(Dataset, TripleImageMixin):
    """Case-level dataset for secondary mean-error regression (427 rows)."""

    def __init__(
        self,
        case_df: pd.DataFrame,
        image_dir: str | Path,
        orientations: List[str],
        extension: str,
        transform,
        cache_images: bool = True,
    ):
        self.df = case_df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.orientations = orientations
        self.extension = extension
        self.transform = transform
        self._init_image_cache()
        if cache_images:
            self.preload_case_images(self.df["case_id"].tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = int(row["case_id"])
        x_img = self._get_cached_triple(case_id)
        y = torch.tensor(float(row["mean_error"]), dtype=torch.float32)
        return {
            "images": x_img,
            "target": y,
            "case_id": torch.tensor(case_id, dtype=torch.long),
        }


class ImageOnlyDecisionDataset(Dataset, TripleImageMixin):
    """Decision-level image-only dataset (5551 rows, binary error-rating target)."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        orientations: List[str],
        extension: str,
        transform,
        target_col: str,
        cache_images: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.orientations = orientations
        self.extension = extension
        self.transform = transform
        self.target_col = target_col
        self._init_image_cache()
        if cache_images:
            self.preload_case_images(self.df["case_id"].tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = int(row["case_id"])
        rater_id = int(row["rater_id"])
        x_img = self._get_cached_triple(case_id)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return {
            "images": x_img,
            "target": y,
            "case_id": torch.tensor(case_id, dtype=torch.long),
            "rater_id": torch.tensor(rater_id, dtype=torch.long),
        }


class MultimodalDecisionDataset(Dataset, TripleImageMixin):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        orientations: List[str],
        extension: str,
        transform,
        feature_cols: List[str],
        target_col: str,
        cache_images: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.orientations = orientations
        self.extension = extension
        self.transform = transform
        self.feature_cols = feature_cols
        self.target_col = target_col
        self._init_image_cache()
        if cache_images:
            self.preload_case_images(self.df["case_id"].tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = int(row["case_id"])
        rater_id = int(row["rater_id"])
        x_img = self._get_cached_triple(case_id)
        x_tab = torch.tensor(row[self.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return {
            "images": x_img,
            "tabular": x_tab,
            "target": y,
            "case_id": torch.tensor(case_id, dtype=torch.long),
            "rater_id": torch.tensor(rater_id, dtype=torch.long),
        }
