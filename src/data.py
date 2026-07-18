from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def make_case_level_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Aggregate decision-level rows into one row per case (analysis / secondary metrics)."""
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
    def _ensure_image_cache(self) -> None:
        if not hasattr(self, "_image_cache"):
            self._image_cache: Dict[int, torch.Tensor] = {}

    def _load_triple(self, case_id: int) -> torch.Tensor:
        self._ensure_image_cache()
        cached = self._image_cache.get(case_id)
        if cached is not None:
            return cached

        tensors = []
        paths = get_image_paths(case_id, self.image_dir, self.orientations, self.extension)
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing image file: {path}")
            img = pil_loader(path)
            tensors.append(self.transform(img))
        stacked = torch.stack(tensors, dim=0)  # [3, C, H, W]
        self._image_cache[case_id] = stacked
        return stacked


class ImageOnlyDataset(Dataset, TripleImageMixin):
    """Decision-level image-only dataset: one row per rater-case, binary error target.

    Rows that share a case_id reuse the same three images (logical 13x replication).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        orientations: List[str],
        extension: str,
        transform,
        target_col: str,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.orientations = orientations
        self.extension = extension
        self.transform = transform
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = int(row["case_id"])
        rater_id = int(row["rater_id"])
        x_img = self._load_triple(case_id)
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
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.orientations = orientations
        self.extension = extension
        self.transform = transform
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = int(row["case_id"])
        rater_id = int(row["rater_id"])
        x_img = self._load_triple(case_id)
        x_tab = torch.tensor(row[self.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return {
            "images": x_img,
            "tabular": x_tab,
            "target": y,
            "case_id": torch.tensor(case_id, dtype=torch.long),
            "rater_id": torch.tensor(rater_id, dtype=torch.long),
        }
