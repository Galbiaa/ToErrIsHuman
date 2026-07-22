"""Data loading for the two-stage pipeline.

TARGET may be used as the diagnostic label and for join checks,
but must never appear in feature matrices for solution D.
Excel rater-accuracy / rater-confidence are global aggregates and must
not be used in CV (replaced later by fold-safe LOO).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

USER_REQUIRED_COLUMNS = [
    "id",
    "rating-confidence",
    "case-difficulty",
    "rater-expertise",
    "rater-accuracy",
    "rater-confidence",
    "rating",
    "error-rating",
]

# Columns that come from Excel as global rater aggregates (unsafe in CV).
EXCEL_GLOBAL_RATER_STATS = ("rater-accuracy", "rater-confidence")


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


def load_ground_truth(config: dict, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Load IMAGE-GROUND-TRUTH: one row per case with CASE-ID and TARGET."""
    data_cfg = config["data"]
    path = project_path(data_cfg["ground_truth_file"], base_dir)
    df = pd.read_excel(path)

    case_col = data_cfg.get("case_id_column", "CASE-ID")
    target_col = data_cfg.get("target_diagnostic", data_cfg.get("target_diagnostic", "TARGET"))
    missing = [c for c in (case_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = pd.DataFrame(
        {
            "case_id": pd.to_numeric(df[case_col], errors="raise").astype(int),
            "TARGET": pd.to_numeric(df[target_col], errors="raise").astype(int),
        }
    )
    if out["case_id"].duplicated().any():
        dups = out.loc[out["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate CASE-ID values in ground truth: {dups[:20]}")
    if not set(out["TARGET"].unique()).issubset({0, 1}):
        raise ValueError(f"TARGET must be binary 0/1, got {sorted(out['TARGET'].unique())}")
    return out.sort_values("case_id").reset_index(drop=True)


def load_user_infos(config: dict, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Load USER-INFOS-CORRECTED: one row per case–rater decision."""
    data_cfg = config["data"]
    path = project_path(data_cfg["user_infos_file"], base_dir)
    df = pd.read_excel(path)

    missing = [c for c in USER_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    sep = data_cfg.get("case_id_separator", "-")
    id_col = data_cfg.get("id_column", "id")
    parsed = df[id_col].apply(lambda x: parse_decision_id(x, sep=sep))

    out = df.copy()
    out["case_id"] = parsed.apply(lambda x: x.case_id)
    out["rater_id"] = parsed.apply(lambda x: x.rater_id)

    numeric_cols = [
        "rating-confidence",
        "case-difficulty",
        "rater-expertise",
        "rater-accuracy",
        "rater-confidence",
        "rating",
        "error-rating",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")

    # Mark Excel globals as unsafe for CV.
    out.attrs["excel_global_rater_stats"] = list(
        data_cfg.get("excel_global_rater_stats", EXCEL_GLOBAL_RATER_STATS)
    )
    out.attrs["excel_global_rater_stats_usable_in_cv"] = False
    return out


def load_diagnostic_case_table(config: dict, base_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Case-level table for the diagnostic model: one row per case, no 13× replication.
    Columns: case_id, axial_path, coronal_path, sagittal_path, target
    """
    gt = load_ground_truth(config, base_dir=base_dir)
    image_dir = project_path(config["data"]["image_dir"], base_dir)
    orientations = list(config["images"]["orientations"])
    extension = config["images"].get("extension", "jpg")

    rows = []
    for _, row in gt.iterrows():
        case_id = int(row["case_id"])
        paths = get_image_paths(case_id, image_dir, orientations, extension)
        path_map = {ori: str(p) for ori, p in zip(orientations, paths)}
        rows.append(
            {
                "case_id": case_id,
                "axial_path": path_map.get("axial"),
                "coronal_path": path_map.get("coronal"),
                "sagittal_path": path_map.get("sagittal"),
                "target": int(row["TARGET"]),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def features_for_solution_d(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Return a copy safe for solution D feature construction.
    Raises if TARGET (or other forbidden columns) are still present.
    """
    from methodology_guards import assert_d_features

    forbidden = list(config.get("solution_d", {}).get("forbidden_features", ["TARGET"]))
    assert_d_features(df.columns, forbidden=forbidden)
    return df.copy()


def get_image_paths(
    case_id: int, image_dir: str | Path, orientations: Iterable[str], extension: str = "jpg"
) -> List[Path]:
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


def list_image_case_ids(image_dir: str | Path, orientations: Sequence[str], extension: str = "jpg") -> List[int]:
    """Parse case IDs present in the image directory from filenames."""
    image_dir = Path(image_dir)
    found: set[int] = set()
    for path in image_dir.glob(f"case*.{extension}"):
        name = path.stem  # e.g. case12axial
        if not name.startswith("case"):
            continue
        rest = name[4:]
        for ori in orientations:
            if rest.endswith(ori):
                case_str = rest[: -len(ori)]
                if case_str.isdigit():
                    found.add(int(case_str))
                break
    return sorted(found)


def pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() can leave the file in a bad state; reopen for a light load
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


class TripleImageMixin:
    def _load_triple(self, case_id: int) -> torch.Tensor:
        tensors = []
        paths = get_image_paths(case_id, self.image_dir, self.orientations, self.extension)
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing image file: {path}")
            img = pil_loader(path)
            tensors.append(self.transform(img))
        return torch.stack(tensors, dim=0)  # [3, C, H, W]


class DiagnosticDataset(Dataset, TripleImageMixin):
    """Case-level dataset for the diagnostic model (label = TARGET)."""

    def __init__(
        self,
        case_df: pd.DataFrame,
        image_dir: str | Path,
        orientations: List[str],
        extension: str,
        transform,
        target_col: str = "target",
    ):
        self.df = case_df.reset_index(drop=True).copy()
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
        x_img = self._load_triple(case_id)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return {
            "images": x_img,
            "target": y,
            "case_id": torch.tensor(case_id, dtype=torch.long),
        }
