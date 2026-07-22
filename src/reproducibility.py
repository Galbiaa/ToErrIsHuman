"""Reproducibility helpers: seeds and environment snapshots."""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path
from typing import Any


def set_global_seed(seed: int) -> None:
    """Fix seeds for Python, NumPy, PyTorch, and scikit-learn where available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from sklearn.utils import check_random_state

        check_random_state(seed)
    except ImportError:
        pass


def xgb_seed_for_fold(base_seed: int, fold_index: int) -> int:
    """XGBoost seed = base_seed + fold number."""
    return int(base_seed) + int(fold_index)


def write_environment_file(path: str | Path) -> Path:
    """Record Python version and installed package versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"executable: {sys.executable}",
        "",
        "packages:",
    ]

    try:
        import importlib.metadata as metadata

        names = [
            "numpy",
            "pandas",
            "openpyxl",
            "scikit-learn",
            "xgboost",
            "joblib",
            "PyYAML",
            "matplotlib",
            "Pillow",
            "tqdm",
            "torch",
            "torchvision",
        ]
        for name in names:
            try:
                ver = metadata.version(name)
            except metadata.PackageNotFoundError:
                ver = "not-installed"
            lines.append(f"  {name}: {ver}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  (could not list packages: {exc})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_config_snapshot(config: dict[str, Any], path: str | Path) -> Path:
    """Save the run config so experiments are auditable."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return path


def bootstrap_run_reproducibility(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Path]:
    """
    Apply seed from config and write environment + config snapshots under outputs/.
    Call at the start of training / validation entrypoints.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    seed = int(config.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)

    out = config.get("outputs", {})
    env_rel = out.get("environment_file", "outputs/environment.txt")
    cfg_rel = out.get("run_config_snapshot", "outputs/run_config_snapshot.yaml")
    env_path = base / env_rel if not Path(env_rel).is_absolute() else Path(env_rel)
    cfg_path = base / cfg_rel if not Path(cfg_rel).is_absolute() else Path(cfg_rel)

    return {
        "environment": write_environment_file(env_path),
        "config_snapshot": write_config_snapshot(config, cfg_path),
    }
