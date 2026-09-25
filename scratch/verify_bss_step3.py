"""Temporary sanity check for Step 3 (BSS) — reads artifacts only, writes nothing."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_D_vs_baseline import load_aligned_scores  # noqa: E402
from data import load_config, project_path  # noqa: E402
from metrics import brier_skill_score  # noqa: E402

cfg = load_config("config.yaml")
base_dir = Path("config.yaml").resolve().parent
base_csv = project_path(cfg["outputs"]["baseline_dir"], base_dir) / "decision_level_features.csv"
d_dir = project_path(cfg["outputs"]["d_dir"], base_dir)

df = load_aligned_scores(baseline_csv=base_csv, d_csv=d_dir / "oof_predictions.csv")
y = df["error-rating"].to_numpy(int)
b = df["q_ir"].to_numpy(float)
d = df["d_probability"].to_numpy(float)

print(f"n={len(y)} prevalence={y.mean():.4f}")
print(f"brier_q_ir={brier_score_loss(y, b):.6f} bss_q_ir={brier_skill_score(y, b):.6f}")
print(f"brier_D={brier_score_loss(y, d):.6f} bss_D={brier_skill_score(y, d):.6f}")

# Manual cross-check of the formula
prev = y.mean()
print(f"manual bss_q_ir={1.0 - brier_score_loss(y, b) / (prev * (1 - prev)):.6f}")