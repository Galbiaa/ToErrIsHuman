from __future__ import annotations

import argparse
from pathlib import Path

from data import (
    build_fold_assignments,
    compute_fold_stats,
    load_config,
    load_training_dataframe,
    project_path,
    resolve_folds_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate shared GroupKFold assignments (one fold per case_id)."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    base_dir = Path(args.config).resolve().parent
    cfg = load_config(args.config)
    df = load_training_dataframe(cfg, base_dir=base_dir).reset_index(drop=True)

    n_splits = int(cfg["validation"]["n_splits"])
    target_col = cfg["data"]["target_column"]

    assignments = build_fold_assignments(df, n_splits=n_splits)
    assignments_path = resolve_folds_path(cfg, base_dir=base_dir)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)

    stats = compute_fold_stats(df, assignments, target_col=target_col)
    stats_path = project_path("outputs/folds/fold_stats.csv", base_dir=base_dir)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)

    print(f"Saved fold assignments: {assignments_path}")
    print(f"Saved fold statistics:  {stats_path}")
    print(f"Decisions: {len(assignments)} | Cases: {assignments['case_id'].nunique()} | Folds: {n_splits}")
    print("\nPer-fold summary:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
