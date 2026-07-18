from __future__ import annotations

import argparse
from pathlib import Path

from data import load_config, load_training_dataframe, project_path, validate_image_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    base_dir = Path(args.config).resolve().parent
    cfg = load_config(args.config)
    df = load_training_dataframe(cfg, base_dir=base_dir)

    print(f"Loaded table: {cfg['data']['training_file']}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Cases: {df['case_id'].nunique()} ({df['case_id'].min()} to {df['case_id'].max()})")
    print(f"Raters: {df['rater_id'].nunique()} ({df['rater_id'].min()} to {df['rater_id'].max()})")
    print("Rows per case:")
    print(df.groupby("case_id").size().describe())
    print("Rows per rater:")
    print(df.groupby("rater_id").size().describe())

    target_col = cfg["data"]["target_column"]
    print(f"Target prevalence ({target_col}=1): {df[target_col].mean():.4f}")

    image_dir = project_path(cfg["data"]["image_dir"], base_dir=base_dir)
    expected, missing = validate_image_files(
        case_ids=df["case_id"].unique(),
        image_dir=image_dir,
        orientations=cfg["images"]["orientations"],
        extension=cfg["images"].get("extension", "jpg"),
    )
    print(f"Expected image files: {len(expected)}")
    if missing:
        print(f"Missing image files: {len(missing)}")
        for p in missing[:20]:
            print(f"  - {p}")
        if len(missing) > 20:
            print("  ...")
        raise SystemExit(1)
    print("All expected image files were found.")


if __name__ == "__main__":
    main()
