"""Validate images + IMAGE-GROUND-TRUTH + USER-INFOS-CORRECTED."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/validate_dataset.py` from project root
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import (  # noqa: E402
    EXCEL_GLOBAL_RATER_STATS,
    features_for_solution_d,
    image_is_readable,
    list_image_case_ids,
    load_config,
    load_diagnostic_case_table,
    load_ground_truth,
    load_user_infos,
    project_path,
    validate_image_files,
)
from reproducibility import bootstrap_run_reproducibility  # noqa: E402

EXPECTED_N_CASES = 427
EXPECTED_N_RATERS = 13
EXPECTED_N_ROWS = EXPECTED_N_CASES * EXPECTED_N_RATERS  # 5551
EXPECTED_TARGET_NEG = 236
EXPECTED_TARGET_POS = 191

SCALE_RANGES = {
    "rating-confidence": (1, 5),
    "case-difficulty": (1, 4),
    "rater-expertise": (2, 4),
    "rating": (0, 1),
    "error-rating": (0, 1),
}


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset for two-stage pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    print("=== Ground truth (IMAGE-GROUND-TRUTH) ===")
    gt = load_ground_truth(cfg, base_dir=base_dir)
    print(f"Loaded: {cfg['data']['ground_truth_file']}")
    print(f"Rows: {len(gt)}")
    if len(gt) != EXPECTED_N_CASES:
        _fail(f"Expected {EXPECTED_N_CASES} cases, got {len(gt)}")
    _ok(f"{EXPECTED_N_CASES} cases")

    n_neg = int((gt["TARGET"] == 0).sum())
    n_pos = int((gt["TARGET"] == 1).sum())
    print(f"TARGET=0: {n_neg}; TARGET=1: {n_pos}; prevalence={n_pos / len(gt):.4f}")
    if n_neg != EXPECTED_TARGET_NEG or n_pos != EXPECTED_TARGET_POS:
        _fail(f"Expected TARGET counts {EXPECTED_TARGET_NEG}/{EXPECTED_TARGET_POS}, got {n_neg}/{n_pos}")
    _ok("TARGET distribution matches expected counts (~44.7% positive)")

    case_ids_gt = set(gt["case_id"].tolist())

    print("\n=== User infos (USER-INFOS-CORRECTED) ===")
    users = load_user_infos(cfg, base_dir=base_dir)
    print(f"Loaded: {cfg['data']['user_infos_file']}")
    print(f"Rows: {len(users)}; columns: {list(users.columns)}")
    if len(users) != EXPECTED_N_ROWS:
        _fail(f"Expected {EXPECTED_N_ROWS} decision rows, got {len(users)}")
    _ok(f"{EXPECTED_N_ROWS} decision rows")

    if users["id"].duplicated().any():
        _fail("Duplicate decision id values found")
    _ok("Unique decision ids")

    n_cases = users["case_id"].nunique()
    n_raters = users["rater_id"].nunique()
    print(f"Cases: {n_cases} ({users['case_id'].min()}–{users['case_id'].max()})")
    print(f"Raters: {n_raters} ({users['rater_id'].min()}–{users['rater_id'].max()})")
    if n_cases != EXPECTED_N_CASES:
        _fail(f"Expected {EXPECTED_N_CASES} unique cases in USER, got {n_cases}")
    if n_raters != EXPECTED_N_RATERS:
        _fail(f"Expected {EXPECTED_N_RATERS} raters, got {n_raters}")
    _ok(f"{EXPECTED_N_CASES} cases × {EXPECTED_N_RATERS} raters")

    rows_per_case = users.groupby("case_id").size()
    if not (rows_per_case == EXPECTED_N_RATERS).all():
        bad = rows_per_case[rows_per_case != EXPECTED_N_RATERS]
        _fail(f"Not every case has {EXPECTED_N_RATERS} ratings; examples:\n{bad.head()}")
    _ok(f"Every case has exactly {EXPECTED_N_RATERS} ratings")

    rows_per_rater = users.groupby("rater_id").size()
    if not (rows_per_rater == EXPECTED_N_CASES).all():
        bad = rows_per_rater[rows_per_rater != EXPECTED_N_CASES]
        _fail(f"Not every rater rated all cases; examples:\n{bad.head()}")
    _ok(f"Every rater rated all {EXPECTED_N_CASES} cases")

    case_ids_user = set(users["case_id"].tolist())
    if case_ids_user != case_ids_gt:
        only_user = sorted(case_ids_user - case_ids_gt)[:20]
        only_gt = sorted(case_ids_gt - case_ids_user)[:20]
        _fail(f"CASE-ID mismatch USER vs GT. Only USER: {only_user}; only GT: {only_gt}")
    _ok("USER case_id set matches ground-truth CASE-ID set")

    for col, (lo, hi) in SCALE_RANGES.items():
        vmin, vmax = float(users[col].min()), float(users[col].max())
        if vmin < lo or vmax > hi:
            _fail(f"{col} out of expected range [{lo}, {hi}]: observed [{vmin}, {vmax}]")
        print(f"  {col}: [{vmin}, {vmax}] (expected [{lo}, {hi}])")
    _ok("Observed scales within expected ranges")

    print(f"error-rating prevalence: {users['error-rating'].mean():.4f}")
    print(
        "Note: Excel columns "
        f"{list(users.attrs.get('excel_global_rater_stats', EXCEL_GLOBAL_RATER_STATS))} "
        "are GLOBAL aggregates and must NOT be used in CV "
        f"(usable_in_cv={users.attrs.get('excel_global_rater_stats_usable_in_cv', False)})."
    )

    print("\n=== Join check: error-rating == 1(rating != TARGET) ===")
    joined = users.merge(gt, on="case_id", how="left", validate="many_to_one")
    if joined["TARGET"].isna().any():
        _fail("Some USER rows have no matching TARGET")
    expected_error = (joined["rating"] != joined["TARGET"]).astype(int)
    n_mismatch = int((joined["error-rating"] != expected_error).sum())
    if n_mismatch:
        bad = joined.loc[
            joined["error-rating"] != expected_error, ["id", "rating", "TARGET", "error-rating"]
        ]
        _fail(f"{n_mismatch} rows where error-rating != 1(rating!=TARGET):\n{bad.head(10)}")
    _ok("All 5551 rows satisfy error-rating == 1(rating != TARGET)")

    # TARGET must not leak into D feature frames
    try:
        features_for_solution_d(joined, cfg)
        _fail("features_for_solution_d should reject frames that still contain TARGET")
    except ValueError:
        _ok("features_for_solution_d rejects TARGET")
    users_no_target = features_for_solution_d(users, cfg)
    if "TARGET" in users_no_target.columns:
        _fail("TARGET still present after features_for_solution_d")
    _ok("USER frame without TARGET is accepted for D feature construction")

    print("\n=== Images ===")
    image_dir = project_path(cfg["data"]["image_dir"], base_dir=base_dir)
    orientations = cfg["images"]["orientations"]
    extension = cfg["images"].get("extension", "jpg")

    file_case_ids = list_image_case_ids(image_dir, orientations, extension)
    file_set = set(file_case_ids)
    if file_set != case_ids_gt:
        only_files = sorted(file_set - case_ids_gt)[:20]
        only_excel = sorted(case_ids_gt - file_set)[:20]
        _fail(f"Image case IDs != Excel CASE-ID. Only files: {only_files}; only Excel: {only_excel}")
    _ok("Image filename case IDs match Excel CASE-ID set")

    expected, missing = validate_image_files(case_ids_gt, image_dir, orientations, extension)
    print(f"Expected image files: {len(expected)}")
    if missing:
        for p in missing[:20]:
            print(f"  missing: {p}")
        _fail(f"{len(missing)} missing image files")
    if len(expected) != EXPECTED_N_CASES * len(orientations):
        _fail(f"Expected {EXPECTED_N_CASES * len(orientations)} images, got {len(expected)}")
    _ok(f"All {len(expected)} expected image paths exist")

    unreadable = [p for p in expected if not image_is_readable(p)]
    if unreadable:
        for p in unreadable[:20]:
            print(f"  unreadable: {p}")
        _fail(f"{len(unreadable)} images failed readability/RGB conversion")
    _ok("All images readable and convertible to RGB")

    print("\n=== diagnostic case table ===")
    case_table = load_diagnostic_case_table(cfg, base_dir=base_dir)
    print(f"Rows: {len(case_table)}; columns: {list(case_table.columns)}")
    if len(case_table) != EXPECTED_N_CASES:
        _fail("diagnostic case table row count mismatch")
    if case_table["case_id"].duplicated().any():
        _fail("diagnostic case table has duplicated case_id (must not replicate ×13)")
    _ok("diagnostic case table is one row per case (no 13× replication)")

    print("\n=== SUMMARY ===")
    print("All dataset checks passed.")


if __name__ == "__main__":
    main()
