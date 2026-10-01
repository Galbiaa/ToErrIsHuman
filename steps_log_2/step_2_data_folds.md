# Step 2 — Dataset and Fold Integrity Checks

**Date:** 2026-10-01
**Status:** Complete (verification only; no pipeline artifact was regenerated)
**Scope:** `Implementation Plan.md` §2 (data and unit of analysis), §18.1 (case disjointness), §24 (dataset acceptance criteria)

---

## What we did

Verified that the dataset and the fold assignment satisfy every structural
requirement of the plan, both by running the pipeline's own validator and by an
independent read of the saved artifacts:

1. Ran `src/validate_dataset.py` (read-only: it only asserts and prints).
2. Loaded the **existing** `outputs/folds/case_fold_assignment.csv` and the
   USER/ground-truth tables through the project loaders and re-checked counts,
   case-level grouping, one-fold-per-case and train/test disjointness.

We deliberately did **not** re-run `generate_folds.py`, so the committed fold
assignment is left untouched (verified, not regenerated).

## Why

Everything downstream (the diagnostic folds, the nested D split, the case-level
bootstrap) depends on two facts: the counts (427 / 13 / 5,551) and the rule that
all decisions of a case stay in one fold. If either is wrong, every later metric
is invalid. The plan makes this the first acceptance block (§24) and the first
leakage guard (§18.1).

## Commands run

```bash
.venv\\Scripts\\python.exe -u src/validate_dataset.py --config config.yaml
```

```python
# independent cross-check (loaders only, no writes)
from data import load_config, load_user_infos, load_ground_truth
from generate_folds import load_fold_assignment
cfg = load_config("config.yaml")
gt, user = load_ground_truth(cfg), load_user_infos(cfg)
folds = load_fold_assignment("outputs/folds/case_fold_assignment.csv")
```

## Evidence — dataset (`validate_dataset.py`)

```
=== Ground truth ===
Rows: 427  -> OK 427 cases
TARGET=0: 236; TARGET=1: 191; prevalence=0.4473  -> OK

=== User infos ===
Rows: 5551 (columns include case_id, rater_id parsed from id)  -> OK
Unique decision ids -> OK
Cases: 427 (1–427); Raters: 13 (0–12)  -> OK 427 × 13
OK: Every case has exactly 13 ratings
OK: Every rater rated all 427 cases
OK: USER case_id set matches ground-truth CASE-ID set
OK: Observed scales within expected ranges
error-rating prevalence: 0.1904
OK: All 5551 rows satisfy error-rating == 1(rating != TARGET)
OK: features_for_solution_d rejects TARGET
OK: Image filename case IDs match Excel CASE-ID set
OK: All 1281 expected image paths exist
OK: All images readable and convertible to RGB
OK: diagnostic case table is one row per case (no 13× replication)
=== SUMMARY === All dataset checks passed.
```

## Evidence — fold assignment (independent cross-check)

```
== FOLD ASSIGNMENT ==
rows: 427   unique cases: 427
folds: [1, 2, 3, 4, 5]
each case exactly once: True
train/test disjoint (all folds): True
  fold 1 cases: 86
  fold 2 cases: 86
  fold 3 cases: 85
  fold 4 cases: 85
  fold 5 cases: 85

== USER DECISIONS ==
rows: 5551   unique cases: 427   raters: 13
decisions/case min/median/max: 13 13.0 13
all cases have 13 decisions: True

== CROSS-ALIGNMENT ==
USER cases == fold cases: True
GT cases == fold cases: True
```

Fold sizes are 86/86/85/85/85 (not exactly equal because 427 is not divisible by
5); every case appears in exactly one fold, so all decisions of a case stay
together (no row-level splitting).

## Fold stratification (`outputs/folds/fold_balance_stats.csv`)

| fold | cases | decisions | TARGET prevalence | error>median prevalence |
|------|-------|-----------|-------------------|-------------------------|
| 1 | 86 | 1118 | 0.4419 | 0.6279 |
| 2 | 86 | 1118 | 0.4419 | 0.6163 |
| 3 | 85 | 1105 | 0.4471 | 0.6235 |
| 4 | 85 | 1105 | 0.4588 | 0.6235 |
| 5 | 85 | 1105 | 0.4471 | 0.6235 |
| overall | 427 | 5551 | 0.4473 | 0.6230 |

Folds are stratified on `TARGET` and on the case-mean `error-rating`
dichotomised at the median (threshold = 0.076923), so the 4 stratum cells
(`TARGET × error_above_median`) are balanced across folds — this is what
stabilises the per-fold metrics.

## Code-level enforcement of §18.1

Disjointness is enforced, not just observed:

| Guard | Where |
|-------|-------|
| `assert_no_case_leakage(assignment, n_splits)` | `generate_folds.py` (main, after assigning folds) |
| `assert_case_disjoint(train_pool, test_df)` / `(tr, va)` | `train_diagnostic.py` (outer train/test and train/val) |
| `assert_case_disjoint(train_cases, test_cases)` | `train_D.py` (outer split) |
| `assert_case_disjoint(inner_train, inner_test)` | `train_D.py` (inner CV split) |
| `assert_case_disjoint(fit_cases, thr_cases)` | `train_D.py` (threshold-validation split) |

## Result

All dataset and fold integrity criteria of §2, §18.1 and §24 are satisfied on the
current artifacts:

- 427 unique cases, 13 raters, 5,551 decisions, 13 decisions per case;
- `TARGET` binary (236/191, prevalence 0.4473);
- `error-rating == 1(rating != TARGET)` for all 5,551 rows (prevalence 0.1904);
- images: 1,281 expected files all present and readable;
- diagnostic case table is one row per case (no 13× replication);
- folds are case-level, each case in exactly one fold, train ∩ test = ∅ for all
  five folds, and stratification is balanced.

No action required by this step beyond this verification; no artifact was
changed.

## Important caveats

- **Read-only.** `generate_folds.py` was **not** re-run; the committed
  `case_fold_assignment.csv` was validated as-is. (Re-running it with
  `random_state=42` should reproduce the same assignment, but that would rewrite
  the artifact and is intentionally avoided here.)
- Fold sizes differ by one case (86 vs 85) — expected for a stratified 5-fold on
  427 cases, not a bug.
- `validate_dataset.py` also checks readability of all 1,281 MRI files, so it is
  slower than a pure CSV check.

## How to re-run

```bash
# full dataset validation (read-only)
.venv\\Scripts\\python.exe -u src/validate_dataset.py --config config.yaml

# fold assignment + cross-alignment (read-only)
.venv\\Scripts\\python.exe -u src/generate_folds.py --config config.yaml   # regenerates folds (writes outputs/folds/)
```

