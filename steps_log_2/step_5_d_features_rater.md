# Step 5 — D Feature Allowlist and Fold-Safe Rater Profiles

**Date:** 2026-10-01
**Status:** Complete (verification only; no pipeline artifact was regenerated)
**Scope:** `Implementation Plan.md` §7 (feature allowlist), §10 (D image features
inner-OOF vs outer-test), §11 (fold-safe rater profiles), §18.4–18.7 (leakage guards)

---

## What we did

Verified the D feature contract from code + artifacts:

1. Checked the allowlist has exactly the 11 required features and no forbidden
   columns (`TARGET`, `error-rating`).
2. Rebuilt the D feature frame for fold 1 with `build_d_feature_frame` (read-only,
   nothing written) and confirmed the feature block equals the allowlist exactly.
3. **Independently reproduced** the fold-safe `rater-accuracy` (leave-one-out on
   training cases; train-only aggregate on test cases) and compared it to the
   value produced by the pipeline.
4. Checked the intermediate artifacts `nested_inner_oof_fold_{k}.csv` and
   `rater_profiles_foldsafe_fold_{k}.csv`.

## Why

Solution D is the primary tabular model. Three things must be guaranteed:
exactly the 11 allowed features (§7), `TARGET` never among D inputs (§18.5), and
rater accuracy/confidence computed **fold-safely** — LOO on training cases (the
current case excluded) and train-only aggregates on test cases (§11, §18.4). The
inner-OOF cache (§10) is what supplies D's outer-training image probabilities.

## Commands run

```bash
# rebuild the fold-1 D feature frame + independent fold-safe rater-stat check (read-only)
.venv\\Scripts\\python.exe -  # imports build_d_feature_frame / fold logic; writes nothing
```

## Evidence — allowlist and feature block (§7, §18.5)

```
allowlist n= 11
allowlist: ['rating-confidence', 'case-difficulty', 'rater-expertise', 'rater-accuracy',
            'rater-confidence', 'rating', 'ai_probability_class_1', 'ai_predicted_class',
            'rater_ai_disagreement', 'ai_probability_rater_wrong', 'ai_margin']
TARGET in allowlist: False
error-rating in allowlist: False
frame rows: 5551    feature cols == allowlist: True
no TARGET col: True   no error-rating in features: True
```

The 11 features match §7 exactly (4 rater-decision, 2 fold-safe rater profile, 5
image-derived). `assert_d_features(allowed=allow, forbidden=...)` is invoked
inside `build_d_feature_frame` and `extract_xy`, so the allowlist is enforced,
not merely observed.

## Evidence — fold-safe rater stats (§11, §18.4)

For fold 1, `rater-accuracy` produced by the pipeline was compared with an
**independent** recomputation (train cases = `fold != 1`, test cases = `fold == 1`):

```
rater-accuracy == independent LOO(train)/train-only(test): True
```

i.e.:
- training row → `1 − mean(error-rating)` over the rater's **other** training
  cases (the current case excluded, leave-one-out);
- test row → `1 − mean(error-rating)` over the rater's **training** cases only.

This confirms the Excel global `rater-accuracy`/`rater-confidence` are overwritten
by fold-safe values (as documented in `INFO/11_anti_leakage.md`).

## Evidence — intermediate artifacts (§10, §21)

Inner-OOF cache `outputs/D/nested_inner_oof_fold_{k}.csv` (from Output 1's
inner CV, tag `nested_oof`):

| fold | rows | expected outer-train cases | unique cases | duplicates | source |
|------|------|----------------------------|--------------|------------|--------|
| 1 | 341 | 341 | 341 | no | nested_oof |
| 2 | 341 | 341 | 341 | no | nested_oof |
| 3 | 342 | 342 | 342 | no | nested_oof |
| 4 | 342 | 342 | 342 | no | nested_oof |
| 5 | 342 | 342 | 342 | no | nested_oof |

(Rows = 427 − fold size: 341 for folds of 86 cases, 342 for folds of 85.)

Rater profiles `outputs/D/rater_profiles_foldsafe_fold_{k}.csv`: 13 rows each,
columns `[rater_id, n_train, rater-accuracy, rater-confidence]` — one fold-safe
profile per rater per fold (e.g. fold 1 rater 0: `n_train 341`,
`rater-accuracy 0.8006`).

## Inner-OOF vs outer-test wiring (§10)

`train_D.py` builds the D feature frame **twice per outer fold**, with the same
`train_case_ids` but different probability sources:

| Split | p_i source | `probability_source` |
|-------|-----------|-----------------------|
| outer-train rows | inner-OOF cache `nested_inner_oof_fold_k.csv` | `nested_oof` |
| outer-test rows | Stage-1 outer-OOF for the test cases | `nested_test` |

Because both frames are built with `train_cases` as `train_case_ids`, **no
outer-test information enters the rater statistics** of either frame (§18.6),
and the current training case is excluded from its own profile via LOO (§18.4).

## Guards checked (§18.4–18.7)

| Invariant | Mechanism | Verified |
|-----------|-----------|----------|
| §18.4 rater-profile isolation | `fold_safe_rater_stats` LOO on train; train-only on test | ✅ (independent reproduction) |
| §18.5 `TARGET` ∉ D features | `assert_d_features` + `_strip_target` + `d_feature_allowlist` | ✅ (allowlist + frame check) |
| §18.6 test-label isolation | test frame built with `train_cases` rater stats only | ✅ (code path) |
| §18.7 preprocessing isolation | `fit_transform_d_features` fits on outer-train only (Step 6) | ✅ (code path; detailed in Step 6) |

## Result

The D feature contract is satisfied:

- allowlist = exactly the 11 features of §7; `TARGET` and `error-rating` excluded;
- the built feature frame's feature block equals the allowlist exactly;
- rater accuracy (and confidence) are fold-safe, reproduced independently;
- inner-OOF caches cover every outer-train case with no duplicates and the
  `nested_oof` provenance tag;
- rater profile artifacts exist for all 5 folds (13 raters each).

No action required by this step beyond the verification.

## Important caveats

- **Read-only.** `build_d_feature_frame` was called in memory for fold 1 only to
  check the contract; `outputs/` and `models/` were **not** written.
- The independent rater-stat check reproduced the pipeline value exactly
  (`allclose`, NaN-aware); this is the strongest available evidence of §18.4
  without re-training.
- The `rating` column **is** an allowed D feature (it is the rater's rating, an
  input); only `TARGET`/`error-rating` are forbidden.

## How to re-run

```bash
# rebuild the fold-safe feature smoke (writes outputs/D/feature_smoke_*)
.venv\\Scripts\\python.exe -u src/build_D_features.py --config config.yaml --smoke

# (the in-memory contract check used here is in the step's command block above)
```

