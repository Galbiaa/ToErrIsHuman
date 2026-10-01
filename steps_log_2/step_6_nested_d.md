# Step 6 — Nested D Training, Preprocessing Isolation, Pooled OOF

**Date:** 2026-10-01
**Status:** Complete (verification only; no re-training performed)
**Scope:** `Implementation Plan.md` §8 (nested outer loop), §9 (nested p_i), §12
(preprocessing), §13 (preprocessing isolation), §18.8 (fold alignment)

---

## What we did

Verified the nested D training mechanics and outputs from the committed joblibs
and artifacts (no re-training):

1. Inspected the saved model payloads `models/D/scenario_D_fold_{k}.joblib`
   (keys, `outer_fold`, `protocol`, feature names, threshold).
2. Inspected the saved preprocessor `models/D/preprocessor_fold_{k}.joblib`.
3. **Independently reproduced the preprocessor fit** for every fold from the
   outer-train feature frame and compared it against the saved preprocessor
   (imputer statistics, scaler mean/scale) → §13 isolation.
4. Verified the pooled OOF files and recomputed the pooled metrics.

## Why

The whole validity of D rests on the nested discipline: features and
preprocessing fit on outer-train only, outer-test never touched, and one pooled
OOF prediction per decision. §13 in particular requires the preprocessing to be
fit on outer-train only and saved with the model.

## Commands run

```bash
# inspect joblibs + reproduce the preprocessor fit per fold (read-only)
.venv\\Scripts\\python.exe -  # imports build_d_feature_frame / make_d_preprocessor / extract_xy
```

## Evidence — D model joblib payloads (§8, §12)

```
fold k: keys=['feature_names','model','outer_fold','protocol','run_name',
              'scale_pos_weight','test_probability_source','threshold_youden',
              'train_probability_source']
  fold 1: outer_fold=1 protocol=nested feat==allow=True thr=0.4864
  fold 2: outer_fold=2 protocol=nested feat==allow=True thr=0.4492
  fold 3: outer_fold=3 protocol=nested feat==allow=True thr=0.4456
  fold 4: outer_fold=4 protocol=nested feat==allow=True thr=0.5368
  fold 5: outer_fold=5 protocol=nested feat==allow=True thr=0.3723
```

Each payload stores the model, the exact feature names (= the 11-feature
allowlist), the outer fold index, the protocol, the run name, the effective
`scale_pos_weight`, the Youden threshold, and the train/test probability
provenance (`nested_oof` / `nested_test`) — so the artifact is self-describing and
re-loadable for inference.

Preprocessor payload: `Pipeline([('imputer', SimpleImputer), ('scaler', StandardScaler)])`
with `imputer.strategy = "median"`.

## Evidence — preprocessing isolation (§13, §18.7)

For each fold, a fresh `make_d_preprocessor()` was fit on the fold's outer-train
feature frame (`build_d_feature_frame` with the inner-OOF p_i, filtered to the
outer-train cases) and compared with the saved preprocessor:

```
fold 1: train_rows=4433 imputer_stats=True scaler_mean=True scaler_scale=True
fold 2: train_rows=4433 imputer_stats=True scaler_mean=True scaler_scale=True
fold 3: train_rows=4446 imputer_stats=True scaler_mean=True scaler_scale=True
fold 4: train_rows=4446 imputer_stats=True scaler_mean=True scaler_scale=True
fold 5: train_rows=4446 imputer_stats=True scaler_mean=True scaler_scale=True
ALL preprocessors fit exactly on outer-train (no test data): True
```

The saved preprocessor for every fold is **exactly** the one obtained by fitting
only on the outer-train rows (4433 = 341 cases × 13; 4446 = 342 × 13). No
outer-test information enters the imputation statistics nor the scaling.

## Evidence — pooled OOF (§9, §21)

```
D oof columns: ['id','case_id','rater_id','fold','error-rating','d_probability',
                'probability_source_ai','run','threshold_youden']
rows: 5551   unique id: 5551   cases: 427
recomputed AUROC: 0.6327   AUPRC: 0.3247   Brier: 0.152
rows per fold: {1: 1118, 2: 1118, 3: 1105, 4: 1105, 5: 1105}
probability_source_ai: ['nested_test']
unweighted oof rows: 5551
```

The 5,551 rows = exactly one pooled outer-OOF prediction per decision (fold sizes
× 13: 1118/1118/1105/1105/1105). Recomputed metrics match
`outputs/D/metrics_pooled_weighted.csv` (AUROC 0.6327297, AUPRC 0.3247056, Brier
0.1520183).

## Training mechanics (§8, §9, §14)

Per outer fold `k` (`src/train_D.py`, `train_d_one_outer_fold`):

1. `train_cases` = `fold != k`, `test_cases` = `fold == k`; `assert_case_disjoint`.
2. `p_i` for outer-train = **inner-CV OOF** (`nested_oof`); `p_i` for outer-test =
   Stage-1 outer-OOF (`nested_test`).
3. Feature frames built **separately** for train/test but with the **same**
   `train_case_ids` (§10/§18.6).
4. Threshold split: `fit_cases`/`thr_cases` = 80/20 of outer-train, stratified by
   `TARGET`, `random_state = validation.random_state + fold`; `assert_case_disjoint`.
5. **Threshold model**: preprocessor fit on `fit_cases`; XGBoost fit on
   `fit_cases`; threshold = Youden on `thr_cases` (validation only, never test).
6. **Final model**: preprocessor fit on **all** outer-train; XGBoost on all
   outer-train; predict outer-test **once**. `assert clf.classes_ == [0, 1]`.
7. `scale_pos_weight = n_negative / n_positive` (over outer-train), documented in
   `scale_pos_weight_per_fold.csv`; an unweighted run is also produced.

## Fold alignment (§18.8)

All fold-`k` artifacts refer to the same case partition (fold assignment from
`outputs/folds/case_fold_assignment.csv`):

| Artifact | Keyed by |
|----------|----------|
| `models/diagnostic/diagnostic_fold_{k}.pt` | fold `k` |
| `outputs/D/nested_inner_oof_fold_{k}.csv` | fold `k` |
| `outputs/D/rater_profiles_foldsafe_fold_{k}.csv` | fold `k` |
| `models/D/preprocessor_fold_{k}.joblib` | fold `k` |
| `models/D/scenario_D_fold_{k}.joblib` (`outer_fold=k`) | fold `k` |

## Result

The nested D training and its outputs satisfy §8, §9, §12, §13, §18.8:

- nested protocol with inner-OOF (train) / outer-test p_i;
- preprocessor fit **only** on outer-train, reproduced exactly for all 5 folds;
- model + preprocessor saved per fold with self-describing metadata;
- pooled OOF = 5,551 predictions (weighted and unweighted);
- pooled metrics reproduce from the OOF file (AUROC 0.6327, AUPRC 0.3247,
  Brier 0.1520).

No action required by this step beyond the verification.

## Important caveats

- **Read-only.** `build_d_feature_frame` / `make_d_preprocessor` were called in
  memory; `outputs/` and `models/` were **not** written.
- The preprocessor is fit on **all** outer-train rows (not just the 80 % fit
  subset); the threshold-selection model uses a separate preprocessor fit on the
  fit subset. Both are training-only — no leakage.
- Minor code observation (non-blocking): the fallback `if not runs:` block in
  `train_D.py` (≈ line 415) is effectively dead given the current config
  (`use_scale_pos_weight` and `also_run_without_weighting` are both true) and has
  unusual indentation; it does not affect the results.
- Pooled weighted vs unweighted: weighted AUROC 0.6327 (lower), unweighted 0.6516;
  weighted Brier 0.1520 (lower/better) — carried into Step 9.

## How to re-run

```bash
# single fold (single-fold runs skip pooling/export)
.venv\\Scripts\\python.exe -u src/train_D.py --config config.yaml --protocol nested --fold 1

# full nested run (long; GPU recommended for the inner diagnostic trainings)
.venv\\Scripts\\python.exe -u src/train_D.py --config config.yaml --protocol nested
```

