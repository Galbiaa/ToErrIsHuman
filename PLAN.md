# Action Plan: Verifications and Improvements

## Overview
This plan addresses 7 points from `richieste-1.txt`. Sections are ordered by execution sequence, not by request number.

---

## Step 0: Verify Definitions (Context - No Code Changes)

**q_ir**: Image-only baseline on error-rating. For case i, p_i = P(TARGET=1 | images). Given rating R_ir: q_ir = p_i if R_ir=0, else 1-p_i. This is P(rater error | images + rating).

**D**: XGBoost on error-rating using 11 features. TARGET never used.

**D nested**: Outer-train probabilities from inner CV OOF. Outer-test from diagnostic on outer-train only.

### Verification:
- CONFIRMED `build_baseline.py:65`: `q_ir = np.where(r == 0, p, 1.0 - p)`
- CONFIRMED `config.yaml:57-69`: 11 features, TARGET forbidden
- CONFIRMED `train_D.py:229-263`: nested protocol correct

**Action: No code changes needed. Document definitions in report.**

---

## Step 1: Verify predict_proba[:,1] = P(error-rating=1)

**What:** Explicitly verify column 1 corresponds to error class.

**File:** `src/train_D.py` (lines 308, 316)

**Implementation:** Add after model fitting (line 315):
```python
assert list(clf.classes_) == [0, 1], f"Unexpected class order: {clf.classes_}"
```

---

## Step 2: Document scale_pos_weight Per Fold

**What:** Report n_pos, n_neg, scale_pos_weight per fold. Verify formula is n_neg/n_pos.

**File:** `src/train_D.py` (lines 291-294)

**Current code:**
```python
n_pos = max(int((y_all_train == 1).sum()), 1)
n_neg = int((y_all_train == 0).sum())
runs.append(("weighted", float(n_neg) / float(n_pos)))
```

**Verification:** CONFIRMED formula is n_neg/n_pos (correct for XGBoost).

**Implementation:** Add logging after line 294 and save `outputs/D/scale_pos_weight_per_fold.csv` with columns: fold, n_positive, n_negative, scale_pos_weight.

---

## Step 3: Add Brier Skill Score

**What:** Add BSS to comparison. Calibration slope/intercept already computed.

**Files:** `src/metrics.py`, `src/compare_D_vs_baseline.py`

**Implementation in metrics.py:**
```python
def brier_skill_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    brier_model = brier_score_loss(y_true, y_prob)
    prevalence = np.mean(y_true)
    brier_ref = prevalence * (1.0 - prevalence)
    if brier_ref == 0:
        return np.nan
    return 1.0 - (brier_model / brier_ref)
```

Add BSS to comparison output for both q_ir and D.

---

## Step 4: Export OOF Prediction Files

**What:** Clean files with: case_id, rater_id, y_true, predicted_probability, fold.

**New file:** `src/export_oof_predictions.py`

**Implementation:**
1. Read `outputs/baseline/decision_level_features.csv`, extract case_id, rater_id, error-rating as y_true, q_ir as predicted_probability, fold. Save to `outputs/baseline/oof_q_ir_predictions.csv`.
2. Read `outputs/D/oof_predictions.csv`, extract case_id, rater_id, error-rating as y_true, d_probability as predicted_probability, fold. Save to `outputs/D/oof_D_predictions.csv`.

---

## Step 5: Unified Pooled Metrics Table

**What:** Side-by-side table for D weighted vs D unweighted.

**File:** `src/compare_D_vs_baseline.py`

**Implementation:** Add `generate_unified_metrics_table()` that reads both pooled CSVs and outputs: Metric, D_weighted, D_unweighted for AUROC, AUPRC, Brier, log_loss, precision, recall, specificity, balanced_accuracy, F1. Save to `outputs/compare/unified_D_metrics_table.csv`.

---

## Step 6: Implement Fast Protocol

**What:** Train diagnostic on ALL outer-train (in-sample probabilities for D training).

**File:** `src/train_D.py` (implement `run_fast()` at line 433)

**Key difference from nested:**
- Nested: outer-train probabilities from inner CV OOF (unseen)
- Fast: outer-train probabilities from diagnostic trained on ALL outer-train (in-sample)

**Implementation steps:**
1. For each outer fold: train diagnostic on all outer-train, predict train (in-sample) and test
2. Build D features, train XGBoost, evaluate
3. Save to `outputs/D/oof_predictions_fast.csv` and `models/D/scenario_D_fold_*_fast.joblib`
4. Add comparison: D nested AUROC vs D fast AUROC (expect ~0.63 vs ~0.70)

---

## Summary

| File | Changes |
|------|---------|
| `src/train_D.py` | class verification, scale_pos_weight logging, fast protocol |
| `src/compare_D_vs_baseline.py` | Brier Skill Score, unified metrics table |
| `src/metrics.py` | add `brier_skill_score()` |
| `src/export_oof_predictions.py` | NEW - export clean OOF files |