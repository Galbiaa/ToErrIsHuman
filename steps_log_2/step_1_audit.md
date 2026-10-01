# Step 1 — Audit of current pipeline vs Implementation Plan

**Date:** 2026-10-01
**Status:** Complete (read-only audit; no code and no artifacts were modified)
**Scope:** `Implementation Plan.md` §1, §17, §21, §26 (+ cross-references used in later steps)

---

## What we did

A full, read-only audit of the existing two-stage pipeline against the
requirements of `Implementation Plan.md`. We read every pipeline script,
`config.yaml`, the methodological docs in `INFO/`, the verification scripts in
`tmp/`, and the current artifacts under `outputs/`, then produced a
requirement-by-requirement mapping (implemented / partial / missing) plus a
comparison of the current metrics against the plan's reference benchmarks.

No file in `src/`, `config.yaml`, `outputs/` or `models/` was changed.

## Why

Before modifying anything, we must know exactly what the current code already
does, what it does not, and where its numbers sit relative to the plan's
independent verification targets. The plan is explicit (§9, §20) that the
benchmarks must **not** be reached by weakening the anti-leakage protocol, so we
first establish the gap between "what exists" and "what the plan asks for".

## Files and artifacts inspected

| Area | Files |
|------|-------|
| Data / folds | `src/data.py`, `src/validate_dataset.py`, `src/generate_folds.py` |
| Guards | `src/methodology_guards.py` |
| Output 1 (diagnostic) | `src/train_diagnostic.py`, `src/models.py` |
| Output 2 (baseline) | `src/build_baseline.py` |
| Features + Output 3 | `src/build_D_features.py`, `src/train_D.py` |
| Evaluation | `src/compare_D_vs_baseline.py`, `src/metrics.py`, `src/analyze_errors.py` |
| Inference / export | `src/infer_new_case.py`, `src/export_oof_predictions.py`, `src/web_inference.py` |
| Config / docs | `config.yaml`, `INFO/*.md`, `INFO/11_anti_leakage.md`, `STEP_LOGS/*.md` |
| Verification | `tmp/verify_predict_proba_class_order.py`, `tmp/verify_q_ir_and_rater_stats.py` |
| Existing artifacts | `outputs/diagnostic/metrics_pooled.csv`, `outputs/baseline/baseline_q_ir_metrics.csv`, `outputs/D/metrics_pooled_weighted.csv`, `outputs/compare/compare_summary.json` |

## Mapping: plan requirement → current implementation

| Plan | Requirement | Where implemented | Status |
|------|-------------|-------------------|--------|
| §1, §26 | Three distinct outputs: diagnostic / `q_ir` / D | `train_diagnostic.py`, `build_baseline.py`, `train_D.py` | OK |
| §2, §18.1 | 427 cases, 13 raters, 5,551 decisions; case-level folds; train ∩ test = ∅ | `validate_dataset.py` (427/13/5551 asserts), `generate_folds.py` (`assert_no_case_leakage`), `assert_case_disjoint` in `train_D.py` | OK |
| §3.1 | Stage 1 predicts `TARGET` | `DiagnosticDataset` label = `TARGET`; `models.py` ResNet-18 tri-view | OK |
| §3.2 | `error-rating = 1(rating != TARGET)` | enforced in `validate_dataset.py` (all 5,551 rows) | OK |
| §4.2, §4.3 | 5 outer folds; 1 prediction/case; `outputs/diagnostic/oof_predictions.csv` | `train_diagnostic.py` (oof rows: `case_id,target,ai_probability_class_1,fold,threshold_youden`) | OK (column names differ cosmetically from plan) |
| §5, §5.4 | `q_ir = p_i if R=0 else 1-p_i`; 5,551 preds; baseline outputs | `build_baseline.py` (`compute_ai_decision_features`); writes `decision_level_features.csv`, `oof_q_ir_predictions.csv`, `baseline_q_ir_metrics.csv` | OK |
| §6, §7 | D = XGBoost on exactly 11 allowed features | `config.yaml:solution_d.features` (11), `d_feature_allowlist`, `assert_d_features` | OK |
| §8, §9 | Nested D: inner-OOF for outer-train, outer-test p_i for outer-test | `train_D.py` `compute_inner_oof_probabilities` (4-fold, tag `nested_oof`), `outer_test_probabilities_from_diagnostic` (tag `nested_test`) | OK |
| §10 | D image features built from inner-OOF vs outer-test | `build_d_feature_frame` called separately for train/test with distinct `probability_source` | OK |
| §11 | Fold-safe rater-accuracy/confidence (LOO on train; train-only on test) | `fold_safe_rater_stats`; Excel globals overwritten in `build_d_feature_frame` | OK |
| §12, §13 | Fit preprocessor on outer-train only; saved with model | `fit_transform_d_features` (fit on `X_train`), `save_preprocessor` → `preprocessor_fold_k.joblib` | OK |
| §14 | Threshold chosen on validation, never on test | `select_threshold_on_validation` (Youden) in `train_diagnostic.py` and `train_D.py` (`fit_cases`/`thr_cases` split) | OK |
| §17 | Scripts for compare/analysis | `compare_D_vs_baseline.py`, `analyze_errors.py` | OK |
| §19 | Case-level bootstrap, n=1000, paired Δ = D − q_ir | `case_bootstrap_indices` used in `compare_D_vs_baseline.py` | OK |
| §21 | Expected output tree | all listed files present under `outputs/` | OK |
| §24 | Acceptance criteria (counts, target, allowlist, leakage) | enforced across the scripts above | Mostly OK |
| §18 | "explicit automated checks" for every guard | inline `assert_*` + `tmp/verify_*` scripts | **Partial** (no single consolidated guard-test script) |
| §20 | Reference benchmarks as verification targets | not yet formally compared/reported | **Open** (this audit starts it) |
| §25 | README documentation of methodology | `README.md` covers pipeline order but not the plan's §25 points | **Open** |

## Findings / gaps

1. **Protocol is already aligned (nested).** The nested cross-fitting required by
   §8–§9 is implemented, including inner-OOF for outer-train rows and outer-test
   probabilities from a diagnostic that never saw the test cases. The plan
   (§9, Step 2) explicitly accepts reusing the Stage-1 outer-OOF as the outer-test
   diagnostic input ("equivalent to the outer OOF diagnostic predictions"), which
   is what `outer_test_probabilities_from_diagnostic` does.

2. **Anti-leakage controls are the same 11 points already documented** in
   `INFO/11_anti_leakage.md` (TARGET excluded, fold-safe rater stats, threshold on
   validation, case-level bootstrap, declared probability provenance). They map
   1:1 onto the plan's §18 invariants.

3. **Consolidated guard verification is missing (plan §18 / §26).** Guards exist as
   inline asserts and scattered `tmp/verify_*.py`, but there is no single script
   that asserts every invariant end-to-end and can be re-run as a "unit test"
   of the pipeline (this becomes **Step 8**).

4. **README does not yet document the plan's §25 points** (original scientific
   question, `TARGET` vs `error-rating`, three outputs, `q_ir`, allowlist, nested
   protocol, fold-safe profiles, case bootstrap, reference benchmarks). This
   becomes **Step 10**.

5. **Metric discrepancies vs the plan's benchmarks exist and must be reported,
   not "fixed" by changing the protocol.** See the table below. The most
   important: the plan's D benchmark (AUROC ≈ 0.714) is much closer to the
   repo's **fast** protocol (0.7028) than to the primary **nested** protocol
   (0.6327). Per §20 this must **not** be resolved by switching to fast.

6. **Working-tree hygiene (not a code gap).** On branch `pipeline/3-outputs` the
   tree is not perfectly clean: `.gitignore` is modified (renames the ignored
   plan file to `Implementation Plan.md`) and there is a new untracked
   `steps_log_2/`. To be committed deliberately.

## Metric comparison vs plan benchmarks (§20)

| Quantity | Plan target | Current (repo) | Δ | Source |
|----------|-------------|----------------|---|--------|
| Diagnostic AUROC (427 cases) | ≈ 0.759 | 0.7671 | +0.008 | `outputs/diagnostic/metrics_pooled.csv` |
| Diagnostic Brier | ≈ 0.201 | 0.2101 | +0.009 | same |
| D AUROC (nested, weighted) | ≈ 0.714 | 0.6327 | −0.081 | `metrics_pooled_weighted.csv` |
| D AUPRC (nested, weighted) | ≈ 0.431 | 0.3247 | −0.106 | same |
| D Brier (nested, weighted) | ≈ 0.154 | 0.1520 | −0.002 | same |
| D AUROC (fast, declared) | — | 0.7028 | — | closer to plan target than nested |
| `q_ir` AUROC / AUPRC | — | 0.7159 / 0.4158 | — | `outputs/baseline/baseline_q_ir_metrics.csv` |

Interpretation: the image-only numbers are within ~0.01 of the plan targets;
the D numbers are not. The plan (§20, §24) says to **investigate** data
processing, architecture, preprocessing, seeds, folds or feature construction —
never to lower the anti-leakage bar to match the target.

## Result

The pipeline is **structurally aligned** with the plan: units of analysis, fold
policy, three outputs, nested protocol, fold-safe rater profiles, preprocessing
isolation, validation-only thresholds and case-level bootstrap are all present.
The remaining work is not a redesign but: (a) formalising the guards as automated
checks (Step 8), (b) documenting the methodology in the README (Step 10), and
(c) producing an evidence-based verification of the reference benchmarks and a
discrepancy report (Step 9) — with the D AUROC/AUPRC gap as the key item to
investigate.

## Important caveats

- This step is **read-only**: no script was executed and no artifact was
  regenerated. All numbers come from the existing committed artifacts.
- The metric differences are **expected** for an independent implementation and
  are not treated as errors to be suppressed; they are the object of Step 9.
- The plan file is `Implementation Plan.md` (with a space and capitals), not
  `implementation_plan.md`; the initial task reference used the wrong name.

## How to re-run (verification, no re-training)

```bash
# dataset + counts + TARGET/error-rating consistency
.venv\\Scripts\\python.exe -u src/validate_dataset.py --config config.yaml

# inspect current diagnostic metrics
.venv\\Scripts\\python.exe -c "import pandas as pd; print(pd.read_csv('outputs/diagnostic/metrics_pooled.csv').T)"
```
