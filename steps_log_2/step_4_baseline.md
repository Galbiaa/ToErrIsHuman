# Step 4 — Output 2: Image-Only Error Baseline (q_ir)

**Date:** 2026-10-01
**Status:** Complete (verification only; no re-training performed)
**Scope:** `Implementation Plan.md` §3.2 (rater error target), §5 (Output 2)

---

## What we did

Verified the image-only error baseline `q_ir` from the committed artifacts:

1. Loaded `outputs/baseline/decision_level_features.csv` (5,551 decision rows) and
   checked the `q_ir` formula, its range and its identity with
   `ai_probability_rater_wrong`.
2. Confirmed the per-case probability used is exactly the diagnostic **outer-OOF**
   `p_i` (joined by `case_id`), i.e. `probability_source = nested_oof`.
3. Loaded the clean export `outputs/baseline/oof_q_ir_predictions.csv` and
   recomputed AUROC / AUPRC / Brier / log-loss, comparing with
   `baseline_q_ir_metrics.csv`.
4. Confirmed the three required outputs exist and read the calibration note.

## Why

`q_ir` is the "human error predicted from images only" reference that solution D
is compared against — on the **same 5,551 decisions** and the **same target**
`error-rating`. It must use the case-level outer-OOF `p_i` and the exact formula
`q_ir = p_i if R=0 else 1−p_i`; any deviation would make the paired D comparison
meaningless.

## Commands run

```bash
# independent recompute from the exported baseline file (no writes)
.venv\\Scripts\\python.exe -  # reads decision_level_features.csv + oof_q_ir_predictions.csv
```

## Evidence (independent recompute)

```
base columns: ['id','case_id','rater_id','fold','rating','error-rating','q_ir',
               'ai_probability_class_1','ai_predicted_class','rater_ai_disagreement',
               'ai_probability_rater_wrong','ai_margin','rating-confidence',
               'case-difficulty','rater-expertise','rater-accuracy','rater-confidence',
               'probability_source','ai_decision_threshold']
base rows: 5551   cases: 427   raters: 13
q_ir == (p_i if rating==0 else 1-p_i): True
q_ir == ai_probability_rater_wrong: True
q_ir range: 0.0002 .. 0.9998
p_i equals diagnostic outer-OOF: True
probability_source: ['nested_oof']
oof_q_ir columns: ['id','case_id','rater_id','y_true','predicted_probability','fold']
oof_q_ir rows: 5551   cases: 427
recomputed AUROC: 0.7159   AUPRC: 0.4158   Brier: 0.2101   logloss: 0.6914
mean q_ir: 0.3622   error prevalence: 0.1904
```

## Formula and provenance (§5.1, §5.2)

- `q_ir = p_i if R=0 else 1−p_i` holds **exactly** on all 5,551 rows.
- `q_ir` is identical to `ai_probability_rater_wrong` — the same quantity is reused
  as a D image feature, so baseline and D are guaranteed consistent by construction.
- The per-case probability is the **case-level outer-OOF** `p_i` from Output 1
  (join by `case_id` kept all 5,551 rows; `probability_source = "nested_oof"`).
- Each case's single `p_i` is copied to its 13 decisions and transformed by the
  rating, exactly as §5.2 requires.

## Outputs present (§5.4)

| Path | Contents | Rows |
|------|----------|------|
| `outputs/baseline/decision_level_features.csv` | decision-level table with `q_ir` + AI features | 5,551 |
| `outputs/baseline/oof_q_ir_predictions.csv` | clean export `id, case_id, rater_id, y_true, predicted_probability, fold` | 5,551 |
| `outputs/baseline/baseline_q_ir_metrics.csv` | pooled metrics for `q_ir` | 1 |
| `outputs/baseline/calibration_note.txt` | calibration caveat | — |
| `outputs/baseline/roc_q_ir.png`, `pr_q_ir.png`, `calibration_q_ir.png` | figures | — |

## Baseline metrics

| Metric | Recomputed from export | `baseline_q_ir_metrics.csv` |
|--------|------------------------|------------------------------|
| AUROC | 0.7159 | 0.7159383 |
| AUPRC | 0.4158 | 0.4157643 |
| Brier | 0.2101 | 0.2101082 |
| log-loss | 0.6914 | 0.6913666 |

The export and the metric file agree; the metrics match
`outputs/compare/compare_summary.json` (`auroc_q_ir = 0.7159383`).

## Calibration caveat (§5.4)

Mean predicted `q_ir` is **0.3622** against an observed error frequency of
**0.1904** — the baseline systematically **overestimates** error probability by
~1.9×. This is exactly the caveat the plan raises in §5.4 ("`q_ir` can
systematically overestimate the observed error frequency"), and
`calibration_note.txt` warns not to treat raw probabilities as clinically
reliable. `q_ir` must be read as a **predictive score**, not a calibrated
probability.

## Result

Output 2 satisfies §3.2 and §5 on the current artifacts:

- exactly 5,551 `q_ir` predictions, one per case–rater decision;
- target `error-rating`, evaluated on all 5,551 decisions;
- `q_ir` derived only from the outer-OOF image-only probability and the rating;
- the three required output files are present and mutually consistent
  (AUROC 0.7159, AUPRC 0.4158, Brier 0.2101).

No action required by this step beyond the verification.

## Important caveats

- **Read-only.** No script was executed for the pipeline; values are recomputed
  from the committed artifacts.
- `q_ir` is **not** assumed calibrated (mean 0.3622 vs prevalence 0.1904); it is a
  ranking/score baseline.
- `q_ir` equals `ai_probability_rater_wrong` by construction, so the D feature and
  the baseline share the same quantity.

## How to re-run

```bash
# recompute baseline metrics from the export (instant, read-only)
.venv\\Scripts\\python.exe -c "import pandas as pd; from sklearn.metrics import roc_auc_score,brier_score_loss; d=pd.read_csv('outputs/baseline/oof_q_ir_predictions.csv'); print(roc_auc_score(d['y_true'],d['predicted_probability']), brier_score_loss(d['y_true'],d['predicted_probability']))"

# regenerate the baseline from the diagnostic OOF (fast)
.venv\\Scripts\\python.exe -u src/build_baseline.py --config config.yaml
```

