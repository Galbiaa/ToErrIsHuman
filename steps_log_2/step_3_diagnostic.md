# Step 3 — Output 1: Image-Only Diagnostic (TARGET)

**Date:** 2026-10-01
**Status:** Complete (verification only; no re-training performed)
**Scope:** `Implementation Plan.md` §3.1 (diagnostic target), §4 (Output 1), §20 (reference benchmarks)

---

## What we did

Verified Output 1 — the image-only diagnostic stage — from the committed
artifacts and code, without re-training:

1. Loaded `outputs/diagnostic/oof_predictions.csv` and independently recomputed
   AUROC and Brier at case level; checked columns, coverage, target binarity and
   probability range.
2. Inspected the model architecture (`src/models.py`) against the plan's
   requirements (§4.1).
3. Confirmed the outer-CV flow in `src/train_diagnostic.py` (§4.2): train on
   outer-train only, predict the held-out fold, save one prediction per case,
   select the threshold on validation.
4. Compared the observed case-level metrics with the plan benchmarks (§20).

## Why

Output 1 is the single source of `p_i` for **both** the image-only baseline
`q_ir` (Output 2) and the D image features (Output 3). If its target, its
case-level coverage or its outer-OOF discipline were wrong, everything
downstream would be invalid. The plan fixes three things: target = `TARGET`,
exactly one outer-OOF prediction per case, and reference metrics AUROC ≈ 0.759 /
Brier ≈ 0.201.

## Commands run

```bash
# independent recompute (no writes)
.venv\\Scripts\\python.exe -  # loads outputs/diagnostic/oof_predictions.csv, recomputes AUROC/Brier
```

## Evidence — OOF artifact (independent recompute)

```
columns: ['case_id', 'target', 'ai_probability_class_1', 'fold', 'threshold_youden']
rows: 427   unique cases: 427
folds: [1, 2, 3, 4, 5]
each case once: True
target values: [0, 1]
prob range: 0.0002 .. 0.9998
nan probs: 0
recomputed AUROC: 0.7671
recomputed Brier: 0.2101
mean predicted prob: 0.4535   observed prevalence: 0.4473
```

The recomputed AUROC/Brier match `outputs/diagnostic/metrics_pooled.csv`
exactly (`auroc = 0.767060...`, `brier = 0.210108...`), so the pooled file and the
OOF file are mutually consistent. Mean predicted probability (0.4535) is close
to the observed prevalence (0.4473), i.e. no gross miscalibration at the case
level.

## Architecture (§4.1)

| Aspect | Implementation |
|--------|----------------|
| Model | `DiagnosticNet` — `TripleImageEncoder` = shared **ResNet-18** |
| Views | axial / coronal / sagittal (3 views), shared weights |
| Aggregation | `concat` → 1536-d embedding |
| Head | LayerNorm → Linear(1536→256) → ReLU → Dropout(0.35) → Linear(256→1) logit |
| Target | `TARGET` (case-level) |
| Config | `images.pretrained: true`, `image_size 224`, `hidden_dim 256`, `dropout 0.35` |

Consistent with the plan's note (§4.1) that the architecture should remain the
existing tri-view ResNet-18 unless a code-level incompatibility is found — none
was found.

## Training / validation flow (§4.2)

Per outer fold `k = 1..5` (`src/train_diagnostic.py`):

1. `test_df` = cases with `fold == k`; `train_pool` = cases with `fold != k`.
2. `assert_case_disjoint(train_pool, test_df)` — no overlap.
3. Inner split of `train_pool` into train/val (StratifiedShuffleSplit, val 20%,
   stratified by `TARGET`), `assert_case_disjoint(tr, va)`.
4. Early stopping / model selection on **validation AUROC** only.
5. Predict the held-out fold **once**; save `diagnostic_fold_{k}.pt`.
6. Threshold = Youden on **validation** (`select_threshold_on_validation`),
   never on the test fold.

The OOF file is assembled with exactly one row per case (`len(oof) == 427` and
no duplicated `case_id` are asserted in code).

Plan §4.3 specifies columns `case_id, TARGET, p_i_outer_oof, fold`; the artifact
uses `case_id, target, ai_probability_class_1, fold` (+ `threshold_youden`).
The **semantics are identical** — only the column names differ (cosmetic).

## Benchmark comparison (§20)

| Quantity | Plan benchmark | Observed | Δ |
|----------|----------------|----------|---|
| AUROC (427 cases, case-level) | ≈ 0.759 | **0.7671** | +0.008 |
| Brier (427 cases) | ≈ 0.201 | **0.2101** | +0.009 |

Both differences are ~0.01, i.e. the independent implementation **agrees** with
the reference within a small tolerance. Per §20 the observed values are reported;
no protocol change is warranted.

## Per-fold metrics (test @ 0.5)

| fold | n | AUROC | Brier |
|------|---|-------|-------|
| 1 | 86 | 0.8300 | 0.2165 |
| 2 | 86 | 0.8531 | 0.1543 |
| 3 | 85 | 0.8387 | 0.1938 |
| 4 | 85 | 0.7726 | 0.2046 |
| 5 | 85 | 0.7178 | 0.2820 |

(Brier is threshold-independent; the pooled AUROC/Brier above are the weighted
aggregate over the 427 cases.) Fold 5 is the weakest fold (AUROC 0.7178 /
Brier 0.2820), consistent with the run-to-run spread of a 427-case dataset.

## Result

Output 1 satisfies §3.1 and §4 on the current artifacts:

- target = `TARGET`, binary (0/1);
- exactly **one outer-OOF prediction per case**, 427/427, folds 1–5;
- probabilities finite and in [0,1]; pooled metrics reproduce exactly from the
  OOF file;
- architecture and validation-only model/threshold selection match the plan;
- case-level metrics (AUROC 0.7671, Brier 0.2101) are within ~0.01 of the plan
  benchmarks (0.759 / 0.201).

No action required by this step beyond the verification.

## Important caveats

- **No re-training.** All numbers come from the committed OOF/checkpoints; the
  training script was read, not executed.
- The pooled AUROC/Brier are threshold-independent; the threshold column is the
  per-fold validation Youden value.
- Brier is marginally higher than the plan target (0.2101 vs 0.201) — recorded
  as a minor item to carry into **Step 9** (benchmark verification), not to be
  "fixed" by altering the CV protocol.
- Column names differ cosmetically from the plan's wording (§4.3).

## How to re-run

```bash
# recompute metrics from the existing OOF (instant, read-only)
.venv\\Scripts\\python.exe -c "import pandas as pd; from sklearn.metrics import roc_auc_score,brier_score_loss; d=pd.read_csv('outputs/diagnostic/oof_predictions.csv'); print(roc_auc_score(d['target'],d['ai_probability_class_1']), brier_score_loss(d['target'],d['ai_probability_class_1']))"

# full re-training (long, GPU recommended) — only if you intend to regenerate Output 1
.venv\\Scripts\\python.exe -u src/train_diagnostic.py --config config.yaml
```

