# Step 7 — Comparison q_ir vs D with Case-Level Paired Bootstrap

**Date:** 2026-10-01
**Status:** Complete (verification only; no re-training performed)
**Scope:** `Implementation Plan.md` §17 (comparison script), §19 (uncertainty via
case-level bootstrap, paired Δ = D − q_ir)

---

## What we did

Verified the comparison and its uncertainty estimation from code + artifacts:

1. Confirmed the baseline `q_ir` and D OOF are **paired** on the same decisions
   (`load_aligned_scores`, 5,551 rows, `error-rating` consistency checked).
2. Inspected `bootstrap_paired_deltas`: case-level resampling
   (`case_bootstrap_indices`), n = 1,000, seed = 42, CIs via `np.quantile(0.025/0.975)`.
3. **Independently reproduced** the bootstrap CI of ΔAUROC with
   `case_bootstrap_indices` and compared to the artifact.
4. Inspected the comparison artifacts and the reference numbers.

## Why

The plan requires deciding **on the same 5,551 decisions** whether D beats the
image-only baseline, with uncertainty that respects the case grouping (§19):
decisions of one case must not be treated as independent. A paired Δ with a
case-level bootstrap CI is the required evidence.

## Commands run

```bash
# independent reproduction of the ΔAUROC bootstrap CI (read-only)
.venv\\Scripts\\python.exe -  # load_aligned_scores + case_bootstrap_indices(seed=42)
```

## Evidence — pairing (§17)

```
aligned rows: 5551   cases: 427
point delta_auroc: -0.083209
```

`load_aligned_scores` inner-joins `q_ir` (from
`outputs/baseline/decision_level_features.csv`) with `d_probability` (from
`outputs/D/oof_predictions.csv`) on `id`, requires exactly 5,551 rows, and raises
if `error-rating` disagrees between the two sources. The point ΔAUROC matches
`compare_summary.json` (`delta_auroc = -0.0832086`).

## Evidence — case-level bootstrap (§19)

```
replicates: 1000
reproduced delta_auroc CI95: -0.115514 .. -0.050914
csv delta CI95:              -0.115514 .. -0.050914
```

The independent reproduction (resampling `case_id` with replacement, all 13 rows
of each drawn case together, seed 42, 1,000 replicates, 2.5/97.5 percentiles)
reproduces the artifact's `delta_ci95_low`/`delta_ci95_high` for AUROC **exactly**.
This confirms the bootstrap is case-level (not row-level) and deterministic.

`outputs/compare/weighted/bootstrap_paired_deltas.csv` has 14 metric rows
(`auroc, auprc, brier, bss, log_loss, calibration_slope, calibration_intercept,
ece, precision@0.5, recall@0.5, specificity@0.5, f1@0.5, accuracy@0.5,
balanced_accuracy@0.5`), each with baseline, D and Δ point values and their
95 % CIs.

## Comparison metrics (paired, `error-rating`, n = 5,551)

Δ = D − q_ir (negative = D lower). Weighted D:

| Metric | q_ir | D | Δ = D − q_ir | Δ CI95 |
|--------|------|---|--------------|--------|
| AUROC | 0.7159 | 0.6327 | **−0.0832** | [−0.1155, −0.0509] |
| AUPRC | 0.4158 | 0.3247 | **−0.0911** | [−0.1491, −0.0502] |
| Brier | 0.2101 | 0.1520 | **−0.0581** | [−0.0826, −0.0344] |
| BSS | −0.3629 | +0.0139 | +0.3768 | (in CSV) |
| log-loss | 0.6914 | 0.5070 | −0.1844 | (in CSV) |
| precision@0.5 | 0.3494 | 0.4945 | +0.1451 | (in CSV) |
| recall@0.5 | 0.6093 | 0.1277 | −0.4816 | (in CSV) |
| specificity@0.5 | 0.7332 | 0.9693 | +0.2361 | (in CSV) |

Interpretation (the key result of the project):

- **Discrimination:** D is **significantly worse** than `q_ir` on AUROC and AUPRC —
  Δ < 0 with the 95 % CI excluding 0. The image-only baseline ranks better.
- **Probabilistic quality:** D is **significantly better** on Brier (and BSS: q_ir
  is negative, i.e. worse than the base rate, while D ≈ 0) — Δ < 0 with CI
  excluding 0. So D carries a calibration advantage even though its ranking is weaker.

This is exactly the complementarity the plan/report describe: `q_ir` wins on
ranking, D wins on probability quality.

## Other artifacts present (§21)

- `outputs/compare/weighted/` and `outputs/compare/unweighted/`: each with
  `bootstrap_paired_deltas.csv`, `summary.json`, `metrics_point.csv`,
  `operating_point_rowwise_youden.json`, `decision_curve.csv`/`.png`,
  `calibration_note.txt`, `dca_note.txt`, `roc_*.png`, `pr_*.png`,
  `calibration_*.png`, `roc_overlay.png`.
- `outputs/compare/compare_summary.json` — summary for weighted + unweighted.
- `outputs/compare/unified_D_metrics_table.csv` — D weighted vs unweighted, 9 metrics.
- `outputs/compare/nested_vs_fast_comparison.csv` — nested vs fast (declared).

Unweighted D and nested-vs-fast reference numbers:

| Variant | AUROC | AUPRC | Brier |
|---------|-------|-------|-------|
| D nested weighted (primary) | 0.6327 | 0.3247 | 0.1520 |
| D nested unweighted | 0.6516 | 0.3383 | 0.1577 |
| D fast (declared, optimistic) | 0.7028 | 0.3888 | 0.1556 |

## Result

The comparison satisfies §17 and §19:

- paired `q_ir` vs D on the same 5,551 decisions (labels cross-checked);
- **case-level** bootstrap, n = 1,000, seed 42 — CI reproduced **exactly**;
- paired Δ reported for 14 metrics with 95 % CIs.

Outcome: `q_ir` is better on discrimination (AUROC/AUPRC), D is better on
Brier/BSS — both differences statistically distinguishable from 0. The D AUROC/AUPRC
gap versus the plan benchmark is carried into **Step 9**.

## Important caveats

- **Read-only.** No script of the pipeline was executed; values come from the
  committed artifacts.
- Point metrics use threshold 0.5; a Youden operating point (mean of D fold
  thresholds) and a row-wise operating point are also recorded in
  `metrics_point.csv` / `operating_point_rowwise_youden.json`.
- CIs are the **percentile** bootstrap (2.5 / 97.5), matching the code and the
  independent reproduction.
- DCA (`decision_curve.csv`) is explicitly labelled descriptive, not a clinical
  decision rule.

## How to re-run

```bash
# full paired comparison + case bootstrap (fast)
.venv\\Scripts\\python.exe -u src/compare_D_vs_baseline.py --config config.yaml --n-bootstrap 1000
```

