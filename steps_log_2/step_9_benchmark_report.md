# Step 9 — Reference Benchmark Verification and Discrepancy Report

**Date:** 2026-10-01
**Status:** Complete (verification + report; no protocol change, no re-training)
**Scope:** `Implementation Plan.md` §20 (Reference Benchmarks), §23 (Verification
Commands), §24 (Acceptance Criteria), §26 (Invariants 9)

---

## What we did

Compared every quantity reported by the pipeline against the plan's §20 reference
benchmarks and produced a discrepancy report, explicitly **without** altering the
protocol (§20, §24). We also checked that the pipeline reproduces the numbers the
plan itself attributes to the existing report.

## Why

§20 states that the reference values are "reference/unit-test targets, not
mandatory exact values", and that any discrepancy must be **investigated** in data
processing, architecture, preprocessing, seeds, hyperparameters, folds or feature
construction — never by replacing nested cross-fitting with the faster, optimistic
protocol (§20 bullet list; Invariant 9).

## Benchmark vs observed

### Output 1 — image-only-with-TARGET (427 cases)

| Metric | §20 benchmark | Observed | Δ |
|--------|---------------|----------|---|
| AUROC | ≈ 0.759 | **0.7671** | +0.0081 |
| Brier | ≈ 0.201 | **0.2101** | +0.0091 |

Both within ~0.01 → **agreement**.

### Output 3 — Solution D (5,551 decisions)

| Variant | AUROC | Δ vs 0.714 | AUPRC | Δ vs 0.431 | Brier | Δ vs 0.154 |
|---------|-------|-----------|-------|-----------|-------|-----------|
| §20 benchmark | 0.714 | — | 0.431 | — | 0.154 | — |
| **D nested weighted (primary)** | **0.6327** | **−0.0813** | **0.3247** | **−0.1063** | **0.1520** | −0.0020 |
| D nested unweighted | 0.6516 | −0.0624 | 0.3383 | −0.0927 | 0.1577 | +0.0037 |
| D fast (declared optimistic) | 0.7028 | −0.0112 | 0.3888 | −0.0422 | 0.1556 | +0.0016 |

The image-only numbers match; the **D benchmark does not** match the primary
(nested) protocol, and is closest to the **fast** protocol (Brier already matches
under all variants).

## The pipeline reproduces the report's own numbers exactly

§20 records the existing report's nested and fast results. The current artifacts
reproduce them **to 4 decimals**:

| Quantity | §20 "existing report" | This pipeline | Match |
|----------|-----------------------|---------------|-------|
| D nested AUROC / AUPRC / Brier | 0.6327 / 0.3247 / 0.1520 | 0.6327 / 0.3247 / 0.1520 | ✅ |
| D fast AUROC / AUPRC / Brier | 0.7028 / 0.3888 / 0.1556 | 0.7028 / 0.3888 / 0.1556 | ✅ |
| AUROC difference fast − nested | ≈ +0.07 | +0.0701 | ✅ |

So the independent implementation is **consistent with the documented run**; the
gap is between the *reference benchmark* and the *nested protocol result*, not
between this implementation and the report.

## Discrepancy analysis

**Image-only (Output 1):** agreement (~0.01). No issue.

**Solution D (Output 3): the one real discrepancy.**

- The benchmark (AUROC 0.714) sits far from the primary **nested** result
  (0.6327, Δ = −0.081) and very close to the **fast** result (0.7028, Δ = −0.011).
- The fast − nested AUROC gap is **+0.0701**, matching the plan's own documented
  "≈ +0.07" (§20). The fast protocol is explicitly the *optimistic* one
  (outer-train `p_i` are in-sample); the nested protocol is the leakage-free one.
- Reading: the reference D values appear to have been produced under an
  **optimistic (fast-like) construction**, not under the leakage-free nested
  protocol. The independent nested implementation therefore cannot reach 0.714
  without weakening the anti-leakage controls — which §20 forbids.

## Hypotheses to investigate (per §20), ranked by likely impact

| # | Area | Assessment |
|---|------|------------|
| 1 | **Protocol / leakage** | Explains most of the gap: nested is conservative (inner-OOF train p_i, fold-safe rater profiles, outer-test p_i from a test-blind model). The benchmark matches the optimistic fast variant. |
| 2 | **Feature construction (p_i for train)** | Using inner-OOF `p_i` (not in-sample / not global OOF) lowers the D AUROC; a global-OOF or in-sample `p_i` would inflate it toward the benchmark. |
| 3 | **Rater profiles** | Fold-safe LOO (case excluded) is more conservative than Excel global / full-data rater-accuracy. |
| 4 | Preprocessing / architecture / hyperparameters / seeds / folds | Low likelihood: the pipeline reproduces the *documented* nested and fast numbers to 4 decimals, so these cannot explain the benchmark gap. |

## What we deliberately did NOT do (Invariant 9, §20)

- **Did not** replace nested cross-fitting with fast OOF construction to approach 0.714.
- **Did not** tune the protocol, folds, seeds or thresholds against the test result.
- **Did not** reuse outer-test labels / test-derived statistics in any D feature.
- Reported the **actual** nested result (0.6327) instead.

## Result

| Output | Benchmark | Observed (primary) | Verdict |
|--------|-----------|--------------------|---------|
| Image-only AUROC | ≈ 0.759 | 0.7671 | agreement (~0.01) |
| Image-only Brier | ≈ 0.201 | 0.2101 | agreement (~0.01) |
| D AUROC | ≈ 0.714 | 0.6327 (nested) | **below benchmark**, matches documented nested run |
| D AUPRC | ≈ 0.431 | 0.3247 (nested) | **below benchmark**, matches documented nested run |
| D Brier | ≈ 0.154 | 0.1520 (nested) | agreement |

Conclusion: the pipeline reproduces the documented run exactly and reports its
true nested performance. The D benchmark is best explained by an optimistic
(fast-like) evaluation; per §20 it is treated as a **reference target that the
leakage-free protocol does not reach**, not as a value to chase by relaxing the
protocol. The nested result (AUROC 0.6327) stands as the primary answer.

## Important caveats

- **Read-only.** No protocol, fold, seed or artifact was changed; no re-training.
- Benchmarks are reference/unit-test targets with tolerance (§20), not pass/fail
  thresholds; the §20 lines are reported as INFO by `src/verify_pipeline_guards.py`.
- The `q_ir` baseline AUROC (0.7159) is close to the D benchmark value — a further
  signal that the benchmark may reflect a different (image-only-ish) construction
  than the leakage-free tabular D.

## How to re-run

```bash
# benchmark lines are reported by the consolidated guard script
.venv\\Scripts\\python.exe -u src/verify_pipeline_guards.py --config config.yaml

# reproduce the nested vs fast D comparison
.venv\\Scripts\\python.exe -c "import pandas as pd; print(pd.read_csv('outputs/compare/nested_vs_fast_comparison.csv').T)"
```

