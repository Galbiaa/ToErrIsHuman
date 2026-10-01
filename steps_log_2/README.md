# steps_log_2 — Realignment of the Two-Stage Pipeline

Working index for the step-by-step execution of `../Implementation Plan.md`.

Each completed step produces one log file in this folder, in the same style as
`../STEP_LOGS/` (Date, Status, What we did, Why, What changed, Result,
Important caveats, How to re-run).

> Status of this index: **draft, pending user confirmation of the breakdown.**

## What the plan asks for (summary)

Realign the two-stage pipeline with the original problem formulation and
independently verify its outputs. Three distinct outputs must all be produced
and verified:

1. **Image-only diagnostic** — predict `TARGET` from MRI, one prediction per
   case (n = 427).
2. **Image-only error baseline `q_ir`** — `p_i` combined with each rater rating,
   one prediction per case-rater decision (n = 5,551).
3. **Solution D** — predict `error-rating` from image + rater context, one
   prediction per case-rater decision (n = 5,551); `TARGET` never an input.

Primary D protocol = **nested cross-fitting** (inner-OOF for outer-train rows,
outer-test diagnostic model for outer-test rows). Fold-safe rater profiles.
Case-level bootstrap (1000 replicates). Reference benchmarks are independent
verification targets and must NOT be met by weakening the anti-leakage protocol.

## Proposed step breakdown (to confirm)

| Step | Title | Plan sections |
|------|-------|---------------|
| 1 | Audit current pipeline vs plan (gap analysis) | §1, §17, §21, §26 |
| 2 | Dataset & fold integrity checks (427 / 5,551 / 13, case-level, disjointness) | §2, §18.1, §24 |
| 3 | Output 1: image-only diagnostic (`TARGET`, outer-OOF, benchmark) | §3.1, §4 |
| 4 | Output 2: image-only baseline `q_ir` | §3.2, §5 |
| 5 | D feature allowlist + fold-safe rater profiles (inner-OOF wiring) | §7, §10, §11, §18.4-18.7 |
| 6 | Nested D training + preprocessing isolation + pooled OOF | §8, §9, §12, §13, §18.8 |
| 7 | Comparison + case-level paired bootstrap (Δ = D − q_ir) | §17, §19 |
| 8 | Leakage guards as automated checks (all invariants) | §18, §26 |
| 9 | Reference benchmark verification + discrepancy report | §20, §23, §24 |
| 10 | README documentation update | §25 |

## Logs

_(Appended as each step completes.)_

- `step_1_audit.md`
- `step_2_data_folds.md`
- `step_3_diagnostic.md`
- `step_4_baseline.md`
- `step_5_d_features_rater.md`
- `step_6_nested_d.md`
- `step_7_comparison_bootstrap.md`
- `step_8_guard_checks.md`
- `step_9_benchmark_report.md`
- `step_10_readme.md`
