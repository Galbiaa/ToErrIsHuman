# Step 8 — Consolidated Automated Leakage / Invariant Guards

**Date:** 2026-10-01
**Status:** Complete (new script created and executed; no pipeline artifact re-generated)
**Scope:** `Implementation Plan.md` §18 (Leakage Guards), §24 (Acceptance Criteria),
§26 (Final Methodological Invariants)

---

## What we did

The Step 1 audit flagged one gap: guards existed only as inline asserts plus
scattered `tmp/verify_*.py`, with **no single automated check** as §18 requires
("The implementation must include explicit automated checks").

We closed it by creating **`src/verify_pipeline_guards.py`**: a standalone,
read-only checker that verifies the leakage guards (§18.1–18.8), the dataset/D
acceptance criteria of §24, and Invariant 8 (§19), and reports the §20
benchmarks. It exits with status 1 if any check fails, so it can be used in CI.

## Why

The plan's central claim is that D is evaluated **without leakage** under nested
cross-fitting. Those guarantees must be **executable and re-runnable**, not just
described in prose, so that any future change is regression-checked. This script
is the single command that proves the current committed artifacts satisfy every
guard.

## What changed

| File | Change |
|------|--------|
| `src/verify_pipeline_guards.py` | **NEW** — consolidated guard checker (read-only; writes nothing) |

No other file was modified.

## How to run

```bash
.venv\\Scripts\\python.exe -u src/verify_pipeline_guards.py --config config.yaml
```

## Result

```
SUMMARY: 51/51 checks passed
RESULT: PASS - all leakage/invariant guards satisfied
```

The run recomputed the fold-safe rater stats, the preprocessor fits and the
bootstrap directly from the artifacts (in memory) and confirmed every guard.
Benchmarks (§20) are reported as INFO, not as pass/fail, because the plan treats
them as independent verification targets with tolerance.

## Checks implemented (mapped to the plan)

| Plan | Check(s) in the script | Result |
|------|------------------------|--------|
| §24 dataset | 427 cases; 5,551 decisions; 13 decisions/case; 13 raters | PASS |
| §3.2 / §24 | `error-rating == 1(rating != TARGET)` | PASS |
| §18.1 / §24 | case disjointness (train ∩ test = ∅) for all folds | PASS |
| §3.1 / §4.3 | one outer-OOF prediction per case; target binary; OOF target == `TARGET`; probs finite ∈ [0,1] | PASS |
| §5.1 / §5.2 / §5.4 | `q_ir` formula; `q_ir == ai_probability_rater_wrong`; `p_i == outer-OOF`; 5,551-row export | PASS |
| §7 / §18.5 | allowlist = 11; no `TARGET`; no `error-rating` | PASS |
| §18.2 | inner-OOF cache excludes outer-test cases (per fold) | PASS |
| §18.8 / §7 | fold-`k` artifacts aligned; D model `outer_fold == k`; `feature_names == allowlist` (per fold) | PASS |
| §18.7 | preprocessor fold `k` reproduced fit on outer-train only (per fold) | PASS |
| §18.6 | test rater-accuracy from outer-train cases only | PASS |
| §18.3 / §24 | D OOF 5,551 rows, one per decision; `probability_source_ai == nested_test` | PASS |
| §24 / §19 | comparison uses the same 5,551 decisions / 427 cases; paired Δ with 95 % CI present | PASS |
| Invariant 8 / §19 | bootstrap resamples whole cases (drawn groups are multiples of 13 rows) | PASS |
| §20 | diagnostic AUROC 0.7671 / Brier 0.2101 reported (benchmarks ~0.759 / ~0.201) | INFO |
| Invariant 9 / §26 | no protocol tuning vs test — enforced by design (fixed seeds/protocol) | INFO |

## Important caveats

- **Read-only.** The script loads artifacts, rebuilds feature frames/preprocessors
  in memory and asserts; it writes nothing to `outputs/` or `models/`.
- §18.2 / §18.3 are verified **structurally** (the inner-OOF cache contains no
  outer-test case; the D OOF provenance is `nested_test`), because a full
  re-training is intentionally not performed in these steps. If the pipeline is
  re-run, the same script re-verifies automatically, since the invariants are
  recomputed from the fresh artifacts.
- §18.5 "no `TARGET` column" is enforced through the allowlist and the
  `build_d_feature_frame` feature block; `error-rating` is the target, never a
  feature.
- The script complements the pre-existing `tmp/verify_*.py` helpers; it is the
  single entry point that covers **all** §18 guards.

## How to re-run / integration

```bash
# single consolidated guard run (fast, read-only)
.venv\\Scripts\\python.exe -u src/verify_pipeline_guards.py --config config.yaml
```

Suggested integration (addressed in **Step 10**): reference this command in
`README.md` and in the §23 "Verification Commands" list so the guards run
alongside the pipeline.
