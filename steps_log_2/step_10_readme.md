# Step 10 — README Documentation Update

**Date:** 2026-10-01
**Status:** Complete (documentation only; no pipeline logic changed)
**Scope:** `Implementation Plan.md` §25 (README Documentation) and §23 (Verification Commands)

---

## What we did

Updated `README.md` to satisfy §25. We added a new section
**"Metodologia e verifica metodologica"** covering every required point, and
added the consolidated guard script to the verification-commands table (§23).

No code, config, artifact or protocol was changed.

## Why

§25 requires the README to document the scientific question, the `TARGET` vs
`error-rating` distinction, the three outputs, the definition of `q_ir`, the D
feature allowlist, the nested cross-fitting protocol, the fold-safe rater
profiles, the case-level bootstrap, the reference benchmarks, and the D vs
end-to-end-multimodal distinction. It must also state that the reference metrics
are **independent verification targets that must not be obtained by weakening the
anti-leakage protocol**.

## What changed

| File | Change |
|------|--------|
| `README.md` | Added `## Metodologia e verifica metodologica` (10 subsections) before the Web App section |
| `README.md` | Added a row for `src/verify_pipeline_guards.py` in the "Controlli di verifica" table |

## §25 checklist

| §25 item | README location |
|----------|-----------------|
| 1. Original scientific question | `### Domanda scientifica` |
| 2. `TARGET` vs `error-rating` | `### TARGET vs error-rating` |
| 3. Three pipeline outputs | `### I tre output (unità diverse)` |
| 4. Definition of `q_ir` | `### Definizione di q_ir` |
| 5. D feature allowlist | `### Allowlist delle feature di D (11)` |
| 6. Nested cross-fitting protocol | `### Protocollo nested cross-fitting (D)` |
| 7. Fold-safe rater profiles | `### Profili rater fold-safe` |
| 8. Case-level bootstrap | `### Bootstrap per caso` |
| 9. Reference benchmarks | `### Benchmark di riferimento` |
| 10. D vs end-to-end multimodal | `### Soluzione D vs modello end-to-end multimodale` |
| Explicit anti-leakage statement | `### Benchmark di riferimento` — "target di verifica indipendenti e non devono essere raggiunti indebolendo il protocollo anti-leakage" |
| §23 verification command | `### Verifica automatica delle guardie` + "Controlli di verifica" table |

## Result

`README.md` now documents all §25 points, in Italian (to match the existing file),
plus the consolidated guard command from Step 8. The README explicitly states the
anti-leakage rule and the role of the reference benchmarks as independent
verification targets.

## Important caveats

- **Documentation only.** No source file, config, artifact or protocol changed.
- The README is written in Italian to match the rest of the file; the reference
  benchmarks and the anti-leakage statement are reproduced verbatim in meaning
  from §20/§25.
- The end-to-end multimodal model is described as a **separate experiment** and is
  not to be confused with Solution D (§16).

## How to re-run

Nothing to execute (documentation). To re-check the guards referenced by the
README:

```bash
.venv\\Scripts\\python.exe -u src/verify_pipeline_guards.py --config config.yaml
```
