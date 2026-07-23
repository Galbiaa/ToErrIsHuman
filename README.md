# ToErrIsHuman — pipeline a due stadi

Pipeline per prevedere l’**errore diagnostico umano** da MRI e valutazioni di rater:

1. **Diagnostico** (`diagnostic`): dalle tre MRI di un caso stima \(P(\texttt{TARGET}=1)\).
2. **Baseline** `q_ir`: trasforma quella probabilità in \(P(\text{errore del rater}\mid\text{immagini}, R)\).
3. **Soluzione D**: XGBoost su feature AI + contesto rater (fold-safe) → \(P(\texttt{error-rating}=1)\).

**Dati attesi** (path in `config.yaml`):

- `data/IMAGE-GROUND-TRUTH.xlsx`
- `data/USER-INFOS-CORRECTED.xlsx`
- `data/images/` — file `case{i}axial.jpg`, `case{i}coronal.jpg`, `case{i}sagittal.jpg`

Unità: diagnostico = **caso** (427); D / confronti = **decisione** caso–rater (5551). Fold e bootstrap lavorano per **`case_id`**.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Configurazione: [`config.yaml`](config.yaml).

## Esecuzione (ordine)

```bash
python -u src/validate_dataset.py --config config.yaml
python -u src/generate_folds.py --config config.yaml
python -u src/train_diagnostic.py --config config.yaml          # lungo; GPU consigliata
python -u src/build_baseline.py --config config.yaml
python -u src/build_D_features.py --config config.yaml --smoke  # opzionale
python -u src/train_D.py --config config.yaml --protocol nested # molto lungo
python -u src/compare_D_vs_baseline.py --config config.yaml --n-bootstrap 1000
python -u src/analyze_errors.py --config config.yaml
python -u src/infer_new_case.py --config config.yaml --case-id 1
```

All’avvio degli script vengono aggiornati, se previsto, `outputs/run_config_snapshot.yaml` e `outputs/environment.txt`.

## Output

| Path | Contenuto |
|------|-----------|
| `outputs/folds/` | assegnazione fold |
| `outputs/diagnostic/` | OOF, metriche, curve |
| `models/diagnostic/` | checkpoint `diagnostic_fold_{k}.pt` |
| `outputs/baseline/` | `q_ir` e feature AI a livello decisione |
| `outputs/D/` | OOF D, profili rater, metriche |
| `models/D/` | `scenario_D_fold_{k}.joblib`, `preprocessor_fold_{k}.joblib` |
| `outputs/compare/` | confronto `q_ir` vs D, bootstrap, DCA |
| `outputs/analysis/` | analisi errori e demo inferenza |

## Note operative

- `TARGET` non va tra le feature di D (allowlist in `config.yaml`).
- In CV, `rater-accuracy` / `rater-confidence` sono ricalcolate fold-safe (non usare i globali Excel come input di training).
- Le soglie operative si scelgono sulla validation, non sul test.
- Il confronto ufficiale errore usa `q_ir` e D sullo stesso target `error-rating`.
- Il bootstrap del confronto ricampiona i casi, non le singole decisioni come indipendenti.

Guide aggiuntive (uso interno): cartella [`INFO/`](INFO/).
