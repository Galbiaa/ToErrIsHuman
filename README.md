# ToErrIsHuman — pipeline a due stadi

Pipeline per prevedere l’**errore diagnostico umano** da MRI e valutazioni di rater:

1. **Diagnostico**: dalle tre MRI di un caso stima `P(TARGET = 1)`.
2. **Baseline `q_ir`**: trasforma quella probabilità in stima image-only dell’errore del rater.
3. **Soluzione D**: XGBoost su feature AI + contesto rater (fold-safe) → `P(error-rating = 1)`.

**Dati** (path in [`config.yaml`](config.yaml)):

- `data/IMAGE-GROUND-TRUTH.xlsx` — un caso, un `TARGET`
- `data/USER-INFOS-CORRECTED.xlsx` — decisioni caso–rater
- `data/images/` — `case{i}axial.jpg`, `case{i}coronal.jpg`, `case{i}sagittal.jpg`

Unità: diagnostico = **caso** (427); D / confronti = **decisione** (5551). Fold e bootstrap per **`case_id`**.

## Setup

Testato con **Python 3.11** (run documentato: 3.11.9).

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Per le versioni esatte del run documentato: `pip install -r requirements-lock.txt`.

Su Windows, se `python` non punta al venv: `.\.venv\Scripts\python.exe -u src\...`.

**PyTorch e GPU.** Con sola CPU basta il comando sopra. Con una GPU NVIDIA conviene installare `torch`/`torchvision` da [pytorch.org](https://pytorch.org) con la build CUDA, poi `pip install -r requirements.txt`. Controllo rapido: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. Il training diagnostico e il nested di D sono lunghi: **GPU consigliata**.

## Pipeline: script e ordine

Eseguire **in quest’ordine**: ogni passo usa gli artefatti del precedente.

| # | Script | Cosa fa | Perché dopo il precedente |
|---|--------|---------|---------------------------|
| 1 | `validate_dataset.py` | Controlla immagini, Excel, join `error-rating` ↔ `TARGET` | Evita di addestrare su dati inconsistenti |
| 2 | `generate_folds.py` | Assegna i 427 casi a 5 fold (stratificati) | Una sola suddivisione per tutta la pipeline |
| 3 | `train_diagnostic.py` | Addestra ResNet-18 a 5 fold; salva OOF e checkpoint | Serve `p_i` per baseline e D |
| 4 | `build_baseline.py` | Espande OOF × 13 rater; calcola `q_ir` e feature AI | Baseline sullo stesso target di D (`error-rating`) |
| 5 | `build_D_features.py` | (opz. `--smoke`) verifica feature D fold-safe | Controlla il builder prima del nested lungo |
| 6 | `train_D.py` | Nested CV + XGBoost (± weighting); salva OOF D e joblib | Usa fold, OOF/diagnostico e feature fold-safe |
| 7 | `compare_D_vs_baseline.py` | Metriche appaiate + bootstrap per caso + DCA | Confronta `q_ir` vs D sulle stesse 5551 decisioni |
| 8 | `analyze_errors.py` | Analisi accordo AI–rater, sottogruppi, stabilità fold | Dopo il confronto aggregato |
| 9 | `infer_new_case.py` | Ensemble fold-allineato su un caso | Richiede modelli già salvati |

```bash
python -u src/validate_dataset.py --config config.yaml
python -u src/generate_folds.py --config config.yaml
python -u src/train_diagnostic.py --config config.yaml          # lungo; GPU consigliata
python -u src/build_baseline.py --config config.yaml
python -u src/build_D_features.py --config config.yaml --smoke  # opzionale
python -u src/train_D.py --config config.yaml --protocol nested # molto lungo
python -u src/compare_D_vs_baseline.py --config config.yaml --n-bootstrap 1000
python -u src/analyze_errors.py --config config.yaml
```
## Metodologia e verifica metodologica

Questa sezione documenta, come richiesto dall'`Implementation Plan.md` (§25), la
domanda scientifica, i target, i tre output, il protocollo e i benchmark.

### Domanda scientifica

Il progetto studia se sia possibile **prevedere l'errore diagnostico umano** a
partire dalle **MRI** e dal **contesto del rater**, separando nettamente ciò che
è "solo immagine" da ciò che sfrutta informazione aggiuntiva sul rater. Non è un
modello clinico: le probabilità non sono percentuali cliniche (vedi calibrazione).

### `TARGET` vs `error-rating`

- **`TARGET`** = verità diagnostica **di caso** (da `IMAGE-GROUND-TRUTH.xlsx`); è
  il label dello **stadio 1** (image-only).
- **`error-rating`** = `1` **se e solo se** `rating != TARGET`; label a livello di
  **decisione** (caso–rater) usato da `q_ir` e da D.
- `TARGET` **non è mai** una feature di D.

### I tre output (unità diverse)

| # | Output | Unità | Predice | Artefatto principale |
|---|--------|-------|---------|----------------------|
| 1 | Image-only diagnostico | caso (427) | `TARGET` | `outputs/diagnostic/oof_predictions.csv` |
| 2 | Baseline image-only `q_ir` | decisione (5551) | `error-rating` | `outputs/baseline/oof_q_ir_predictions.csv` |
| 3 | Soluzione D | decisione (5551) | `error-rating` | `outputs/D/oof_predictions.csv` |

### Definizione di `q_ir`

Per ogni decisione `(i, r)`, dato `p_i = P(TARGET = 1 | immagini del caso i)` e il
rating del rater `R`:

```text
q_ir = p_i            se R = 0
q_ir = 1 - p_i        se R = 1     # = P(Y != R | immagini)
```

`q_ir` è uno **score**, non una probabilità calibrata (sovrastima la frequenza di
errore). Coincide con la feature di D `ai_probability_rater_wrong`.

### Allowlist delle feature di D (11)

`rating-confidence, case-difficulty, rater-expertise, rater-accuracy,
rater-confidence, rating, ai_probability_class_1, ai_predicted_class,
rater_ai_disagreement, ai_probability_rater_wrong, ai_margin`

`rating-accuracy`/`rating-confidence` = profili **fold-safe** (non i globali Excel).
`TARGET` **e** `error-rating` sono esclusi dalle feature.

### Protocollo nested cross-fitting (D)

Per ogni fold esterno `k`:

- `p_i` dei casi **outer-train** = OOF di una **CV interna** (tag `nested_oof`);
- `p_i` dei casi **outer-test** = predizioni di un modello diagnostico addestrato
  **solo** sui casi outer-train (tag `nested_test`);
- feature D + preprocessor (mediana + standardize) fit **solo** su outer-train;
- soglia operativa (Youden) scelta su uno split di **validation** dell'outer-train,
  **mai** sul test;
- XGBoost, predizione del test **una sola volta**.

Il protocollo **`fast`** (feature OOF in-sample dichiarate) è una variante
**ottimistica** secondaria: **non** è l'analisi principale.

### Profili rater fold-safe

`rater-accuracy`/`rater-confidence` per ogni decisione si calcolano **solo** dai
casi di training del fold:

- riga di **training** → leave-one-out (il caso corrente escluso);
- riga di **test** → media sui soli casi outer-train.

I globali `rater-accuracy`/`rater-confidence` dell'Excel **non** entrano in CV.

### Bootstrap per caso

L'incertezza del confronto `q_ir` vs D usa un bootstrap che **ricampiona i
`case_id`** (tutte le 13 decisioni di un caso insieme), 1000 repliche; Δ = D −
`q_ir` con IC 95%.

### Benchmark di riferimento

| Output | Metrica | Benchmark |
|--------|---------|-----------|
| Image-only (`TARGET`) | AUROC | ≈ 0.759 |
| Image-only (`TARGET`) | Brier | ≈ 0.201 |
| Soluzione D | AUROC | ≈ 0.714 |
| Soluzione D | AUPRC | ≈ 0.431 |
| Soluzione D | Brier | ≈ 0.154 |

I benchmark sono **target di verifica indipendenti** e **non devono essere
raggiunti indebolendo il protocollo anti-leakage** (nested cross-fitting, profili
rater fold-safe, soglie su validation, bootstrap per caso). Eventuali discrepanze
si indagano in dati, architettura, preprocessing, seed, fold o costruzione delle
feature — **non** cambiando il protocollo né tunando sul test. Nel run attuale i
numeri image-only sono in accordo (~0.01) col benchmark, mentre il valore D del
benchmark è più vicino alla variante **fast** che al nested primario; il risultato
nested reale (`outputs/compare/compare_summary.json`) è quello riportato.

### Soluzione D vs modello end-to-end multimodale

Un modello **end-to-end multimodale** eventualmente presente nel repo è un
**esperimento separato**: non va chiamato né interpretato come Soluzione D. La
pipeline e la verifica primarie riguardano i **tre output** (image-only, `q_ir`, D).

### Verifica automatica delle guardie

```bash
python -u src/verify_pipeline_guards.py --config config.yaml
```

Verifica tutti gli invarianti anti-leakage (§18), i criteri di accettazione (§24) e
riporta i benchmark (§20); esce con codice ≠ 0 se un controllo fallisce.

## Web App Interattiva (Streamlit)

È disponibile un'interfaccia interattiva a più pagine per eseguire l'inferenza fold-safe su casi del dataset o nuove immagini MRI, ispezionare le feature e consultare i benchmark:

```bash
streamlit run web/Home.py
```

Pagine incluse:
- **Home**: Panoramica della pipeline, stato degli artefatti nei 5 fold e configurazione di runtime.
- **Inferenza su un caso**: Selezione caso da dataset o upload tri-planare (assiale, coronale, sagittale), editor what-if del contesto clinico del rater, inferenza ensemble 5-fold, confronto con baseline $q_{ir}$, visualizzazione viste MRI ed esportazione risultati (JSON/CSV).
- **Prestazioni**: Tabelle riassuntive out-of-fold su 5.551 decisioni, curve ROC/PR/Calibrazione e quantificazione del leakage del protocollo (Nested vs Fast).
- **Metodo e limiti**: Documentazione approfondita sulle guardie anticontaminazione e avvertenze cliniche su calibrazione e soglie operative.



`outputs/` e `models/` (incluse le sottocartelle) sono create automaticamente dagli script: non serve pre-crearle. I primi passi aggiornano `outputs/run_config_snapshot.yaml` e `outputs/environment.txt`.

### Moduli di supporto (non si lanciano da soli)

| Modulo | Ruolo |
|--------|--------|
| `data.py` | Caricamento Excel / tabella casi diagnostici |
| `models.py` | `DiagnosticNet` (ResNet-18 a tre viste) |
| `metrics.py` | Metriche, calibrazione, plot |
| `methodology_guards.py` | Anti-leakage: fold-safe rater, soglia su val, bootstrap per caso, … |
| `export_oof_predictions.py` | Export OOF riga per riga (`q_ir` e D) |
| `reproducibility.py` | Seed e snapshot ambiente |

### Controlli di verifica (`tmp/`)

| Script | Cosa verifica | Comando |
|---|---|---|
| `verify_predict_proba_class_order.py` | `predict_proba[:, 1]` = `P(error-rating = 1)` sui 20 modelli D (riscrive `outputs/D/predict_proba_class_verification.csv`) | `python -u tmp/verify_predict_proba_class_order.py --config config.yaml` |
| `verify_q_ir_and_rater_stats.py` | struttura dei dati, profili rater fold-safe (leave-one-out vs leave-one-case-out), evidenza su `q_ir` | `python -u tmp/verify_q_ir_and_rater_stats.py` |
| `src/verify_pipeline_guards.py` | **tutti** gli invarianti anti-leakage (§18) + criteri di accettazione (§24); riporta i benchmark (§20) | `python -u src/verify_pipeline_guards.py --config config.yaml` |

Il secondo non accetta argomenti: legge `config.yaml` dalla root del progetto e richiede `outputs/folds/`, `outputs/diagnostic/oof_predictions.csv` e, per le importanze, `models/D/`.


## Inferenza su un caso

Prerequisiti: setup fatto; presenti `models/diagnostic/diagnostic_fold_1..5.pt`, `models/D/scenario_D_fold_*.joblib`, `models/D/preprocessor_fold_*.joblib`, e `outputs/folds/case_fold_assignment.csv` (se hai ricevuto solo il codice, esegui prima i passi 1–6).

Per ogni fold `k` = 1…5: diagnostico_k → `p_i` → feature → preprocessor_k + D_k; poi **media** delle cinque probabilità di errore. I pezzi dello stesso fold restano abbinati. **`TARGET` non è un input** di D.

### Demo (caso già nel dataset)

```bash
python -u src/infer_new_case.py --config config.yaml --case-id 1
# opzionale: --rater-id 0
# opzionale: --out path/to/out.json
```

Output di default: `outputs/analysis/infer_case_1.json` (probabilità D ensemble per decisione caso–rater).

### Caso nuovo (con lo script attuale)

1. Aggiungere le tre MRI in `data/images/`:  
   `case{ID}axial.jpg`, `case{ID}coronal.jpg`, `case{ID}sagittal.jpg`
2. Aggiungere una riga in `IMAGE-GROUND-TRUTH.xlsx` con `CASE-ID={ID}` (lo script carica la tabella casi da lì).
3. Aggiungere in `USER-INFOS-CORRECTED.xlsx` le righe `id={ID}-{rater}` con almeno:  
   `rating`, `rating-confidence`, `case-difficulty`, `rater-expertise`  
   (e le altre colonne dello schema USER se presenti; `error-rating` può essere placeholder se non noto).
4. Eseguire:

```bash
python -u src/infer_new_case.py --config config.yaml --case-id {ID}
```

## Output principali

| Path | Contenuto |
|------|-----------|
| `outputs/folds/` | assegnazione fold |
| `outputs/diagnostic/` | OOF, metriche, curve |
| `models/diagnostic/` | checkpoint `diagnostic_fold_{k}.pt` |
| `outputs/baseline/` | `q_ir` e feature AI |
| `outputs/D/` | OOF D, profili rater, metriche |
| `models/D/` | `scenario_D_fold_{k}.joblib`, `preprocessor_fold_{k}.joblib` |
| `outputs/compare/` | confronto `q_ir` vs D, bootstrap, DCA |
| `outputs/analysis/` | analisi errori e JSON di inferenza |

Controlli rapidi a fine run: `outputs/diagnostic/metrics_pooled.csv` (AUROC del diagnostico) e `outputs/compare/compare_summary.json` (AUROC e Brier di `q_ir` e D).

## Note

- `TARGET` non va tra le feature di D (allowlist in `config.yaml`).
- In CV, `rater-accuracy` / `rater-confidence` sono ricalcolate fold-safe (non i globali Excel).
- Soglie operative solo su validation, non sul test.
- Confronto ufficiale errore: `q_ir` vs D su `error-rating`.
- Bootstrap del confronto: ricampiona i **casi**, non le 5551 righe come indipendenti.
