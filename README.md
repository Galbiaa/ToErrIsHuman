# Rater Error Prediction: tabular, image-only, and multimodal models

This project trains three families of models for predicting rater decision error.

The expected input table is `TRAINING-FINALE.xlsx`, with one row per rating/decision and the following columns:

```text
id, rating-confidence, case-difficulty, rater-expertise, rater-accuracy, rater-confidence, error-rating
```

The `id` column must follow the format `<case_id>-<rater_id>`, for example `1-0`, `1-1`, ..., `427-12`.

Images are expected in `data/images/`, with exactly this naming convention:

```text
case<d>axial.jpg
case<d>coronal.jpg
case<d>sagittal.jpg
```

For example:

```text
data/images/case1axial.jpg
data/images/case1coronal.jpg
data/images/case1saggital.jpg
```

## Recommended folder structure

```text
rater_error_prediction/
  config.yaml
  requirements.txt
  data/
    TRAINING-FINALE.xlsx
    images/
      case1axial.jpg
      case1coronal.jpg
      case1saggital.jpg
      ...
      case427axial.jpg
      case427coronal.jpg
      case427saggital.jpg
  src/
    data.py
    metrics.py
    models.py
    generate_folds.py
    train_tabular.py
    train_deep.py
    compare_models.py
    validate_dataset.py
    experiment_output.py
  outputs/
  models/
```

## Models implemented

### A. Tabular baselines

The script `src/train_tabular.py` trains:

1. Logistic regression
2. Random forest
3. XGBoost
4. Small MLP using scikit-learn

These models use only the tabular variables:

```text
rating-confidence
case-difficulty
rater-expertise
rater-accuracy
rater-confidence
```

The target is:

```text
error-rating
```

### B. Image-only model (primary: decision-level classifier)

The script `src/train_deep.py --mode image_only` trains a **decision-level binary classifier** on all **5,551** case–rater rows.

For each decision, the model receives the three images of the case. The target is the binary `error-rating` for that specific rater decision. `rater_id` is kept only as metadata and is **not** passed to the network.

The same three images appear 13 times per case with different targets. This is intentional: the same case can be rated correctly by some raters and incorrectly by others.

**Secondary analysis (legacy regression):** `--mode image_only_regression` trains on 427 case-level rows with target = mean `error-rating` across raters. Keep this separate from the main classifier experiment.

### C. Multimodal model

The script `src/train_deep.py --mode multimodal` trains a decision-level model using:

- the three images of the case;
- the rating-specific variables;
- the rater-level variables.

The target is the decision-level binary outcome `error-rating`.

## Cross-validation design

All models split data by `case_id`, not by individual row. This means that all 13 ratings for the same case are kept in the same fold. This avoids contamination between training and validation through shared images.

Generate the shared fold assignments once before any training:

```bash
python src/generate_folds.py --config config.yaml
```

This writes:

```text
outputs/folds/fold_assignments.csv
outputs/folds/fold_stats.csv
```

All training scripts read `outputs/folds/fold_assignments.csv` so tabular, image-only, and multimodal experiments use exactly the same folds.

## Data leakage prevention

The Excel columns `rater-accuracy` and `rater-confidence` are global rater statistics computed on the full dataset. Using them unchanged during cross-validation would leak test-fold information.

Before each fold, tabular and multimodal training recomputes these features from **training decisions only**:

- `rater-accuracy` = 1 - mean(`error-rating`) per rater on the training fold
- `rater-confidence` = mean(`rating-confidence`) per rater on the training fold

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PyTorch, it is often better to use the official installation selector, especially if you want CUDA/GPU support. The CPU-only installation is sufficient for testing the code, but image models will train faster on GPU.

## Validate files before training

```bash
python src/validate_dataset.py --config config.yaml
```

This checks:

- required table columns;
- parseability of `id` into case/rater;
- expected number of cases and raters;
- availability of the three images for each case.

## Generate shared cross-validation folds

```bash
python src/generate_folds.py --config config.yaml
```

Run this once before training any model. If the file is missing, training scripts will fail with a clear error.

## Train tabular baselines

```bash
python src/train_tabular.py --config config.yaml
```

Outputs are written to:

```text
outputs/tabular/
```

### Performance notes (image / multimodal training)

Deep training loads the same three images many times per case (13 decisions). Without caching this is disk-bound and very slow.

The codebase therefore:

- caches transformed image tensors **by `case_id`** in RAM (same tensors, no change to labels or folds);
- preloads the cache when `num_workers: 0` (recommended on Windows);
- uses `pin_memory` and accumulates metrics at end of epoch to reduce GPU/CPU sync.

Tune these keys in `config.yaml` for your machine:

```yaml
training:
  batch_size: 32       # try 16 if GPU OOM; 64 if you have more VRAM
  num_workers: 0       # 0 = preload cache (fastest on Windows)
  pin_memory: true     # set false on CPU-only runs
  device: auto         # auto | cuda | cpu
```

| Hardware situation | Suggested settings |
|--------------------|--------------------|
| ~6 GB VRAM | defaults above (`batch_size: 32`) |
| 4 GB VRAM / CUDA OOM | `batch_size: 16` or `8` |
| ≥8–12 GB VRAM | `batch_size: 64` |
| CPU only | `device: cpu`, `pin_memory: false`, small `batch_size` |
| Linux, many CPU cores | you may try `num_workers: 2` or `4` (preload is disabled when workers > 0; cache becomes lazy per worker) |
| Low system RAM (<8 GB) | avoid full preload: set `num_workers: 2` (lazy cache) or reduce `images.image_size` |

## Train image-only model

Primary classifier (decision-level, 5551 rows):

```bash
python src/train_deep.py --config config.yaml --mode image_only
```

Optional secondary regression on mean case error (427 rows):

```bash
python src/train_deep.py --config config.yaml --mode image_only_regression
```

Outputs are written to:

```text
outputs/image_only/
```

## Train multimodal model

```bash
python src/train_deep.py --config config.yaml --mode multimodal
```

Outputs are written to:

```text
outputs/multimodal/
```

The same `training:` settings in `config.yaml` apply to multimodal (same image cache and DataLoader options).

## Compare models (clustered bootstrap)

After tabular, image-only, and multimodal OOF predictions exist:

```bash
python src/compare_models.py --config config.yaml --n-bootstrap 1000
```

This script:

- loads pooled OOF predictions from `outputs/tabular/oof/`, `outputs/image_only/oof/`, and `outputs/multimodal/oof/`;
- selects the best tabular model by OOF AUROC (among logistic regression, random forest, XGBoost);
- runs a **case_id-clustered** bootstrap (resamples the 427 cases with replacement, keeps all decisions per case);
- computes paired metric differences on the same resampled cases;
- writes results to `outputs/comparisons/` (`point_metrics.csv`, `bootstrap_summary.csv`, `paired_deltas.csv`).

Do **not** bootstrap individual case–rater rows: that would treat the 13 ratings of the same case as independent.

## Notes

- Binary classifiers use a fixed threshold of `0.5` by default (`evaluation.classification_threshold` in `config.yaml`).
- Each experiment saves `config_used.yaml`, `run_metadata.json`, `metrics.csv`, `metrics_summary.csv`, and pooled OOF predictions under `oof/`.
- Per-fold plots include calibration, ROC, precision-recall, and confusion matrix PNG/CSV files.
- The visual encoder is a ResNet-18 backbone by default.
- The default setting freezes the visual backbone and trains only the fusion/classification layers.
- After obtaining baseline results, one can selectively unfreeze the last ResNet block, reduce the learning rate, and repeat training.
