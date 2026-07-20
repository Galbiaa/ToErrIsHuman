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
case<d>saggital.jpg
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
    train_tabular.py
    train_deep.py
    validate_dataset.py
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

### B. Image-only model

The script `src/train_deep.py --mode image_only` trains a case-level image model.

Because the same three images are shared by the 13 rater decisions for a given case, the image-only model uses 427 case-level observations. Its target is the mean `error-rating` across the 13 raters for that case. The model therefore estimates the average probability that a case elicits an erroneous decision.

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

## Train image-only model

```bash
python src/train_deep.py --config config.yaml --mode image_only
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

## Notes

- Binary classifiers use a fixed threshold of `0.5` by default (`evaluation.classification_threshold` in `config.yaml`).
- Each experiment saves `config_used.yaml`, `run_metadata.json`, `metrics.csv`, `metrics_summary.csv`, and pooled OOF predictions under `oof/`.
- Per-fold plots include calibration, ROC, precision-recall, and confusion matrix PNG/CSV files.
- The visual encoder is a ResNet-18 backbone by default.
- The default setting freezes the visual backbone and trains only the fusion/classification layers. This is usually preferable with 427 image cases.
- After obtaining baseline results, one can selectively unfreeze the last ResNet block, reduce the learning rate, and repeat training.
