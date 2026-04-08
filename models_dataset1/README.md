# Dataset 1 CSV Models (Clean Restart)

This workspace trains exactly **two tabular models** on **Dataset 1 disease-only videos**.

## Models
1. `RandomForestClassifier`
2. `LogisticRegression` (with `StandardScaler`)

## Workflow
1. Build CSV features and splits:
   - `python models_dataset1/csv_models/prepare_csv_features.py`
2. Train and evaluate two models:
   - `python models_dataset1/csv_models/train_two_models.py`

## Outputs
- `models_dataset1/csv_models/artifacts/dataset1_disease_features.csv`
- `models_dataset1/csv_models/artifacts/dataset1_disease_splits.csv`
- `models_dataset1/csv_models/artifacts/model_metrics.json`
- `models_dataset1/csv_models/artifacts/confusion_matrix_*.csv`
