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

## Knowledge Distillation (Teacher -> Student)

This project can distill a deployment teacher model into compact students (linear softmax, tiny MLP, KD tiny HGB, KD pseudo-label logistic), then auto-select the best student by validation macro-F1.

1. Ensure teacher artifact exists (best model export):
   - `python models_dataset1/csv_models/export_best_model.py`
2. Train distilled student and export deployment-ready artifact:
   - `python models_dataset1/csv_models/train_distilled_student.py --augment`

Optional:

- Force linear only: `python models_dataset1/csv_models/train_distilled_student.py --student linear --augment`
- Force tiny MLP only: `python models_dataset1/csv_models/train_distilled_student.py --student mlp --augment`
- Force KD tiny HGB only: `python models_dataset1/csv_models/train_distilled_student.py --student hgb --augment`
- Force KD pseudo-label logistic only: `python models_dataset1/csv_models/train_distilled_student.py --student pseudo-lr --augment`

Teacher variants:

- Single teacher (default): `python models_dataset1/csv_models/train_distilled_student.py --augment`
- Ensemble teacher (SVM artifact + aux HGB): `python models_dataset1/csv_models/train_distilled_student.py --teacher-mode svm-hgb-ensemble --augment`

Confidence calibration:

- Enabled by default (temperature scaling on validation set).
- Disable calibration: `python models_dataset1/csv_models/train_distilled_student.py --no-calibrate-temperature --augment`

Outputs written to `models_dataset1/csv_models/artifacts/`:

- `distilled_student.joblib`
- `distilled_student_metadata.json`
- `distillation_report.json`
- `confusion_matrix_distilled_student_val.csv`
- `confusion_matrix_distilled_student_test.csv`

`distillation_report.json` includes teacher-vs-student metrics, candidate comparisons, runtime benchmarks (artifact size + predict latency), and confidence-quality metrics (NLL, ECE, top-3 accuracy).

Additional calibration diagnostics are exported to `models_dataset1/csv_models/artifacts/calibration/`:

- `classwise_confidence_val.csv`
- `classwise_confidence_test.csv`
- `reliability_bins_val.csv`
- `reliability_bins_test.csv`
- `reliability_plot_val.png` (when matplotlib is available)
- `reliability_plot_test.png` (when matplotlib is available)

## Outputs

- `models_dataset1/csv_models/artifacts/dataset1_disease_features.csv`
- `models_dataset1/csv_models/artifacts/dataset1_disease_splits.csv`
- `models_dataset1/csv_models/artifacts/model_metrics.json`
- `models_dataset1/csv_models/artifacts/confusion_matrix_*.csv`
