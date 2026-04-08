from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"
METRICS_RF_LR = ARTIFACTS / "model_metrics.json"
METRICS_SVM_HGB = ARTIFACTS / "model_metrics_svm_hgb.json"
LC_DIR = ARTIFACTS / "learning_curves"
LC_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_models(metrics_main: dict, metrics_extra: dict):
    rf_params = metrics_main.get("random_forest_best_params", {})
    rf_allowed = {
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "bootstrap",
        "random_state",
        "n_jobs",
    }
    rf_cfg = {k: v for k, v in rf_params.items() if k in rf_allowed}
    rf_cfg.setdefault("random_state", 42)
    rf_cfg.setdefault("n_jobs", -1)

    lr_params = metrics_main.get("logistic_regression_best_params", {})
    lr_c = float(lr_params.get("clf__C", 1.0))
    lr_class_weight = lr_params.get("clf__class_weight", None)

    svm_params = metrics_extra.get("svm_best_params", {})
    svm_c = float(svm_params.get("clf__C", 1.0))
    svm_gamma = svm_params.get("clf__gamma", "scale")
    svm_class_weight = svm_params.get("clf__class_weight", None)

    hgb_params = metrics_extra.get("hgb_best_params", {})
    hgb_allowed = {
        "learning_rate",
        "max_iter",
        "max_depth",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "random_state",
    }
    hgb_cfg = {k: v for k, v in hgb_params.items() if k in hgb_allowed}
    hgb_cfg.setdefault("random_state", 42)

    return {
        "random_forest": RandomForestClassifier(**rf_cfg),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=lr_c,
                        class_weight=lr_class_weight,
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=svm_c,
                        gamma=svm_gamma,
                        class_weight=svm_class_weight,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(**hgb_cfg),
    }


def _plot_curve(df: pd.DataFrame, model_name: str, metric_label: str, out_path: Path) -> None:
    x = df["train_size"].to_numpy()
    train_mean = df["train_mean"].to_numpy()
    train_std = df["train_std"].to_numpy()
    val_mean = df["val_mean"].to_numpy()
    val_std = df["val_std"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, train_mean, marker="o", label="Train")
    plt.plot(x, val_mean, marker="o", label="Validation")
    plt.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(x, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.title(f"Learning Curve ({model_name}) - {metric_label}")
    plt.xlabel("Training samples")
    plt.ylabel(metric_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Run prepare_csv_features.py first.")

    metrics_main = _load_json(METRICS_RF_LR)
    metrics_extra = _load_json(METRICS_SVM_HGB)

    df = pd.read_csv(FEATURES_CSV)
    splits = pd.read_csv(SPLITS_CSV)
    data = df.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")

    feature_cols = [
        c
        for c in data.columns
        if c not in {"sample_id", "category", "video_path", "split"}
    ]

    x = data[feature_cols].fillna(0.0).to_numpy()
    y = LabelEncoder().fit_transform(data["category"].astype(str))

    train_mask = data["split"] == "train"
    x_train = x[train_mask]
    y_train = y[train_mask]

    models = _build_models(metrics_main, metrics_extra)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    train_sizes_frac = np.linspace(0.2, 1.0, 5)

    summary: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        for scoring, tag, label in [
            ("accuracy", "acc", "Accuracy"),
            ("f1_macro", "f1_macro", "Macro-F1"),
        ]:
            sizes, train_scores, val_scores = learning_curve(
                estimator=model,
                X=x_train,
                y=y_train,
                train_sizes=train_sizes_frac,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                shuffle=True,
                random_state=42,
            )

            curve_df = pd.DataFrame(
                {
                    "train_size": sizes,
                    "train_mean": train_scores.mean(axis=1),
                    "train_std": train_scores.std(axis=1),
                    "val_mean": val_scores.mean(axis=1),
                    "val_std": val_scores.std(axis=1),
                }
            )

            csv_path = LC_DIR / f"learning_curve_{name}_{tag}.csv"
            png_path = LC_DIR / f"learning_curve_{name}_{tag}.png"
            curve_df.to_csv(csv_path, index=False)
            _plot_curve(curve_df, name, label, png_path)

            key = f"{name}_{tag}"
            summary[key] = {
                "max_train_size": float(sizes[-1]),
                "train_mean_at_max_size": float(curve_df.iloc[-1]["train_mean"]),
                "val_mean_at_max_size": float(curve_df.iloc[-1]["val_mean"]),
                "csv": str(csv_path),
                "png": str(png_path),
            }

    out_json = LC_DIR / "learning_curve_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved learning curve summary:", out_json)


if __name__ == "__main__":
    main()
