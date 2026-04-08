from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"
METRICS_RF_LR = ARTIFACTS / "model_metrics.json"
METRICS_SVM_HGB = ARTIFACTS / "model_metrics_svm_hgb.json"


def augment_with_noise(
    x: np.ndarray,
    y: np.ndarray,
    n_copies: int = 2,
    noise_frac: float = 0.04,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    col_std = x.std(axis=0) + 1e-8
    parts = [x]
    for _ in range(n_copies):
        noise = rng.standard_normal(x.shape) * col_std * noise_frac
        parts.append(x + noise)
    return np.vstack(parts), np.tile(y, n_copies + 1)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pick_best_model(metric_name: str = "test_accuracy") -> tuple[str, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics_main = _load_json(METRICS_RF_LR)
    metrics_extra = _load_json(METRICS_SVM_HGB)

    candidates: list[tuple[str, float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    if "random_forest" in metrics_main:
        candidates.append((
            "random_forest",
            float(metrics_main["random_forest"].get(metric_name, -np.inf)),
            metrics_main["random_forest"],
            metrics_main,
            metrics_extra,
        ))
    if "logistic_regression" in metrics_main:
        candidates.append((
            "logistic_regression",
            float(metrics_main["logistic_regression"].get(metric_name, -np.inf)),
            metrics_main["logistic_regression"],
            metrics_main,
            metrics_extra,
        ))
    if "svm_rbf" in metrics_extra:
        candidates.append((
            "svm_rbf",
            float(metrics_extra["svm_rbf"].get(metric_name, -np.inf)),
            metrics_extra["svm_rbf"],
            metrics_main,
            metrics_extra,
        ))
    if "hist_gradient_boosting" in metrics_extra:
        candidates.append((
            "hist_gradient_boosting",
            float(metrics_extra["hist_gradient_boosting"].get(metric_name, -np.inf)),
            metrics_extra["hist_gradient_boosting"],
            metrics_main,
            metrics_extra,
        ))

    if not candidates:
        raise RuntimeError("No model candidates found in metrics files.")

    best = max(candidates, key=lambda item: item[1])
    return best


def build_model(name: str, metrics_main: dict[str, Any], metrics_extra: dict[str, Any]):
    if name == "random_forest":
        params = metrics_main.get("random_forest_best_params", {})
        allowed = {
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
        rf_params = {k: v for k, v in params.items() if k in allowed}
        if "random_state" not in rf_params:
            rf_params["random_state"] = 42
        if "n_jobs" not in rf_params:
            rf_params["n_jobs"] = -1
        return RandomForestClassifier(**rf_params)

    if name == "logistic_regression":
        params = metrics_main.get("logistic_regression_best_params", {})
        c_value = float(params.get("clf__C", 1.0))
        class_weight = params.get("clf__class_weight", None)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        class_weight=class_weight,
                        max_iter=5000,
                        solver="lbfgs",
                        n_jobs=None,
                        random_state=42,
                    ),
                ),
            ]
        )

    if name == "svm_rbf":
        params = metrics_extra.get("svm_best_params", {})
        c_value = float(params.get("clf__C", 1.0))
        gamma_value = params.get("clf__gamma", "scale")
        class_weight = params.get("clf__class_weight", None)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=c_value,
                        gamma=gamma_value,
                        kernel="rbf",
                        class_weight=class_weight,
                        probability=False,
                        random_state=42,
                    ),
                ),
            ]
        )

    if name == "hist_gradient_boosting":
        params = metrics_extra.get("hgb_best_params", {})
        allowed = {
            "learning_rate",
            "max_iter",
            "max_depth",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "random_state",
        }
        hgb_params = {k: v for k, v in params.items() if k in allowed}
        if "random_state" not in hgb_params:
            hgb_params["random_state"] = 42
        return HistGradientBoostingClassifier(**hgb_params)

    raise ValueError(f"Unsupported model name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + export best Dataset-1 model for deployment")
    parser.add_argument("--metric", default="test_accuracy", help="Metric key used for best-model selection")
    parser.add_argument("--output", default=str(ARTIFACTS / "best_model.joblib"), help="Artifact output path")
    parser.add_argument("--metadata-output", default=str(ARTIFACTS / "best_model_metadata.json"), help="Metadata output path")
    parser.add_argument("--augment", action="store_true", help="Apply Gaussian noise augmentation before fit")
    parser.add_argument("--n-copies", type=int, default=2, help="Augmentation copies if --augment is set")
    parser.add_argument("--noise-frac", type=float, default=0.04, help="Augmentation noise fraction")
    args = parser.parse_args()

    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Missing features/splits. Run prepare_csv_features.py first.")

    best_name, best_value, best_metrics, metrics_main, metrics_extra = pick_best_model(args.metric)

    df = pd.read_csv(FEATURES_CSV)
    splits = pd.read_csv(SPLITS_CSV)
    data = df.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")

    feature_cols = [
        c
        for c in data.columns
        if c
        not in {
            "sample_id",
            "category",
            "video_path",
            "split",
        }
    ]

    x = data[feature_cols].fillna(0.0).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["category"].astype(str))

    train_val_mask = data["split"].isin(["train", "val"])
    x_train = x[train_val_mask]
    y_train = y[train_val_mask]

    if args.augment:
        x_train, y_train = augment_with_noise(
            x_train,
            y_train,
            n_copies=args.n_copies,
            noise_frac=args.noise_frac,
            seed=42,
        )

    model = build_model(best_name, metrics_main, metrics_extra)
    model.fit(x_train, y_train)

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "classes": label_encoder.classes_.tolist(),
        "selection_metric": args.metric,
        "selection_value": float(best_value),
        "selection_metrics": best_metrics,
        "trained_on": "train+val",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, output_path)

    metadata = {
        k: v
        for k, v in artifact.items()
        if k != "model"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Best model selected: {best_name} ({args.metric}={best_value:.4f})")
    print(f"Saved model artifact: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
