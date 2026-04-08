from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def augment_with_noise(
    x: np.ndarray, y: np.ndarray, n_copies: int = 2, noise_frac: float = 0.04, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    col_std = x.std(axis=0) + 1e-8
    parts = [x]
    for _ in range(n_copies):
        noise = rng.standard_normal(x.shape) * col_std * noise_frac
        parts.append(x + noise)
    return np.vstack(parts), np.tile(y, n_copies + 1)


def to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return repr(value)


def evaluate_model(name: str, model, x_train, y_train, x_val, y_val, x_test, y_test) -> dict[str, float]:
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
    }

    pd.DataFrame(confusion_matrix(y_val, val_pred)).to_csv(
        ARTIFACTS / f"confusion_matrix_{name}_val.csv", index=False
    )
    pd.DataFrame(confusion_matrix(y_test, test_pred)).to_csv(
        ARTIFACTS / f"confusion_matrix_{name}_test.csv", index=False
    )

    return metrics


def tune_svm(x_train: np.ndarray, y_train: np.ndarray):
    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    random_state=42,
                ),
            ),
        ]
    )
    param_dist = {
        "clf__C": np.logspace(-1, 2, 16),
        "clf__gamma": ["scale", "auto"] + list(np.logspace(-4, -1, 6)),
        "clf__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=18,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def tune_hgb(x_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    base = HistGradientBoostingClassifier(random_state=42)
    param_dist = {
        "learning_rate": np.logspace(-2, -0.2, 12),
        "max_iter": [200, 300, 500, 700],
        "max_depth": [None, 8, 12, 16],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [5, 10, 20, 30],
        "l2_regularization": np.logspace(-6, 1, 10),
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=18,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"


def main() -> None:
    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Run prepare_csv_features.py first.")

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

    x = data[feature_cols].fillna(0.0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["category"].astype(str))

    train_mask = data["split"] == "train"
    val_mask = data["split"] == "val"
    test_mask = data["split"] == "test"

    x_train, y_train = x[train_mask].to_numpy(), y[train_mask]
    x_val, y_val = x[val_mask].to_numpy(), y[val_mask]
    x_test, y_test = x[test_mask].to_numpy(), y[test_mask]

    x_train_aug, y_train_aug = augment_with_noise(x_train, y_train, n_copies=2, noise_frac=0.04)
    print(f"Augmented training: {len(y_train)} -> {len(y_train_aug)} samples")

    svm = tune_svm(x_train_aug, y_train_aug)
    hgb = tune_hgb(x_train_aug, y_train_aug)

    svm_metrics = evaluate_model("svm_rbf", svm, x_train_aug, y_train_aug, x_val, y_val, x_test, y_test)
    hgb_metrics = evaluate_model("hist_gradient_boosting", hgb, x_train_aug, y_train_aug, x_val, y_val, x_test, y_test)

    metrics = {
        "svm_rbf": svm_metrics,
        "hist_gradient_boosting": hgb_metrics,
        "svm_best_params": to_jsonable(svm.get_params()),
        "hgb_best_params": to_jsonable(hgb.get_params()),
        "classes": label_encoder.classes_.tolist(),
        "feature_count": len(feature_cols),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }

    out_json = ARTIFACTS / "model_metrics_svm_hgb.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved metrics:", out_json)
    print()
    print(
        "SVM-RBF        | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
            metrics["svm_rbf"]["val_accuracy"],
            metrics["svm_rbf"]["val_macro_f1"],
            metrics["svm_rbf"]["test_accuracy"],
            metrics["svm_rbf"]["test_macro_f1"],
        )
    )
    print(
        "HistGradBoost  | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
            metrics["hist_gradient_boosting"]["val_accuracy"],
            metrics["hist_gradient_boosting"]["val_macro_f1"],
            metrics["hist_gradient_boosting"]["test_accuracy"],
            metrics["hist_gradient_boosting"]["test_macro_f1"],
        )
    )


if __name__ == "__main__":
    main()
