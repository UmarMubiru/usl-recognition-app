from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone


def augment_with_noise(
    x: np.ndarray, y: np.ndarray, n_copies: int = 2, noise_frac: float = 0.04, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic training samples by adding scaled Gaussian noise."""
    rng = np.random.default_rng(seed)
    col_std = x.std(axis=0) + 1e-8
    parts = [x]
    for _ in range(n_copies):
        noise = rng.standard_normal(x.shape) * col_std * noise_frac
        parts.append(x + noise)
    return np.vstack(parts), np.tile(y, n_copies + 1)


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"

CATEGORY_TO_GROUP = {
    "ASCARIASIS": "parasitic",
    "HOOKWORM INFECTION": "parasitic",
    "SCHISTOSOMIASIS": "parasitic",
    "TRICHURIASIS": "parasitic",
    "MALARIA": "parasitic",
    "COVID": "viral",
    "EBOLA VIRUS": "viral",
    "HEPATITIS B": "viral",
    "HEPATITIS C": "viral",
    "HERPES SIMPLEX": "viral",
    "HIV": "viral",
    "YELLOW FEVER": "viral",
    "MOLLUSCUM CONTAGIUSUM": "viral",
    "CHOLERA": "bacterial",
    "GONORRHEA": "bacterial",
    "SYPHILLIS": "bacterial",
    "TUBERCULOSIS": "bacterial",
    "TYPHOID": "bacterial",
    "PELVIC INFLAMMATORY DISEASE (PID)": "bacterial",
    "ECTOPIC PREGNANCY": "maternal",
    "PLACENTA PREVIA": "maternal",
    "POSPARTUM HEMORRHAGE": "maternal",
    "PREECLAMPSIA": "maternal",
    "PRETERM  LABOR": "maternal",
    "SUBSTANCE USE DISORDER": "other",
}


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

    pd.DataFrame(confusion_matrix(y_val, val_pred)).to_csv(ARTIFACTS / f"confusion_matrix_{name}_val.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred)).to_csv(ARTIFACTS / f"confusion_matrix_{name}_test.csv", index=False)

    return metrics


def train_hierarchical_model(name: str, base_model, x_train, y_train_fine, groups_train, x_eval, y_eval_fine, groups_eval) -> tuple[np.ndarray, dict]:
    coarse_le = LabelEncoder()
    coarse_train = coarse_le.fit_transform(groups_train)

    coarse_model = clone(base_model)
    coarse_model.fit(x_train, coarse_train)

    group_models = {}
    group_constant = {}
    for group_name in np.unique(groups_train):
        mask = groups_train == group_name
        y_sub = y_train_fine[mask]
        unique_sub = np.unique(y_sub)
        if len(unique_sub) == 1:
            group_constant[group_name] = int(unique_sub[0])
            continue
        sub_model = clone(base_model)
        sub_model.fit(x_train[mask], y_sub)
        group_models[group_name] = sub_model

    coarse_pred_eval = coarse_model.predict(x_eval)
    pred_fine = np.zeros(len(x_eval), dtype=np.int64)
    for idx, coarse_idx in enumerate(coarse_pred_eval):
        gname = coarse_le.inverse_transform([coarse_idx])[0]
        if gname in group_models:
            pred_fine[idx] = int(group_models[gname].predict(x_eval[idx : idx + 1])[0])
        elif gname in group_constant:
            pred_fine[idx] = int(group_constant[gname])
        else:
            pred_fine[idx] = int(y_train_fine[0])

    metrics = {
        "hierarchical_accuracy": float(accuracy_score(y_eval_fine, pred_fine)),
        "hierarchical_macro_f1": float(f1_score(y_eval_fine, pred_fine, average="macro", zero_division=0)),
        "coarse_accuracy": float(accuracy_score(groups_eval, coarse_le.inverse_transform(coarse_pred_eval))),
        "coarse_macro_f1": float(f1_score(groups_eval, coarse_le.inverse_transform(coarse_pred_eval), average="macro", zero_division=0)),
    }

    pd.DataFrame(confusion_matrix(y_eval_fine, pred_fine)).to_csv(ARTIFACTS / f"confusion_matrix_{name}_hierarchical.csv", index=False)
    return pred_fine, metrics


def tune_random_forest(x_train, y_train) -> RandomForestClassifier:
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [400, 700, 1000, 1300],
        "max_depth": [None, 12, 18, 24, 32],
        "min_samples_leaf": [1, 2, 4, 6],
        "min_samples_split": [2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy", "log_loss"],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True],
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=24,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def tune_logistic_regression(x_train, y_train):
    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight=None,
                    n_jobs=None,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    param_dist = {
        "clf__C": np.logspace(-2, 2, 20),
        "clf__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def main() -> None:
    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Run prepare_csv_features.py first.")

    df = pd.read_csv(FEATURES_CSV)
    splits = pd.read_csv(SPLITS_CSV)

    data = df.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")
    data["coarse_group"] = data["category"].map(lambda x: CATEGORY_TO_GROUP.get(str(x), "other"))

    feature_cols = [
        c
        for c in data.columns
        if c
        not in {
            "sample_id",
            "category",
            "video_path",
            "split",
            "coarse_group",
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
    g_train = data.loc[train_mask, "coarse_group"].astype(str).to_numpy()
    g_val = data.loc[val_mask, "coarse_group"].astype(str).to_numpy()
    g_test = data.loc[test_mask, "coarse_group"].astype(str).to_numpy()

    # Augment training set with Gaussian noise copies (val/test untouched)
    x_train_aug, y_train_aug = augment_with_noise(x_train, y_train, n_copies=2, noise_frac=0.04)
    g_train_aug = np.tile(g_train, 3)  # 3 = 1 original + 2 copies
    print(f"Augmented training: {len(y_train)} -> {len(y_train_aug)} samples")

    rf = tune_random_forest(x_train_aug, y_train_aug)
    lr = tune_logistic_regression(x_train_aug, y_train_aug)

    rf_flat = evaluate_model("random_forest", rf, x_train_aug, y_train_aug, x_val, y_val, x_test, y_test)
    lr_flat = evaluate_model("logistic_regression", lr, x_train_aug, y_train_aug, x_val, y_val, x_test, y_test)

    _, rf_h_val = train_hierarchical_model("random_forest_val", rf, x_train_aug, y_train_aug, g_train_aug, x_val, y_val, g_val)
    _, rf_h_test = train_hierarchical_model("random_forest_test", rf, x_train_aug, y_train_aug, g_train_aug, x_test, y_test, g_test)
    _, lr_h_val = train_hierarchical_model("logistic_regression_val", lr, x_train_aug, y_train_aug, g_train_aug, x_val, y_val, g_val)
    _, lr_h_test = train_hierarchical_model("logistic_regression_test", lr, x_train_aug, y_train_aug, g_train_aug, x_test, y_test, g_test)

    metrics = {
        "random_forest": rf_flat,
        "logistic_regression": lr_flat,
        "random_forest_hierarchical": {
            "val_accuracy": rf_h_val["hierarchical_accuracy"],
            "val_macro_f1": rf_h_val["hierarchical_macro_f1"],
            "test_accuracy": rf_h_test["hierarchical_accuracy"],
            "test_macro_f1": rf_h_test["hierarchical_macro_f1"],
            "val_coarse_accuracy": rf_h_val["coarse_accuracy"],
            "test_coarse_accuracy": rf_h_test["coarse_accuracy"],
        },
        "logistic_regression_hierarchical": {
            "val_accuracy": lr_h_val["hierarchical_accuracy"],
            "val_macro_f1": lr_h_val["hierarchical_macro_f1"],
            "test_accuracy": lr_h_test["hierarchical_accuracy"],
            "test_macro_f1": lr_h_test["hierarchical_macro_f1"],
            "val_coarse_accuracy": lr_h_val["coarse_accuracy"],
            "test_coarse_accuracy": lr_h_test["coarse_accuracy"],
        },
        "random_forest_best_params": to_jsonable(rf.get_params()),
        "logistic_regression_best_params": to_jsonable(lr.get_params()),
        "coarse_groups": sorted(data["coarse_group"].unique().tolist()),
        "classes": label_encoder.classes_.tolist(),
        "feature_count": len(feature_cols),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }

    out_json = ARTIFACTS / "model_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved metrics:", out_json)
    print()
    print("RandomForest  | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
        metrics["random_forest"]["val_accuracy"],
        metrics["random_forest"]["val_macro_f1"],
        metrics["random_forest"]["test_accuracy"],
        metrics["random_forest"]["test_macro_f1"],
    ))
    print("LogisticReg   | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
        metrics["logistic_regression"]["val_accuracy"],
        metrics["logistic_regression"]["val_macro_f1"],
        metrics["logistic_regression"]["test_accuracy"],
        metrics["logistic_regression"]["test_macro_f1"],
    ))
    print("RandomForest-Hierarchical | val_acc={:.4f} test_acc={:.4f} val_coarse_acc={:.4f} test_coarse_acc={:.4f}".format(
        metrics["random_forest_hierarchical"]["val_accuracy"],
        metrics["random_forest_hierarchical"]["test_accuracy"],
        metrics["random_forest_hierarchical"]["val_coarse_accuracy"],
        metrics["random_forest_hierarchical"]["test_coarse_accuracy"],
    ))
    print("LogisticReg-Hierarchical  | val_acc={:.4f} test_acc={:.4f} val_coarse_acc={:.4f} test_coarse_acc={:.4f}".format(
        metrics["logistic_regression_hierarchical"]["val_accuracy"],
        metrics["logistic_regression_hierarchical"]["test_accuracy"],
        metrics["logistic_regression_hierarchical"]["val_coarse_accuracy"],
        metrics["logistic_regression_hierarchical"]["test_coarse_accuracy"],
    ))


if __name__ == "__main__":
    main()
