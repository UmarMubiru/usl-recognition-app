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
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"
RF_LR_METRICS = ARTIFACTS / "model_metrics.json"
SVM_HGB_METRICS = ARTIFACTS / "model_metrics_svm_hgb.json"
XAI_DIR = ARTIFACTS / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_models(metrics_rf_lr: dict, metrics_svm_hgb: dict):
    rf_params = metrics_rf_lr.get("random_forest_best_params", {})
    rf_keys = {
        "bootstrap",
        "class_weight",
        "criterion",
        "max_depth",
        "max_features",
        "max_leaf_nodes",
        "max_samples",
        "min_impurity_decrease",
        "min_samples_leaf",
        "min_samples_split",
        "min_weight_fraction_leaf",
        "monotonic_cst",
        "n_estimators",
        "n_jobs",
        "oob_score",
        "random_state",
        "verbose",
        "warm_start",
        "ccp_alpha",
    }
    rf_cfg = {k: v for k, v in rf_params.items() if k in rf_keys}
    rf_cfg.setdefault("random_state", 42)
    rf = RandomForestClassifier(**rf_cfg)

    lr_params = metrics_rf_lr.get("logistic_regression_best_params", {})
    lr_c = float(lr_params.get("clf__C", 1.0))
    lr_class_weight = lr_params.get("clf__class_weight", None)
    lr = Pipeline(
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
    )

    svm_params = metrics_svm_hgb.get("svm_best_params", {})
    svm_c = float(svm_params.get("clf__C", 1.0))
    svm_gamma = svm_params.get("clf__gamma", "scale")
    svm_class_weight = svm_params.get("clf__class_weight", None)
    svm = Pipeline(
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
    )

    hgb_params = metrics_svm_hgb.get("hgb_best_params", {})
    hgb_keys = {
        "categorical_features",
        "class_weight",
        "early_stopping",
        "interaction_cst",
        "l2_regularization",
        "learning_rate",
        "loss",
        "max_bins",
        "max_depth",
        "max_features",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "monotonic_cst",
        "n_iter_no_change",
        "random_state",
        "scoring",
        "tol",
        "validation_fraction",
        "verbose",
        "warm_start",
    }
    hgb_cfg = {k: v for k, v in hgb_params.items() if k in hgb_keys}
    hgb_cfg.setdefault("random_state", 42)
    hgb = HistGradientBoostingClassifier(**hgb_cfg)

    return {
        "random_forest": rf,
        "logistic_regression": lr,
        "svm_rbf": svm,
        "hist_gradient_boosting": hgb,
    }


def _save_importance_plot(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 20) -> None:
    head = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(head["feature"], head["importance_mean"], xerr=head.get("importance_std", None))
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _intrinsic_importance(model_name: str, model, feature_cols: list[str]) -> pd.DataFrame | None:
    if model_name == "random_forest" and hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        return pd.DataFrame({"feature": feature_cols, "importance_mean": imp, "importance_std": 0.0}).sort_values(
            "importance_mean", ascending=False
        )

    if model_name == "hist_gradient_boosting" and hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        return pd.DataFrame({"feature": feature_cols, "importance_mean": imp, "importance_std": 0.0}).sort_values(
            "importance_mean", ascending=False
        )

    if model_name == "logistic_regression":
        clf = model.named_steps["clf"]
        coef = np.asarray(clf.coef_, dtype=np.float64)
        imp = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame({"feature": feature_cols, "importance_mean": imp, "importance_std": 0.0}).sort_values(
            "importance_mean", ascending=False
        )

    return None


def _quick_permutation_importance(
    model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_cols: list[str],
    max_features: int = 100,
    n_repeats: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Fast, stable approximation of permutation importance for Windows environments."""
    rng = np.random.default_rng(seed)
    baseline_acc = float((model.predict(x_eval) == y_eval).mean())

    variances = np.var(x_eval, axis=0)
    idx_sorted = np.argsort(-variances)
    selected_idx = idx_sorted[: min(max_features, x_eval.shape[1])]

    rows: list[dict[str, float | str]] = []
    for idx in selected_idx:
        scores = []
        for _ in range(n_repeats):
            xp = x_eval.copy()
            shuffled = xp[:, idx].copy()
            rng.shuffle(shuffled)
            xp[:, idx] = shuffled
            scores.append(float((model.predict(xp) == y_eval).mean()))
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        rows.append(
            {
                "feature": feature_cols[int(idx)],
                "importance_mean": baseline_acc - mean_score,
                "importance_std": std_score,
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def main() -> None:
    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Missing features/splits. Run prepare_csv_features.py first.")
    if not RF_LR_METRICS.exists() or not SVM_HGB_METRICS.exists():
        raise FileNotFoundError("Missing metrics JSON files. Train all four models first.")

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

    metrics_rf_lr = _read_json(RF_LR_METRICS)
    metrics_svm_hgb = _read_json(SVM_HGB_METRICS)
    models = _build_models(metrics_rf_lr, metrics_svm_hgb)

    audit_summary: dict[str, dict] = {}

    for model_name, model in models.items():
        model.fit(x_train_aug, y_train_aug)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        model_summary = {
            "val_accuracy": float((val_pred == y_val).mean()),
            "test_accuracy": float((test_pred == y_test).mean()),
            "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
            "test_macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        }

        perm_df = _quick_permutation_importance(
            model,
            x_test,
            y_test,
            feature_cols,
            max_features=100,
            n_repeats=2,
            seed=42,
        )
        perm_csv = XAI_DIR / f"permutation_importance_{model_name}.csv"
        perm_df.to_csv(perm_csv, index=False)
        _save_importance_plot(
            perm_df,
            f"Permutation Importance ({model_name})",
            XAI_DIR / f"permutation_importance_{model_name}.png",
            top_n=20,
        )

        intrinsic_df = _intrinsic_importance(model_name, model, feature_cols)
        if intrinsic_df is not None:
            intrinsic_csv = XAI_DIR / f"intrinsic_importance_{model_name}.csv"
            intrinsic_df.to_csv(intrinsic_csv, index=False)
            _save_importance_plot(
                intrinsic_df,
                f"Intrinsic Importance ({model_name})",
                XAI_DIR / f"intrinsic_importance_{model_name}.png",
                top_n=20,
            )
            model_summary["intrinsic_importance_csv"] = str(intrinsic_csv)

        model_summary["permutation_importance_csv"] = str(perm_csv)
        model_summary["top5_permutation_features"] = perm_df.head(5)["feature"].tolist()
        audit_summary[model_name] = model_summary

    summary = {
        "xai_method": [
            "Permutation Importance (model-agnostic)",
            "Intrinsic Importances (RF/HGB/LR coefficients)",
        ],
        "feature_count": len(feature_cols),
        "classes": label_encoder.classes_.tolist(),
        "models": audit_summary,
    }

    out_json = XAI_DIR / "xai_audit_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved XAI audit summary:", out_json)
    for name, info in audit_summary.items():
        print(
            "{:<22} | val_acc={:.4f} test_acc={:.4f} top1_perm={}".format(
                name,
                info["val_accuracy"],
                info["test_accuracy"],
                info["top5_permutation_features"][0] if info["top5_permutation_features"] else "n/a",
            )
        )


if __name__ == "__main__":
    main()
