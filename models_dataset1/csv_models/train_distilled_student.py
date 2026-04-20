from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_dataset1.shared.distilled_student import DistilledMLPClassifier, DistilledSoftmaxClassifier
from models_dataset1.shared.calibration import TemperatureScaledClassifier


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "dataset1_disease_features.csv"
SPLITS_CSV = ARTIFACTS / "dataset1_disease_splits.csv"
DEFAULT_TEACHER = ARTIFACTS / "best_model.joblib"
DEFAULT_STUDENT = ARTIFACTS / "distilled_student.joblib"
DEFAULT_STUDENT_META = ARTIFACTS / "distilled_student_metadata.json"
DEFAULT_REPORT = ARTIFACTS / "distillation_report.json"


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


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return exp_vals / denom


def _teacher_probs(model, x: np.ndarray, n_classes: int, temperature: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError("Teacher predict_proba output must be 2D")
        if probs.shape[1] != n_classes:
            raise ValueError("Teacher class dimension mismatch")
        return probs

    if not hasattr(model, "decision_function"):
        raise ValueError("Teacher must provide either predict_proba or decision_function")

    decision = np.asarray(model.decision_function(x), dtype=np.float64)
    t = max(1e-6, float(temperature))
    if decision.ndim == 1:
        # Binary OVR shape (n_samples,) -> two-logit representation.
        logits = np.column_stack([-decision / t, decision / t])
    else:
        logits = decision / t
    probs = _softmax_rows(logits)

    if probs.shape[1] != n_classes:
        raise ValueError(
            f"Teacher class dimension mismatch: expected {n_classes}, got {probs.shape[1]}"
        )
    return probs


def _blend_probs(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not probs_list:
        raise ValueError("At least one probability matrix is required")
    if len(probs_list) != len(weights):
        raise ValueError("weights must match probs_list length")
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    out = np.zeros_like(probs_list[0], dtype=np.float64)
    for probs, w in zip(probs_list, weights):
        out += probs * (float(w) / total_weight)

    row_sum = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sum


def _build_hgb_teacher(args) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=args.teacher_hgb_learning_rate,
        max_iter=args.teacher_hgb_max_iter,
        max_depth=args.teacher_hgb_max_depth,
        max_leaf_nodes=args.teacher_hgb_max_leaf_nodes,
        min_samples_leaf=args.teacher_hgb_min_samples_leaf,
        l2_regularization=args.teacher_hgb_l2,
        random_state=42,
    )


def _eval(model, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred": pred,
    }


def _nll_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        ece += abs(acc_bin - conf_bin) * (float(np.sum(mask)) / n)
    return float(ece)


def _topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    k = max(1, min(k, probs.shape[1]))
    topk = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    hit = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(hit))


def _classwise_confidence_table(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows: list[dict[str, float | int | str]] = []

    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        support = int(np.sum(mask))
        if support == 0:
            rows.append(
                {
                    "class_index": int(class_idx),
                    "class_name": class_name,
                    "support": 0,
                    "accuracy": 0.0,
                    "mean_top1_confidence": 0.0,
                    "mean_true_class_prob": 0.0,
                    "confidence_gap": 0.0,
                    "class_nll": 0.0,
                }
            )
            continue

        cls_pred = pred[mask]
        cls_conf = conf[mask]
        cls_true_probs = np.clip(probs[mask, class_idx], 1e-12, 1.0)
        acc = float(np.mean(cls_pred == class_idx))
        mean_conf = float(np.mean(cls_conf))
        mean_true_prob = float(np.mean(cls_true_probs))
        class_nll = float(-np.mean(np.log(cls_true_probs)))

        rows.append(
            {
                "class_index": int(class_idx),
                "class_name": class_name,
                "support": support,
                "accuracy": acc,
                "mean_top1_confidence": mean_conf,
                "mean_true_class_prob": mean_true_prob,
                "confidence_gap": float(mean_conf - acc),
                "class_nll": class_nll,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["support", "class_index"], ascending=[False, True]).reset_index(drop=True)


def _reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": i,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "count": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        acc = float(np.mean(correct[mask]))
        mean_conf = float(np.mean(conf[mask]))
        rows.append(
            {
                "bin_index": i,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": count,
                "accuracy": acc,
                "mean_confidence": mean_conf,
                "gap": float(mean_conf - acc),
            }
        )
    return pd.DataFrame(rows)


def _save_reliability_plot(bin_df: pd.DataFrame, out_path: Path, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = bin_df["mean_confidence"].to_numpy(dtype=float)
    y = bin_df["accuracy"].to_numpy(dtype=float)
    sizes = np.maximum(bin_df["count"].to_numpy(dtype=float), 1.0)

    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.scatter(x, y, s=20 + 2.0 * np.sqrt(sizes), alpha=0.8, color="#1f77b4", label="bins")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _build_action_recommendations(classwise_df: pd.DataFrame, top_n: int = 3) -> list[dict[str, float | int | str]]:
    work = classwise_df.copy()
    if work.empty:
        return []

    # Prioritize classes with higher support, lower accuracy, higher uncertainty loss,
    # and stronger confidence mismatch for practical data-collection impact.
    support = work["support"].astype(float)
    inv_acc = 1.0 - work["accuracy"].astype(float)
    abs_gap = np.abs(work["confidence_gap"].astype(float))
    nll = work["class_nll"].astype(float)

    work["priority_score"] = (
        0.45 * (support / np.maximum(support.max(), 1.0))
        + 0.35 * inv_acc
        + 0.15 * np.tanh(nll)
        + 0.05 * np.tanh(abs_gap)
    )

    top = work.sort_values("priority_score", ascending=False).head(max(1, int(top_n)))
    rows: list[dict[str, float | int | str]] = []
    for _, row in top.iterrows():
        rows.append(
            {
                "class_index": int(row["class_index"]),
                "class_name": str(row["class_name"]),
                "support": int(row["support"]),
                "accuracy": float(row["accuracy"]),
                "confidence_gap": float(row["confidence_gap"]),
                "class_nll": float(row["class_nll"]),
                "priority_score": float(row["priority_score"]),
                "recommended_action": "collect more varied examples and add targeted augmentations",
            }
        )
    return rows


def _write_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(out_csv, index=False)


def _to_float(value: Any) -> float | str:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return "unknown"


def _artifact_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def _latency_metrics(model, x_eval: np.ndarray) -> dict[str, float]:
    x_eval = np.asarray(x_eval)
    if len(x_eval) == 0:
        return {
            "predict_ms_per_sample": 0.0,
            "predict_ms_batch32": 0.0,
        }

    # Warmup call to avoid one-time overhead skewing measurements.
    model.predict(x_eval[:1])

    n = min(64, len(x_eval))
    single_times = []
    for i in range(n):
        x_one = x_eval[i : i + 1]
        start = time.perf_counter()
        model.predict(x_one)
        single_times.append((time.perf_counter() - start) * 1000.0)

    batch = x_eval[: min(32, len(x_eval))]
    start_batch = time.perf_counter()
    model.predict(batch)
    batch_ms = (time.perf_counter() - start_batch) * 1000.0

    return {
        "predict_ms_per_sample": float(np.mean(single_times)),
        "predict_ms_batch32": float(batch_ms),
    }


def _build_student_candidates(args) -> list[tuple[str, Pipeline]]:
    linear = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                DistilledSoftmaxClassifier(
                    learning_rate=args.learning_rate,
                    max_iter=args.max_iter,
                    batch_size=args.batch_size,
                    l2=args.l2,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                DistilledMLPClassifier(
                    hidden_dim=args.mlp_hidden_dim,
                    learning_rate=args.mlp_learning_rate,
                    max_iter=args.mlp_max_iter,
                    batch_size=args.mlp_batch_size,
                    l2=args.mlp_l2,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
    hgb = Pipeline(
        steps=[
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=args.hgb_learning_rate,
                    max_iter=args.hgb_max_iter,
                    max_depth=args.hgb_max_depth,
                    max_leaf_nodes=args.hgb_max_leaf_nodes,
                    min_samples_leaf=args.hgb_min_samples_leaf,
                    l2_regularization=args.hgb_l2,
                    random_state=42,
                ),
            ),
        ]
    )
    pseudo_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=args.pseudo_lr_c,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    if args.student == "linear":
        return [("distilled_softmax_linear", linear)]
    if args.student == "mlp":
        return [("distilled_mlp_tiny", mlp)]
    if args.student == "hgb":
        return [("kd_hgb_tiny", hgb)]
    if args.student == "pseudo-lr":
        return [("kd_pseudo_logreg", pseudo_lr)]
    return [
        ("distilled_softmax_linear", linear),
        ("distilled_mlp_tiny", mlp),
        ("kd_hgb_tiny", hgb),
        ("kd_pseudo_logreg", pseudo_lr),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train distilled student for Dataset-1 tabular model")
    parser.add_argument("--teacher-artifact", default=str(DEFAULT_TEACHER), help="Path to teacher artifact")
    parser.add_argument("--output", default=str(DEFAULT_STUDENT), help="Output path for distilled student artifact")
    parser.add_argument("--metadata-output", default=str(DEFAULT_STUDENT_META), help="Output metadata JSON path")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT), help="Output benchmark report JSON path")
    parser.add_argument("--teacher-mode", choices=["single", "svm-hgb-ensemble"], default="single")
    parser.add_argument("--teacher-primary-weight", type=float, default=0.7)
    parser.add_argument("--teacher-aux-weight", type=float, default=0.3)
    parser.add_argument("--calibrate-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=0.6, help="Soft-target blend ratio in [0, 1]")
    parser.add_argument("--temperature", type=float, default=3.0, help="Teacher temperature for logits")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--student", choices=["linear", "mlp", "hgb", "pseudo-lr", "both"], default="both")
    parser.add_argument("--mlp-hidden-dim", type=int, default=96)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.01)
    parser.add_argument("--mlp-max-iter", type=int, default=2400)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-l2", type=float, default=1e-4)
    parser.add_argument("--mlp-patience", type=int, default=180)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--hgb-max-iter", type=int, default=320)
    parser.add_argument("--hgb-max-depth", type=int, default=8)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=63)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=8)
    parser.add_argument("--hgb-l2", type=float, default=1e-3)
    parser.add_argument("--pseudo-lr-c", type=float, default=1.0)
    parser.add_argument("--teacher-hgb-learning-rate", type=float, default=0.06)
    parser.add_argument("--teacher-hgb-max-iter", type=int, default=480)
    parser.add_argument("--teacher-hgb-max-depth", type=int, default=10)
    parser.add_argument("--teacher-hgb-max-leaf-nodes", type=int, default=63)
    parser.add_argument("--teacher-hgb-min-samples-leaf", type=int, default=8)
    parser.add_argument("--teacher-hgb-l2", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true", help="Apply Gaussian augmentation to training set")
    parser.add_argument("--n-copies", type=int, default=2, help="Augmentation copies if --augment is set")
    parser.add_argument("--noise-frac", type=float, default=0.04, help="Augmentation noise fraction")
    args = parser.parse_args()

    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        raise FileNotFoundError("Missing features/splits. Run prepare_csv_features.py first.")

    teacher_artifact_path = Path(args.teacher_artifact)
    if not teacher_artifact_path.exists():
        raise FileNotFoundError(f"Teacher artifact not found: {teacher_artifact_path}")

    teacher_artifact = joblib.load(teacher_artifact_path)
    required = {"model", "model_name", "feature_cols", "classes"}
    missing = required.difference(teacher_artifact.keys())
    if missing:
        raise ValueError(f"Invalid teacher artifact, missing keys: {sorted(missing)}")

    teacher_model = teacher_artifact["model"]
    teacher_name = str(teacher_artifact["model_name"])
    feature_cols = list(teacher_artifact["feature_cols"])
    class_names = list(teacher_artifact["classes"])
    n_classes = len(class_names)

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    df = pd.read_csv(FEATURES_CSV)
    splits = pd.read_csv(SPLITS_CSV)
    data = df.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")

    missing_features = [c for c in feature_cols if c not in data.columns]
    if missing_features:
        raise ValueError(f"Features missing from dataset: {missing_features[:5]}")

    x_all = data[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_labels = data["category"].astype(str).to_numpy()
    unknown_labels = sorted(set(y_labels).difference(class_to_idx.keys()))
    if unknown_labels:
        raise ValueError(f"Labels not seen by teacher classes: {unknown_labels[:5]}")
    y_all = np.array([class_to_idx[name] for name in y_labels], dtype=np.int64)

    train_mask = data["split"] == "train"
    val_mask = data["split"] == "val"
    test_mask = data["split"] == "test"

    x_train = x_all[train_mask.to_numpy()]
    y_train = y_all[train_mask.to_numpy()]
    x_val = x_all[val_mask.to_numpy()]
    y_val = y_all[val_mask.to_numpy()]
    x_test = x_all[test_mask.to_numpy()]
    y_test = y_all[test_mask.to_numpy()]

    if args.augment:
        x_train_used, y_train_used = augment_with_noise(
            x_train,
            y_train,
            n_copies=args.n_copies,
            noise_frac=args.noise_frac,
            seed=42,
        )
    else:
        x_train_used, y_train_used = x_train, y_train

    teacher_components = [teacher_name]
    if args.teacher_mode == "single":
        teacher_soft_train = _teacher_probs(teacher_model, x_train_used, n_classes, args.temperature)
        teacher_soft_val = _teacher_probs(teacher_model, x_val, n_classes, args.temperature)
    else:
        aux_teacher = _build_hgb_teacher(args)
        aux_teacher.fit(x_train_used, y_train_used)
        primary_train = _teacher_probs(teacher_model, x_train_used, n_classes, args.temperature)
        primary_val = _teacher_probs(teacher_model, x_val, n_classes, args.temperature)
        aux_train = _teacher_probs(aux_teacher, x_train_used, n_classes, args.temperature)
        aux_val = _teacher_probs(aux_teacher, x_val, n_classes, args.temperature)
        teacher_soft_train = _blend_probs(
            [primary_train, aux_train],
            [args.teacher_primary_weight, args.teacher_aux_weight],
        )
        teacher_soft_val = _blend_probs(
            [primary_val, aux_val],
            [args.teacher_primary_weight, args.teacher_aux_weight],
        )
        teacher_components.append("hgb_aux")

    hard_train = np.zeros_like(teacher_soft_train)
    hard_train[np.arange(len(y_train_used)), y_train_used] = 1.0
    train_blended = float(args.alpha) * teacher_soft_train + (1.0 - float(args.alpha)) * hard_train
    pseudo_labels = np.argmax(train_blended, axis=1).astype(np.int64)
    pseudo_weights = np.max(train_blended, axis=1).astype(np.float64)

    hard_val = np.zeros_like(teacher_soft_val)
    hard_val[np.arange(len(y_val)), y_val] = 1.0
    val_blended = float(args.alpha) * teacher_soft_val + (1.0 - float(args.alpha)) * hard_val

    student_candidates = _build_student_candidates(args)
    trained_students: list[tuple[str, Pipeline, dict[str, Any], dict[str, Any]]] = []
    for student_name, student_model in student_candidates:
        if student_name in {"kd_hgb_tiny", "kd_pseudo_logreg"}:
            student_model.fit(
                x_train_used,
                pseudo_labels,
                clf__sample_weight=pseudo_weights,
            )
        else:
            patience = args.mlp_patience if student_name == "distilled_mlp_tiny" else args.patience
            student_model.fit(
                x_train_used,
                y_train_used,
                clf__soft_targets=teacher_soft_train,
                clf__alpha=float(args.alpha),
                clf__x_val=x_val,
                clf__y_val_blended=val_blended,
                clf__patience=patience,
            )
        val_metrics = _eval(student_model, x_val, y_val)
        test_metrics = _eval(student_model, x_test, y_test)
        trained_students.append((student_name, student_model, val_metrics, test_metrics))

    best_student_name, best_student, student_val, student_test = max(
        trained_students,
        key=lambda row: (float(row[2]["macro_f1"]), float(row[2]["accuracy"])),
    )

    calibration_summary: dict[str, float | bool] = {
        "enabled": bool(args.calibrate_temperature),
        "temperature": 1.0,
        "pre_nll": 0.0,
        "post_nll": 0.0,
    }
    if bool(args.calibrate_temperature):
        calibrated, cal_info = TemperatureScaledClassifier.fit_from_validation(
            base_model=best_student,
            x_val=x_val,
            y_val=y_val,
        )
        best_student = calibrated
        calibration_summary = {
            "enabled": True,
            "temperature": float(cal_info.get("temperature", 1.0)),
            "pre_nll": float(cal_info.get("pre_nll", 0.0)),
            "post_nll": float(cal_info.get("post_nll", 0.0)),
        }

    student_val = _eval(best_student, x_val, y_val)
    student_test = _eval(best_student, x_test, y_test)

    teacher_val = _eval(teacher_model, x_val, y_val)
    teacher_test = _eval(teacher_model, x_test, y_test)

    _write_confusion(y_val, student_val["pred"], ARTIFACTS / "confusion_matrix_distilled_student_val.csv")
    _write_confusion(y_test, student_test["pred"], ARTIFACTS / "confusion_matrix_distilled_student_test.csv")

    for student_name, _, val_metrics, test_metrics in trained_students:
        _write_confusion(
            y_val,
            val_metrics["pred"],
            ARTIFACTS / f"confusion_matrix_{student_name}_val.csv",
        )
        _write_confusion(
            y_test,
            test_metrics["pred"],
            ARTIFACTS / f"confusion_matrix_{student_name}_test.csv",
        )

    model_name = best_student_name
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    report_path = Path(args.report_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distill_config = {
        "teacher_mode": str(args.teacher_mode),
        "teacher_primary_weight": float(args.teacher_primary_weight),
        "teacher_aux_weight": float(args.teacher_aux_weight),
        "teacher_components": teacher_components,
        "calibrate_temperature": bool(args.calibrate_temperature),
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "student": str(args.student),
        "learning_rate": float(args.learning_rate),
        "max_iter": int(args.max_iter),
        "batch_size": int(args.batch_size),
        "l2": float(args.l2),
        "patience": int(args.patience),
        "mlp_hidden_dim": int(args.mlp_hidden_dim),
        "mlp_learning_rate": float(args.mlp_learning_rate),
        "mlp_max_iter": int(args.mlp_max_iter),
        "mlp_batch_size": int(args.mlp_batch_size),
        "mlp_l2": float(args.mlp_l2),
        "mlp_patience": int(args.mlp_patience),
        "hgb_learning_rate": float(args.hgb_learning_rate),
        "hgb_max_iter": int(args.hgb_max_iter),
        "hgb_max_depth": int(args.hgb_max_depth),
        "hgb_max_leaf_nodes": int(args.hgb_max_leaf_nodes),
        "hgb_min_samples_leaf": int(args.hgb_min_samples_leaf),
        "hgb_l2": float(args.hgb_l2),
        "pseudo_lr_c": float(args.pseudo_lr_c),
        "augment": bool(args.augment),
        "n_copies": int(args.n_copies),
        "noise_frac": float(args.noise_frac),
    }

    candidates_report: dict[str, dict[str, float]] = {}
    for student_name, _, val_metrics, test_metrics in trained_students:
        candidates_report[student_name] = {
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
        }

    artifact = {
        "model": best_student,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "classes": class_names,
        "selection_metric": "distillation_student_test_macro_f1",
        "selection_value": float(student_test["macro_f1"]),
        "selection_metrics": {
            "val_accuracy": float(student_val["accuracy"]),
            "val_macro_f1": float(student_val["macro_f1"]),
            "test_accuracy": float(student_test["accuracy"]),
            "test_macro_f1": float(student_test["macro_f1"]),
        },
        "trained_on": "train (optionally augmented)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model_name": teacher_name,
        "teacher_components": teacher_components,
        "teacher_artifact_path": str(teacher_artifact_path),
        "teacher_selection_metric": str(teacher_artifact.get("selection_metric", "unknown")),
        "teacher_selection_value": _to_float(teacher_artifact.get("selection_value", "unknown")),
        "calibration": calibration_summary,
        "distillation": distill_config,
    }

    joblib.dump(artifact, output_path)

    metadata = {k: v for k, v in artifact.items() if k != "model"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    teacher_artifact_size_mb = _artifact_size_mb(teacher_artifact_path)
    student_latency = _latency_metrics(best_student, x_test)
    teacher_latency = _latency_metrics(teacher_model, x_test)
    student_val_probs = np.asarray(best_student.predict_proba(x_val), dtype=np.float64)
    student_test_probs = np.asarray(best_student.predict_proba(x_test), dtype=np.float64)

    calib_dir = ARTIFACTS / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    classwise_val_df = _classwise_confidence_table(y_val, student_val_probs, class_names)
    classwise_test_df = _classwise_confidence_table(y_test, student_test_probs, class_names)
    classwise_val_csv = calib_dir / "classwise_confidence_val.csv"
    classwise_test_csv = calib_dir / "classwise_confidence_test.csv"
    classwise_val_df.to_csv(classwise_val_csv, index=False)
    classwise_test_df.to_csv(classwise_test_csv, index=False)

    rel_val_df = _reliability_bins(y_val, student_val_probs, n_bins=10)
    rel_test_df = _reliability_bins(y_test, student_test_probs, n_bins=10)
    rel_val_csv = calib_dir / "reliability_bins_val.csv"
    rel_test_csv = calib_dir / "reliability_bins_test.csv"
    rel_val_df.to_csv(rel_val_csv, index=False)
    rel_test_df.to_csv(rel_test_csv, index=False)

    rel_val_png = calib_dir / "reliability_plot_val.png"
    rel_test_png = calib_dir / "reliability_plot_test.png"
    rel_val_plot_ok = _save_reliability_plot(rel_val_df, rel_val_png, title="Validation Reliability")
    rel_test_plot_ok = _save_reliability_plot(rel_test_df, rel_test_png, title="Test Reliability")

    worst_test_gap = (
        classwise_test_df.assign(abs_gap=np.abs(classwise_test_df["confidence_gap"]))
        .sort_values("abs_gap", ascending=False)
        .head(5)
        .drop(columns=["abs_gap"])
    )
    worst_test_gap_records = worst_test_gap.to_dict(orient="records")
    action_recommendations = _build_action_recommendations(classwise_test_df, top_n=3)

    report = {
        "teacher": {
            "model_name": teacher_name,
            "val_accuracy": float(teacher_val["accuracy"]),
            "val_macro_f1": float(teacher_val["macro_f1"]),
            "test_accuracy": float(teacher_test["accuracy"]),
            "test_macro_f1": float(teacher_test["macro_f1"]),
        },
        "student": {
            "model_name": model_name,
            "val_accuracy": float(student_val["accuracy"]),
            "val_macro_f1": float(student_val["macro_f1"]),
            "test_accuracy": float(student_test["accuracy"]),
            "test_macro_f1": float(student_test["macro_f1"]),
        },
        "student_candidates": candidates_report,
        "selection": {
            "criterion": "best validation macro_f1, then validation accuracy",
            "selected_student": model_name,
        },
        "delta_student_minus_teacher": {
            "val_accuracy": float(student_val["accuracy"] - teacher_val["accuracy"]),
            "val_macro_f1": float(student_val["macro_f1"] - teacher_val["macro_f1"]),
            "test_accuracy": float(student_test["accuracy"] - teacher_test["accuracy"]),
            "test_macro_f1": float(student_test["macro_f1"] - teacher_test["macro_f1"]),
        },
        "distillation": distill_config,
        "calibration": calibration_summary,
        "confidence_quality": {
            "val_nll": _nll_score(y_val, student_val_probs),
            "test_nll": _nll_score(y_test, student_test_probs),
            "val_ece": _ece_score(y_val, student_val_probs, n_bins=10),
            "test_ece": _ece_score(y_test, student_test_probs, n_bins=10),
            "val_top1_confidence_mean": float(np.mean(np.max(student_val_probs, axis=1))),
            "test_top1_confidence_mean": float(np.mean(np.max(student_test_probs, axis=1))),
            "val_top3_accuracy": _topk_accuracy(y_val, student_val_probs, k=3),
            "test_top3_accuracy": _topk_accuracy(y_test, student_test_probs, k=3),
            "worst_5_test_classes_by_confidence_gap": worst_test_gap_records,
        },
        "action_recommendations": {
            "top_3_classes_for_data_collection": action_recommendations,
        },
        "samples": {
            "train": int(len(y_train)),
            "train_used": int(len(y_train_used)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "outputs": {
            "student_artifact": str(output_path),
            "student_metadata": str(metadata_path),
            "val_confusion": str(ARTIFACTS / "confusion_matrix_distilled_student_val.csv"),
            "test_confusion": str(ARTIFACTS / "confusion_matrix_distilled_student_test.csv"),
            "classwise_confidence_val": str(classwise_val_csv),
            "classwise_confidence_test": str(classwise_test_csv),
            "reliability_bins_val": str(rel_val_csv),
            "reliability_bins_test": str(rel_test_csv),
            "reliability_plot_val": str(rel_val_png) if rel_val_plot_ok else "not_generated",
            "reliability_plot_test": str(rel_test_png) if rel_test_plot_ok else "not_generated",
        },
        "runtime_benchmarks": {
            "teacher": {
                **teacher_latency,
                "artifact_size_mb": float(teacher_artifact_size_mb),
            },
            "selected_student": {
                **student_latency,
            },
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    student_artifact_size_mb = _artifact_size_mb(output_path)
    report["runtime_benchmarks"]["selected_student"]["artifact_size_mb"] = float(student_artifact_size_mb)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Teacher: {teacher_name}")
    print(
        "Teacher      | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
            teacher_val["accuracy"],
            teacher_val["macro_f1"],
            teacher_test["accuracy"],
            teacher_test["macro_f1"],
        )
    )
    for student_name, _, val_metrics, test_metrics in trained_students:
        print(
            "Candidate {} | val_acc={:.4f} val_f1={:.4f} test_acc={:.4f} test_f1={:.4f}".format(
                student_name,
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                test_metrics["accuracy"],
                test_metrics["macro_f1"],
            )
        )
    print(f"Selected student: {model_name}")
    print(f"Saved student artifact: {output_path}")
    print(f"Saved student metadata: {metadata_path}")
    print(f"Saved distillation report: {report_path}")


if __name__ == "__main__":
    main()
