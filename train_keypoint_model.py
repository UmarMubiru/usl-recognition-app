from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_CSV = ROOT / "DATASET_ug_sign_language" / "sign_annotations.csv"
KEYPOINT_DIR = ROOT / "DATASET_ug_sign_language" / "features"
OUT_MODEL = Path(__file__).resolve().parent / "model.pkl"
OUT_META = Path(__file__).resolve().parent / "model_metadata.json"

LEFT_START = 501
LEFT_END = 522
RIGHT_START = 522
RIGHT_END = 543


def _pick_hand(frames_xyz: np.ndarray) -> np.ndarray:
    left = frames_xyz[:, LEFT_START:LEFT_END, :]
    right = frames_xyz[:, RIGHT_START:RIGHT_END, :]

    left_energy = float(np.linalg.norm(left.reshape(left.shape[0], -1), axis=1).mean())
    right_energy = float(np.linalg.norm(right.reshape(right.shape[0], -1), axis=1).mean())

    hand = right if right_energy >= left_energy else left

    valid = np.any(np.abs(hand) > 1e-9, axis=(1, 2))
    if np.any(valid):
        hand = hand[valid]
    return hand


def _hand_to_feature_63(frames_xyz: np.ndarray) -> np.ndarray:
    hand = _pick_hand(frames_xyz)
    mean_hand = hand.mean(axis=0)

    wrist = mean_hand[0:1, :]
    centered = mean_hand - wrist

    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 1e-9:
        scale = 1.0

    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)


def build_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(ANNOTATIONS_CSV)
    rows = []

    for _, row in df.iterrows():
        video_id = str(row["video_ID"])
        label = str(row["sign_word"])
        stem = Path(video_id).stem
        kp_path = KEYPOINT_DIR / f"{stem}_keypoints.npy"

        if not kp_path.exists():
            continue

        arr = np.load(kp_path)
        if arr.ndim != 3 or arr.shape[1:] != (543, 3):
            continue

        feature = _hand_to_feature_63(arr)
        rows.append((feature, label, stem))

    if not rows:
        raise RuntimeError("No valid samples found to train keypoint model")

    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    ids = [r[2] for r in rows]
    return X, y, ids


def main() -> None:
    X, y, ids = build_dataset()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)
    joblib.dump(model, OUT_MODEL)

    meta = {
        "trained_from": str(ANNOTATIONS_CSV),
        "samples": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "unique_labels": int(np.unique(y).shape[0]),
        "labels": sorted(np.unique(y).tolist()),
        "note": "63-feature hand-landmark model for Streamlit webcam/image demo",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {OUT_MODEL}")
    print(f"Saved metadata: {OUT_META}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {np.unique(y).shape[0]}")


if __name__ == "__main__":
    main()
