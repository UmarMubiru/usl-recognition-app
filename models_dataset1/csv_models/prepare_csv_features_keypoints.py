from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "DATASET_ug_sign_language"
FEATURES_DIR = DATASET_ROOT / "features"
ANNOTATIONS_CSV = DATASET_ROOT / "sign_annotations.csv"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _part_features(kps: np.ndarray, start: int, end: int, name: str) -> dict[str, float]:
    part = kps[:, start:end, :]
    xy = part[..., :2]
    frame_energy = np.linalg.norm(xy, axis=-1).mean(axis=1)
    presence = (frame_energy > 1e-6).astype(np.float32)

    spread = xy.std(axis=1).mean(axis=-1)
    motion = np.zeros_like(frame_energy)
    if len(frame_energy) > 1:
        delta = np.diff(xy, axis=0)
        motion[1:] = np.linalg.norm(delta, axis=-1).mean(axis=1)

    feats = {
        f"{name}_presence_ratio": float(presence.mean()) if len(presence) else 0.0,
    }
    feats.update(_safe_stats(spread, f"{name}_spread"))
    feats.update(_safe_stats(motion, f"{name}_motion"))
    return feats


def keypoint_features(keypoint_path: Path) -> dict[str, float]:
    kps = np.load(keypoint_path)
    if kps.ndim != 3:
        kps = np.asarray(kps).reshape(len(kps), -1, 3)

    t, v, c = kps.shape
    flat = kps.reshape(t, -1)
    velocity = np.diff(flat, axis=0) if t > 1 else np.zeros((0, flat.shape[1]), dtype=np.float64)
    accel = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else np.zeros((0, flat.shape[1]), dtype=np.float64)

    frame_norm = np.linalg.norm(kps[..., :2], axis=-1).mean(axis=1)
    frame_z = np.abs(kps[..., 2]).mean(axis=1)

    feats: dict[str, float] = {
        "num_frames": float(t),
        "num_landmarks": float(v),
        "num_dims": float(c),
    }
    feats.update(_safe_stats(frame_norm, "frame_norm"))
    feats.update(_safe_stats(frame_z, "frame_z_abs"))
    feats.update(_safe_stats(np.abs(velocity).mean(axis=1) if velocity.size else np.array([]), "velocity_abs"))
    feats.update(_safe_stats(np.abs(accel).mean(axis=1) if accel.size else np.array([]), "accel_abs"))

    # Holistic landmark index ranges (pose, face, left hand, right hand)
    feats.update(_part_features(kps, 0, 33, "pose"))
    feats.update(_part_features(kps, 33, 501, "face"))
    feats.update(_part_features(kps, 501, 522, "left_hand"))
    feats.update(_part_features(kps, 522, 543, "right_hand"))

    if t >= 3:
        seg = max(1, t // 3)
        start = frame_norm[:seg]
        middle = frame_norm[seg : 2 * seg] if t >= 2 * seg else frame_norm[seg:]
        end = frame_norm[2 * seg :] if t > 2 * seg else frame_norm[-seg:]
        feats["frame_norm_start"] = float(start.mean()) if len(start) else 0.0
        feats["frame_norm_middle"] = float(middle.mean()) if len(middle) else 0.0
        feats["frame_norm_end"] = float(end.mean()) if len(end) else 0.0
        feats["frame_norm_trend"] = feats["frame_norm_end"] - feats["frame_norm_start"]
    else:
        feats["frame_norm_start"] = 0.0
        feats["frame_norm_middle"] = 0.0
        feats["frame_norm_end"] = 0.0
        feats["frame_norm_trend"] = 0.0

    return feats


def build_dataset() -> pd.DataFrame:
    ann = pd.read_csv(ANNOTATIONS_CSV)
    ann = ann.rename(columns={"video_ID": "video_id", "sign_word": "category"})
    rows: list[dict[str, object]] = []

    for row in ann.itertuples(index=False):
        video_id = str(row.video_id)
        category = str(row.category).strip()
        sample_id = video_id.replace(".mp4", "")

        kp_path = FEATURES_DIR / f"{sample_id}_keypoints.npy"
        if not kp_path.exists():
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "keypoint_path": str(kp_path),
                **keypoint_features(kp_path),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No keypoint samples found from existing dataset")
    return df.sort_values(["category", "sample_id"]).reset_index(drop=True)


def build_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    counts = df["category"].value_counts()
    use_stratify = bool((counts >= 2).all() and df["category"].nunique() > 1)

    strat = df["category"] if use_stratify else None
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)

    temp_counts = temp_df["category"].value_counts()
    use_stratify_temp = bool((temp_counts >= 2).all() and temp_df["category"].nunique() > 1)
    strat_temp = temp_df["category"] if use_stratify_temp else None
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=strat_temp)

    out = pd.concat(
        [
            train_df[["sample_id", "category"]].assign(split="train"),
            val_df[["sample_id", "category"]].assign(split="val"),
            test_df[["sample_id", "category"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    return out.sort_values(["split", "category", "sample_id"]).reset_index(drop=True)


def main() -> None:
    df = build_dataset()
    split_df = build_splits(df)

    features_path = OUT_DIR / "existing_keypoints_features.csv"
    splits_path = OUT_DIR / "existing_keypoints_splits.csv"

    df.to_csv(features_path, index=False)
    split_df.to_csv(splits_path, index=False)

    print(f"Saved features: {features_path}")
    print(f"Saved splits:   {splits_path}")
    print()
    print(f"Samples: {len(df)}")
    print(f"Classes: {df['category'].nunique()}")
    print(split_df['split'].value_counts().to_string())


if __name__ == "__main__":
    main()
