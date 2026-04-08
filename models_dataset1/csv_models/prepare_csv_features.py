from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

MP_HOLISTIC_AVAILABLE = bool(
    mp is not None
    and hasattr(mp, "solutions")
    and hasattr(mp.solutions, "holistic")
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "SIGN LANGUAGE DISEASES FINISHED"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"ALPHABETS AND NUMBERS", "UNIQUE WORDS"}

# Compact HOG descriptor – each sampled frame is resized to 64×64 before computing
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(32, 32),
    _blockStride=(32, 32),
    _cellSize=(16, 16),
    _nbins=9,
)
_HOG_DIM: int = int(_HOG.getDescriptorSize())  # 144


def _landmarks_to_array(landmarks, max_points: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((0, 3), dtype=np.float32)
    coords = []
    for point in landmarks.landmark[:max_points]:
        coords.append([point.x, point.y, point.z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def video_metadata(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    sampled_gray_means: list[float] = []
    sampled_gray_stds: list[float] = []
    motion_diffs: list[float] = []

    pose_presence = 0
    left_presence = 0
    right_presence = 0
    pose_spreads: list[float] = []
    hand_spreads: list[float] = []
    pose_motion: list[float] = []
    hand_motion: list[float] = []
    hog_descs: list[np.ndarray] = []

    prev_gray = None
    prev_pose_arr = None
    prev_left_arr = None
    prev_right_arr = None
    frame_idx = 0
    stride = 12

    if MP_HOLISTIC_AVAILABLE:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % stride == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sampled_gray_means.append(float(gray.mean()))
                    sampled_gray_stds.append(float(gray.std()))
                    hog_descs.append(_HOG.compute(cv2.resize(gray, (64, 64))).flatten())

                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        motion_diffs.append(float(diff.mean()))
                    prev_gray = gray

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)

                    pose_arr = _landmarks_to_array(result.pose_landmarks, 33)
                    left_arr = _landmarks_to_array(result.left_hand_landmarks, 21)
                    right_arr = _landmarks_to_array(result.right_hand_landmarks, 21)

                    if len(pose_arr):
                        pose_presence += 1
                        pose_spreads.append(float(np.std(pose_arr[:, :2])))
                        if prev_pose_arr is not None and len(prev_pose_arr) == len(pose_arr):
                            pose_motion.append(float(np.linalg.norm(pose_arr[:, :2] - prev_pose_arr[:, :2], axis=1).mean()))
                        prev_pose_arr = pose_arr

                    if len(left_arr):
                        left_presence += 1
                        hand_spreads.append(float(np.std(left_arr[:, :2])))
                        if prev_left_arr is not None and len(prev_left_arr) == len(left_arr):
                            hand_motion.append(float(np.linalg.norm(left_arr[:, :2] - prev_left_arr[:, :2], axis=1).mean()))
                        prev_left_arr = left_arr

                    if len(right_arr):
                        right_presence += 1
                        hand_spreads.append(float(np.std(right_arr[:, :2])))
                        if prev_right_arr is not None and len(prev_right_arr) == len(right_arr):
                            hand_motion.append(float(np.linalg.norm(right_arr[:, :2] - prev_right_arr[:, :2], axis=1).mean()))
                        prev_right_arr = right_arr

                frame_idx += 1
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sampled_gray_means.append(float(gray.mean()))
                sampled_gray_stds.append(float(gray.std()))
                hog_descs.append(_HOG.compute(cv2.resize(gray, (64, 64))).flatten())

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion_diffs.append(float(diff.mean()))
                prev_gray = gray

            frame_idx += 1

    cap.release()

    if hog_descs:
        hog_arr = np.stack(hog_descs, axis=0)
        hog_mean = hog_arr.mean(axis=0)
        hog_std = hog_arr.std(axis=0)
    else:
        hog_mean = np.zeros(_HOG_DIM, dtype=np.float32)
        hog_std = np.zeros(_HOG_DIM, dtype=np.float32)

    duration_sec = frame_count / fps if fps > 0 else 0.0

    def safe_stats(values: list[float]) -> tuple[float, float, float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float32)
        return (
            float(arr.mean()),
            float(arr.std()),
            float(np.percentile(arr, 10)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            float(np.percentile(arr, 90)),
        )

    def segment_means(values: list[float]) -> tuple[float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float32)
        n = len(arr)
        seg = max(1, n // 3)
        start = arr[:seg]
        middle = arr[seg : 2 * seg] if n >= 2 * seg else arr[seg:]
        end = arr[2 * seg :] if n > 2 * seg else arr[-seg:]
        trend = float(end.mean() - start.mean())
        return float(start.mean()), float(middle.mean() if len(middle) else 0.0), float(end.mean()), trend

    def slope(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        y = np.asarray(values, dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        return float(np.polyfit(x, y, 1)[0])

    gray_mean_mean, gray_mean_std, gray_mean_p10, gray_mean_p25, gray_mean_p75, gray_mean_p90 = safe_stats(sampled_gray_means)
    gray_std_mean, gray_std_std, gray_std_p10, gray_std_p25, gray_std_p75, gray_std_p90 = safe_stats(sampled_gray_stds)
    motion_mean, motion_std, motion_p10, motion_p25, motion_p75, motion_p90 = safe_stats(motion_diffs)
    pose_spread_mean, pose_spread_std, _, _, _, _ = safe_stats(pose_spreads)
    hand_spread_mean, hand_spread_std, _, _, _, _ = safe_stats(hand_spreads)
    pose_motion_mean, pose_motion_std, _, _, _, _ = safe_stats(pose_motion)
    hand_motion_mean, hand_motion_std, _, _, _, _ = safe_stats(hand_motion)

    gray_start, gray_middle, gray_end, gray_trend = segment_means(sampled_gray_means)
    motion_start, motion_middle, motion_end, motion_trend = segment_means(motion_diffs)

    gray_slope = slope(sampled_gray_means)
    motion_slope = slope(motion_diffs)

    motion_to_duration = motion_mean / max(duration_sec, 1e-6)
    motion_to_frames = motion_mean / max(frame_count, 1.0)
    intensity_to_motion = gray_std_mean / max(motion_mean, 1e-6)

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "aspect_ratio": (width / height) if height > 0 else 0.0,
        "gray_mean_mean": gray_mean_mean,
        "gray_mean_std": gray_mean_std,
        "gray_mean_p10": gray_mean_p10,
        "gray_mean_p25": gray_mean_p25,
        "gray_mean_p75": gray_mean_p75,
        "gray_mean_p90": gray_mean_p90,
        "gray_std_mean": gray_std_mean,
        "gray_std_std": gray_std_std,
        "gray_std_p10": gray_std_p10,
        "gray_std_p25": gray_std_p25,
        "gray_std_p75": gray_std_p75,
        "gray_std_p90": gray_std_p90,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "motion_p10": motion_p10,
        "motion_p25": motion_p25,
        "motion_p75": motion_p75,
        "motion_p90": motion_p90,
        "gray_start": gray_start,
        "gray_middle": gray_middle,
        "gray_end": gray_end,
        "gray_trend": gray_trend,
        "motion_start": motion_start,
        "motion_middle": motion_middle,
        "motion_end": motion_end,
        "motion_trend": motion_trend,
        "gray_slope": gray_slope,
        "motion_slope": motion_slope,
        "motion_to_duration": motion_to_duration,
        "motion_to_frames": motion_to_frames,
        "intensity_to_motion": intensity_to_motion,
        "pose_presence_ratio": float(pose_presence) / max(float(len(sampled_gray_means)), 1.0),
        "left_hand_presence_ratio": float(left_presence) / max(float(len(sampled_gray_means)), 1.0),
        "right_hand_presence_ratio": float(right_presence) / max(float(len(sampled_gray_means)), 1.0),
        "pose_spread_mean": pose_spread_mean,
        "pose_spread_std": pose_spread_std,
        "hand_spread_mean": hand_spread_mean,
        "hand_spread_std": hand_spread_std,
        "pose_motion_mean": pose_motion_mean,
        "pose_motion_std": pose_motion_std,
        "hand_motion_mean": hand_motion_mean,
        "hand_motion_std": hand_motion_std,
        "sampled_frames": float(len(sampled_gray_means)),
        "sampled_motion_steps": float(len(motion_diffs)),
        **{f"hog_mean_{i}": float(hog_mean[i]) for i in range(_HOG_DIM)},
        **{f"hog_std_{i}": float(hog_std[i]) for i in range(_HOG_DIM)},
    }


def build_disease_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name.strip()
        if category in EXCLUDE:
            continue

        for video_path in sorted(category_dir.rglob("*.mp4")):
            rel = video_path.relative_to(DATASET_ROOT).as_posix()
            sample_id = rel.replace("/", "__").replace(".mp4", "")
            rows.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "video_path": str(video_path),
                    **video_metadata(video_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No disease-only videos found in Dataset 1")
    return df.sort_values(["category", "sample_id"]).reset_index(drop=True)


def build_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["category"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["category"],
    )

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
    df = build_disease_inventory()
    split_df = build_splits(df)

    features_path = OUT_DIR / "dataset1_disease_features.csv"
    splits_path = OUT_DIR / "dataset1_disease_splits.csv"

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
