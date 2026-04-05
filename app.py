from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT = APP_DIR / "model.pkl"
FALLBACK_ARTIFACT = APP_DIR.parent / "models_dataset1" / "csv_models" / "artifacts" / "best_model.joblib"
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(LOCAL_ARTIFACT if LOCAL_ARTIFACT.exists() else FALLBACK_ARTIFACT)))

_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(32, 32),
    _blockStride=(32, 32),
    _cellSize=(16, 16),
    _nbins=9,
)
_HOG_DIM: int = int(_HOG.getDescriptorSize())


@st.cache_resource
def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = joblib.load(path)
    required = {"model", "model_name", "feature_cols", "classes"}
    if not isinstance(payload, dict) or required.difference(payload.keys()):
        raise ValueError("Artifact must be dict with keys: model, model_name, feature_cols, classes")
    return payload


def _landmarks_to_array(landmarks, max_points: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((0, 3), dtype=np.float32)
    coords = []
    for point in landmarks.landmark[:max_points]:
        coords.append([point.x, point.y, point.z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _safe_stats(values: list[float]) -> tuple[float, float, float, float, float, float]:
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


def _segment_means(values: list[float]) -> tuple[float, float, float, float]:
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


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float32)
    x = np.arange(len(y), dtype=np.float32)
    return float(np.polyfit(x, y, 1)[0])


def _empty_state() -> dict[str, Any]:
    return {
        "sampled_gray_means": [],
        "sampled_gray_stds": [],
        "motion_diffs": [],
        "pose_presence": 0,
        "left_presence": 0,
        "right_presence": 0,
        "pose_spreads": [],
        "hand_spreads": [],
        "pose_motion": [],
        "hand_motion": [],
        "hog_descs": [],
        "prev_gray": None,
        "prev_pose_arr": None,
        "prev_left_arr": None,
        "prev_right_arr": None,
    }


def _update_from_frame(frame_bgr: np.ndarray, state: dict[str, Any], holistic) -> None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    state["sampled_gray_means"].append(float(gray.mean()))
    state["sampled_gray_stds"].append(float(gray.std()))
    state["hog_descs"].append(_HOG.compute(cv2.resize(gray, (64, 64))).flatten())

    prev_gray = state["prev_gray"]
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        state["motion_diffs"].append(float(diff.mean()))
    state["prev_gray"] = gray

    if holistic is None:
        return

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)

    pose_arr = _landmarks_to_array(result.pose_landmarks, 33)
    left_arr = _landmarks_to_array(result.left_hand_landmarks, 21)
    right_arr = _landmarks_to_array(result.right_hand_landmarks, 21)

    if len(pose_arr):
        state["pose_presence"] += 1
        state["pose_spreads"].append(float(np.std(pose_arr[:, :2])))
        prev_pose = state["prev_pose_arr"]
        if prev_pose is not None and len(prev_pose) == len(pose_arr):
            motion_val = np.linalg.norm(pose_arr[:, :2] - prev_pose[:, :2], axis=1).mean()
            state["pose_motion"].append(float(motion_val))
        state["prev_pose_arr"] = pose_arr

    if len(left_arr):
        state["left_presence"] += 1
        state["hand_spreads"].append(float(np.std(left_arr[:, :2])))
        prev_left = state["prev_left_arr"]
        if prev_left is not None and len(prev_left) == len(left_arr):
            motion_val = np.linalg.norm(left_arr[:, :2] - prev_left[:, :2], axis=1).mean()
            state["hand_motion"].append(float(motion_val))
        state["prev_left_arr"] = left_arr

    if len(right_arr):
        state["right_presence"] += 1
        state["hand_spreads"].append(float(np.std(right_arr[:, :2])))
        prev_right = state["prev_right_arr"]
        if prev_right is not None and len(prev_right) == len(right_arr):
            motion_val = np.linalg.norm(right_arr[:, :2] - prev_right[:, :2], axis=1).mean()
            state["hand_motion"].append(float(motion_val))
        state["prev_right_arr"] = right_arr


def _finalize_features(state: dict[str, Any], fps: float, frame_count: float, width: float, height: float) -> dict[str, float]:
    hog_descs = state["hog_descs"]
    if hog_descs:
        hog_arr = np.stack(hog_descs, axis=0)
        hog_mean = hog_arr.mean(axis=0)
        hog_std = hog_arr.std(axis=0)
    else:
        hog_mean = np.zeros(_HOG_DIM, dtype=np.float32)
        hog_std = np.zeros(_HOG_DIM, dtype=np.float32)

    duration_sec = frame_count / fps if fps > 0 else 0.0

    gray_mean_mean, gray_mean_std, gray_mean_p10, gray_mean_p25, gray_mean_p75, gray_mean_p90 = _safe_stats(state["sampled_gray_means"])
    gray_std_mean, gray_std_std, gray_std_p10, gray_std_p25, gray_std_p75, gray_std_p90 = _safe_stats(state["sampled_gray_stds"])
    motion_mean, motion_std, motion_p10, motion_p25, motion_p75, motion_p90 = _safe_stats(state["motion_diffs"])
    pose_spread_mean, pose_spread_std, _, _, _, _ = _safe_stats(state["pose_spreads"])
    hand_spread_mean, hand_spread_std, _, _, _, _ = _safe_stats(state["hand_spreads"])
    pose_motion_mean, pose_motion_std, _, _, _, _ = _safe_stats(state["pose_motion"])
    hand_motion_mean, hand_motion_std, _, _, _, _ = _safe_stats(state["hand_motion"])

    gray_start, gray_middle, gray_end, gray_trend = _segment_means(state["sampled_gray_means"])
    motion_start, motion_middle, motion_end, motion_trend = _segment_means(state["motion_diffs"])

    gray_slope = _slope(state["sampled_gray_means"])
    motion_slope = _slope(state["motion_diffs"])

    sampled_frames = float(len(state["sampled_gray_means"]))
    motion_to_duration = motion_mean / max(duration_sec, 1e-6)
    motion_to_frames = motion_mean / max(frame_count, 1.0)
    intensity_to_motion = gray_std_mean / max(motion_mean, 1e-6)

    features = {
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
        "pose_presence_ratio": float(state["pose_presence"]) / max(sampled_frames, 1.0),
        "left_hand_presence_ratio": float(state["left_presence"]) / max(sampled_frames, 1.0),
        "right_hand_presence_ratio": float(state["right_presence"]) / max(sampled_frames, 1.0),
        "pose_spread_mean": pose_spread_mean,
        "pose_spread_std": pose_spread_std,
        "hand_spread_mean": hand_spread_mean,
        "hand_spread_std": hand_spread_std,
        "pose_motion_mean": pose_motion_mean,
        "pose_motion_std": pose_motion_std,
        "hand_motion_mean": hand_motion_mean,
        "hand_motion_std": hand_motion_std,
        "sampled_frames": sampled_frames,
        "sampled_motion_steps": float(len(state["motion_diffs"])),
        **{f"hog_mean_{i}": float(hog_mean[i]) for i in range(_HOG_DIM)},
        **{f"hog_std_{i}": float(hog_std[i]) for i in range(_HOG_DIM)},
    }
    return features


def extract_features_from_video(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    state = _empty_state()
    stride = 12
    frame_idx = 0

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
                _update_from_frame(frame, state, holistic)
            frame_idx += 1

    cap.release()
    return _finalize_features(state, fps=fps, frame_count=frame_count, width=width, height=height)


def extract_features_from_image(image_rgb: np.ndarray) -> dict[str, float]:
    h, w = image_rgb.shape[:2]
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    state = _empty_state()

    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        _update_from_frame(frame_bgr, state, holistic)

    return _finalize_features(state, fps=1.0, frame_count=1.0, width=float(w), height=float(h))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom <= 0:
        return np.full_like(exp_vals, 1.0 / len(exp_vals), dtype=float)
    return exp_vals / denom


def predict_with_artifact(
    model,
    classes: list[str],
    feature_cols: list[str],
    features: dict[str, float],
    top_k: int,
) -> tuple[str, float, list[tuple[str, float]], int, int]:
    missing_count = sum(1 for feat in feature_cols if feat not in features)
    unknown_count = sum(1 for feat in features if feat not in feature_cols)
    x_vec = np.array([float(features.get(feat, 0.0)) for feat in feature_cols], dtype=float).reshape(1, -1)

    pred_idx = int(model.predict(x_vec)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)[0].astype(float)
    elif hasattr(model, "decision_function"):
        decision = np.array(model.decision_function(x_vec)).reshape(-1)
        if decision.shape[0] == 1 and len(classes) == 2:
            p1 = 1.0 / (1.0 + np.exp(-decision[0]))
            probs = np.array([1.0 - p1, p1], dtype=float)
        else:
            probs = _softmax(decision)
    else:
        probs = np.zeros(len(classes), dtype=float)
        probs[pred_idx] = 1.0

    top_k = min(top_k, len(classes))
    top_idx = np.argsort(probs)[::-1][:top_k]
    ranked = [(classes[int(i)], float(probs[int(i)])) for i in top_idx]
    return classes[pred_idx], float(probs[pred_idx]), ranked, missing_count, unknown_count


def run_prediction_from_uploaded_video(uploaded_file) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        return extract_features_from_video(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


st.set_page_config(page_title="USL Recognition (338 Features)", layout="centered")
st.title("Uganda Sign Language Recognition")
st.caption("Interface aligned to deployed SVM-RBF (338 engineered features).")

try:
    artifact = load_artifact(ARTIFACT_PATH)
except Exception as exc:
    st.error(f"Could not load artifact: {exc}")
    st.info("Place model.pkl in streamlit_app/ or set MODEL_ARTIFACT_PATH to a valid artifact path.")
    st.stop()

model = artifact["model"]
model_name = str(artifact["model_name"])
feature_cols = list(artifact["feature_cols"])
classes = list(artifact["classes"])

st.subheader("Model Check")
st.write(f"Model: {model_name}")
st.write(f"Feature count expected: {len(feature_cols)}")
st.write(f"Artifact path: {ARTIFACT_PATH}")

if len(feature_cols) != 338:
    st.warning("This app is designed for the 338-feature pipeline.")

top_k = st.slider("Top-k predictions", min_value=1, max_value=10, value=3)
input_mode = st.radio("Choose input method", ["Upload Video", "Upload Image", "Use Webcam Snapshot"])

if input_mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a short sign video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Predict", key="predict_video"):
            with st.spinner("Extracting 338 features from video and predicting..."):
                features = run_prediction_from_uploaded_video(uploaded_video)
                label, confidence, ranked, missing_count, unknown_count = predict_with_artifact(
                    model=model,
                    classes=classes,
                    feature_cols=feature_cols,
                    features=features,
                    top_k=top_k,
                )
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2%}")
            st.write("Top predictions:")
            for name, score in ranked:
                st.write(f"- {name}: {score:.2%}")
            st.caption(f"Missing features: {missing_count} | Unknown features: {unknown_count}")

if input_mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Predict", key="predict_image"):
            with st.spinner("Extracting approximate 338 features from image and predicting..."):
                features = extract_features_from_image(image_np)
                label, confidence, ranked, missing_count, unknown_count = predict_with_artifact(
                    model=model,
                    classes=classes,
                    feature_cols=feature_cols,
                    features=features,
                    top_k=top_k,
                )
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2%}")
            st.write("Top predictions:")
            for name, score in ranked:
                st.write(f"- {name}: {score:.2%}")
            st.caption(f"Missing features: {missing_count} | Unknown features: {unknown_count}")

if input_mode == "Use Webcam Snapshot":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Captured image", use_container_width=True)
        if st.button("Predict", key="predict_camera"):
            with st.spinner("Extracting approximate 338 features from snapshot and predicting..."):
                features = extract_features_from_image(image_np)
                label, confidence, ranked, missing_count, unknown_count = predict_with_artifact(
                    model=model,
                    classes=classes,
                    feature_cols=feature_cols,
                    features=features,
                    top_k=top_k,
                )
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2%}")
            st.write("Top predictions:")
            for name, score in ranked:
                st.write(f"- {name}: {score:.2%}")
            st.caption(f"Missing features: {missing_count} | Unknown features: {unknown_count}")

st.markdown("---")
st.caption("Best fidelity is Upload Video because the deployed model was trained on temporal + HOG video features.")
