from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT = APP_DIR / "model.pkl"
FALLBACK_ARTIFACT = APP_DIR.parent / "models_dataset1" / "csv_models" / "artifacts" / "best_model.joblib"
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(LOCAL_ARTIFACT if LOCAL_ARTIFACT.exists() else FALLBACK_ARTIFACT)))


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
    coords = [[p.x, p.y, p.z] for p in landmarks.landmark[:max_points]]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _safe_stats(values: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        float(np.percentile(values, 90)),
    )


def _make_338_proxy_features(image_rgb: np.ndarray) -> dict[str, float]:
    h, w = image_rgb.shape[:2]
    gray = image_rgb.mean(axis=2).astype(np.float32)

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as holistic:
        result = holistic.process(image_rgb)

    pose = _landmarks_to_array(result.pose_landmarks, 33)
    left = _landmarks_to_array(result.left_hand_landmarks, 21)
    right = _landmarks_to_array(result.right_hand_landmarks, 21)

    gx = np.abs(np.diff(gray, axis=1)).mean() if w > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if h > 1 else 0.0
    motion_proxy = float((gx + gy) / 2.0)

    gmean, gstd, gp10, gp25, gp75, gp90 = _safe_stats(gray.flatten())

    pose_spread = float(np.std(pose[:, :2])) if len(pose) else 0.0
    left_spread = float(np.std(left[:, :2])) if len(left) else 0.0
    right_spread = float(np.std(right[:, :2])) if len(right) else 0.0
    hand_spread = float((left_spread + right_spread) / max((1 if len(left) else 0) + (1 if len(right) else 0), 1))

    base = {
        "fps": 1.0,
        "frame_count": 1.0,
        "duration_sec": 1.0,
        "width": float(w),
        "height": float(h),
        "aspect_ratio": float(w / h) if h > 0 else 0.0,
        "gray_mean_mean": gmean,
        "gray_mean_std": gstd,
        "gray_mean_p10": gp10,
        "gray_mean_p25": gp25,
        "gray_mean_p75": gp75,
        "gray_mean_p90": gp90,
        "gray_std_mean": gstd,
        "gray_std_std": 0.0,
        "gray_std_p10": gstd,
        "gray_std_p25": gstd,
        "gray_std_p75": gstd,
        "gray_std_p90": gstd,
        "motion_mean": motion_proxy,
        "motion_std": 0.0,
        "motion_p10": motion_proxy,
        "motion_p25": motion_proxy,
        "motion_p75": motion_proxy,
        "motion_p90": motion_proxy,
        "gray_start": gmean,
        "gray_middle": gmean,
        "gray_end": gmean,
        "gray_trend": 0.0,
        "motion_start": motion_proxy,
        "motion_middle": motion_proxy,
        "motion_end": motion_proxy,
        "motion_trend": 0.0,
        "gray_slope": 0.0,
        "motion_slope": 0.0,
        "motion_to_duration": motion_proxy,
        "motion_to_frames": motion_proxy,
        "intensity_to_motion": float(gstd / max(motion_proxy, 1e-6)),
        "pose_presence_ratio": 1.0 if len(pose) else 0.0,
        "left_hand_presence_ratio": 1.0 if len(left) else 0.0,
        "right_hand_presence_ratio": 1.0 if len(right) else 0.0,
        "pose_spread_mean": pose_spread,
        "pose_spread_std": 0.0,
        "hand_spread_mean": hand_spread,
        "hand_spread_std": 0.0,
        "pose_motion_mean": 0.0,
        "pose_motion_std": 0.0,
        "hand_motion_mean": 0.0,
        "hand_motion_std": 0.0,
        "sampled_frames": 1.0,
        "sampled_motion_steps": 0.0,
    }

    # Use global intensity histogram bins to fill hog slots deterministically.
    hist, _ = np.histogram(gray, bins=144, range=(0, 255), density=True)
    hist = hist.astype(np.float32)
    base.update({f"hog_mean_{i}": float(hist[i]) for i in range(144)})
    base.update({f"hog_std_{i}": 0.0 for i in range(144)})
    return base


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


def _read_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


st.set_page_config(page_title="Uganda Sign Language Recognition", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        /* Main background and fonts */
        .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        
        /* Header styling */
        .header-title {
            font-size: 3em;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header-subtitle {
            font-size: 1.2em;
            color: #e0e0e0;
            text-align: center;
            margin-bottom: 2em;
        }
        
        /* Card styling */
        .card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 5px solid #667eea;
            border-radius: 10px;
            padding: 1.5em;
            margin: 1em 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75em 2em;
            font-weight: 600;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
        }
        
        /* Success message styling */
        .success-box {
            background: linear-gradient(135deg, #66d9a0 0%, #52c47c 100%);
            color: white;
            padding: 1.5em;
            border-radius: 10px;
            margin: 1em 0;
            box-shadow: 0 4px 6px rgba(102, 217, 160, 0.3);
            font-weight: 500;
        }
        
        /* Prediction result styling */
        .prediction-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2em;
            border-radius: 12px;
            text-align: center;
            margin: 1.5em 0;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }
        
        .prediction-label {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 0.5em;
        }
        
        .confidence-score {
            font-size: 1.8em;
            font-weight: 600;
            color: #ffeb3b;
        }
        
        /* Top predictions styling */
        .top-predictions {
            background: #f5f7fa;
            border-radius: 10px;
            padding: 1.5em;
            margin: 1em 0;
        }
        
        .prediction-item {
            background: white;
            padding: 1em;
            margin: 0.5em 0;
            border-left: 4px solid #667eea;
            border-radius: 6px;
        }
        
        /* Radio button styling */
        .css-1aumxpb { color: #667eea; }
        
        /* Divider */
        hr { border: 1px solid rgba(102, 126, 234, 0.3); }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="header-title">🖐️ Uganda Sign Language Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">AI-Powered Disease Sign Classification System</div>', unsafe_allow_html=True)

try:
    artifact = load_artifact(ARTIFACT_PATH)
except Exception as exc:
    st.error(f"Could not load artifact: {exc}")
    st.error("❌ Model loading failed. Ensure model.pkl exists in the app directory.")
    st.stop()

model = artifact["model"]
model_name = str(artifact["model_name"])
feature_cols = list(artifact["feature_cols"])
classes = list(artifact["classes"])

st.markdown("---")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 Model Type", model_name.upper().replace("_", "-"))
    with col2:
        st.metric("📊 Features", len(feature_cols))
    with col3:
        st.metric("🏥 Diseases", len(classes))

st.markdown("---")

st.markdown("### 📋 Recognition Settings", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("📊 Show top predictions", min_value=1, max_value=10, value=3, help="Number of top predictions to display")
with col2:
    st.info("💡 Tip: Upload a video for best accuracy (trained on temporal features)")

st.markdown("---")
st.markdown("### 🎯 Choose Input Method", unsafe_allow_html=True)

input_mode = st.radio(
    "Select how you want to provide the input:",
    ["🖼️ Upload Image", "📸 Webcam Snapshot"],
    horizontal=True,
    help="Image or webcam snapshot"
)

if "Upload Image" in input_mode:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        try:
            image_np = _read_uploaded_image(uploaded_image)
        except Exception as exc:
            st.error(f"Could not decode uploaded image: {exc}")
            st.stop()
        st.image(image_np, caption="Uploaded image", use_container_width=True)
        if st.button("Predict", key="predict_image"):
            with st.spinner("Extracting 338-compatible features from image and predicting..."):
                features = _make_338_proxy_features(image_np)
                label, confidence, ranked, missing_count, unknown_count = predict_with_artifact(
                    model=model,
                    classes=classes,
                    feature_cols=feature_cols,
                    features=features,
                    top_k=top_k,
                )
            st.markdown('<div class="prediction-result"><div class="prediction-label">✅ {}</div><div class="confidence-score">Confidence: {:.1%}</div></div>'.format(label, confidence), unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="top-predictions"><h4>🏆 Top Predictions</h4>', unsafe_allow_html=True)
                for idx, (name, score) in enumerate(ranked, 1):
                    st.markdown(f'<div class="prediction-item"><strong>#{idx}</strong> {name} <span style="float:right;color:#667eea;font-weight:bold">{score:.1%}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            col1.metric("ℹ️ Missing Features", missing_count)
            col2.metric("⚠️ Unknown Features", unknown_count)

if "Webcam Snapshot" in input_mode:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        try:
            image_np = _read_uploaded_image(camera_file)
        except Exception as exc:
            st.error(f"Could not decode webcam snapshot: {exc}")
            st.stop()
        st.image(image_np, caption="Captured image", use_container_width=True)
        if st.button("Predict", key="predict_camera"):
            with st.spinner("Extracting 338-compatible features from snapshot and predicting..."):
                features = _make_338_proxy_features(image_np)
                label, confidence, ranked, missing_count, unknown_count = predict_with_artifact(
                    model=model,
                    classes=classes,
                    feature_cols=feature_cols,
                    features=features,
                    top_k=top_k,
                )
            st.markdown('<div class="prediction-result"><div class="prediction-label">✅ {}</div><div class="confidence-score">Confidence: {:.1%}</div></div>'.format(label, confidence), unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="top-predictions"><h4>🏆 Top Predictions</h4>', unsafe_allow_html=True)
                for idx, (name, score) in enumerate(ranked, 1):
                    st.markdown(f'<div class="prediction-item"><strong>#{idx}</strong> {name} <span style="float:right;color:#667eea;font-weight:bold">{score:.1%}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            col1.metric("ℹ️ Missing Features", missing_count)
            col2.metric("⚠️ Unknown Features", unknown_count)

st.markdown("---")

with st.container():
    st.markdown("""
        <div style="text-align: center; margin-top: 2em; padding: 1.5em; background: #f5f7fa; border-radius: 10px;">
        <h4>📚 System Information</h4>
        <p><strong>Model:</strong> Support Vector Machine with RBF Kernel</p>
        <p><strong>Training Data:</strong> 338 engineered video features (temporal, HOG, MediaPipe)</p>
        <p><strong>Test Accuracy:</strong> 87.5%</p>
        <p><strong>💡 Note:</strong> Deployed cloud mode currently uses image/webcam proxy features for compatibility.</p>
        <p style="font-size: 0.9em; color: #888;">
            <em>Device Signs: ASCARIASIS, CHOLERA, COVID, EBOLA, MALARIA, HIV, HEPATITIS, & 18 more...</em>
        </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid #667eea;'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<p style='text-align:center'>🐍 Built with Streamlit</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<p style='text-align:center'>🤖 Powered by scikit-learn</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p style='text-align:center'>📍 Makerere University</p>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color: #999; font-size: 0.85em; margin-top: 1em;'>Made for Uganda Sign Language Research | 2026</p>", unsafe_allow_html=True)
