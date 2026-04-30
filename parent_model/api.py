from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field


DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "models_dataset1"
    / "csv_models"
    / "artifacts"
    / "best_model.joblib"
)
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT)))
FALLBACK_ARTIFACT_PATH_RAW = os.getenv("FALLBACK_MODEL_ARTIFACT_PATH", "").strip()
FALLBACK_ARTIFACT_PATH = Path(FALLBACK_ARTIFACT_PATH_RAW) if FALLBACK_ARTIFACT_PATH_RAW else None
FALLBACK_CONFIDENCE_THRESHOLD = float(os.getenv("FALLBACK_CONFIDENCE_THRESHOLD", "0.75"))
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
API_KEY = os.getenv("API_KEY")


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("dataset1_api")


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature-name to numeric-value mapping")
    top_k: int = Field(default=3, ge=1, le=10)


class PredictionItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    model_name: str
    primary_model_name: str
    fallback_model_name: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    primary_confidence: float
    predicted_label: str
    confidence: float
    top_k: list[PredictionItem]
    missing_feature_count: int
    unknown_feature_count: int


app = FastAPI(title="Dataset1 Disease Model API", version="1.0.0")


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    payload = joblib.load(path)
    required = {"model", "model_name", "feature_cols", "classes"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Invalid model artifact. Missing keys: {sorted(missing)}")
    return payload


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom <= 0:
        return np.full_like(exp_vals, 1.0 / len(exp_vals), dtype=float)
    return exp_vals / denom


def _normalize_probs(probs: np.ndarray, length: int) -> np.ndarray:
    probs = np.array(probs, dtype=float).reshape(-1)
    if probs.shape[0] != length:
        return np.full(length, 1.0 / float(length), dtype=float)
    probs = np.clip(probs, 0.0, np.inf)
    total = float(np.sum(probs))
    if total <= 0:
        return np.full(length, 1.0 / float(length), dtype=float)
    return probs / total


def _compute_probs(model: Any, x_vec: np.ndarray, class_count: int, pred_idx: int) -> np.ndarray:
    probs: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x_vec)[0].astype(float)
        except Exception:
            probs = None

    if probs is None:
        if hasattr(model, "decision_function"):
            decision = model.decision_function(x_vec)
            decision = np.array(decision).reshape(-1)
            if decision.shape[0] == 1 and class_count == 2:
                p1 = 1.0 / (1.0 + np.exp(-decision[0]))
                probs = np.array([1.0 - p1, p1], dtype=float)
            else:
                probs = _softmax(decision)
        else:
            probs = np.zeros(class_count, dtype=float)
            probs[pred_idx] = 1.0

    return _normalize_probs(probs, class_count)


def _infer(model: Any, x_vec: np.ndarray, class_count: int) -> tuple[int, np.ndarray]:
    pred_idx = int(model.predict(x_vec)[0])
    probs = _compute_probs(model, x_vec, class_count, pred_idx)
    return pred_idx, probs


def _resolve_fallback_artifact(primary_path: Path) -> dict[str, Any] | None:
    if not ENABLE_FALLBACK:
        return None

    candidate_path = FALLBACK_ARTIFACT_PATH
    if candidate_path is None and primary_path != DEFAULT_ARTIFACT and DEFAULT_ARTIFACT.exists():
        candidate_path = DEFAULT_ARTIFACT

    if candidate_path is None:
        return None

    if candidate_path.resolve() == primary_path.resolve():
        return None

    if not candidate_path.exists():
        LOGGER.warning("Configured fallback model artifact does not exist: %s", candidate_path)
        return None

    payload = _load_artifact(candidate_path)
    return {
        "artifact_path": candidate_path,
        "payload": payload,
    }


ARTIFACT: dict[str, Any] = _load_artifact(ARTIFACT_PATH)
MODEL = ARTIFACT["model"]
FEATURE_COLS: list[str] = list(ARTIFACT["feature_cols"])
CLASS_NAMES: list[str] = list(ARTIFACT["classes"])
MODEL_NAME: str = str(ARTIFACT["model_name"])
MODEL_CREATED_AT: str = str(ARTIFACT.get("created_at_utc", "unknown"))
MODEL_SELECTION_METRIC: str = str(ARTIFACT.get("selection_metric", "unknown"))
MODEL_SELECTION_VALUE: float | str = ARTIFACT.get("selection_value", "unknown")

FALLBACK_ARTIFACT: dict[str, Any] | None = _resolve_fallback_artifact(ARTIFACT_PATH)
FALLBACK_MODEL = FALLBACK_ARTIFACT["payload"]["model"] if FALLBACK_ARTIFACT else None
FALLBACK_FEATURE_COLS: list[str] | None = (
    list(FALLBACK_ARTIFACT["payload"]["feature_cols"]) if FALLBACK_ARTIFACT else None
)
FALLBACK_CLASS_NAMES: list[str] | None = (
    list(FALLBACK_ARTIFACT["payload"]["classes"]) if FALLBACK_ARTIFACT else None
)
FALLBACK_MODEL_NAME: str | None = (
    str(FALLBACK_ARTIFACT["payload"]["model_name"]) if FALLBACK_ARTIFACT else None
)


def _enforce_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    response.headers["x-request-id"] = request_id
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "fallback_model_name": FALLBACK_MODEL_NAME,
        "fallback_enabled": ENABLE_FALLBACK,
        "fallback_active": FALLBACK_MODEL is not None,
        "fallback_confidence_threshold": FALLBACK_CONFIDENCE_THRESHOLD,
        "model_created_at": MODEL_CREATED_AT,
        "selection_metric": MODEL_SELECTION_METRIC,
        "selection_value": MODEL_SELECTION_VALUE,
        "feature_count": len(FEATURE_COLS),
        "artifact_path": str(ARTIFACT_PATH),
        "fallback_artifact_path": str(FALLBACK_ARTIFACT["artifact_path"]) if FALLBACK_ARTIFACT else None,
    }


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {
        "status": "ready" if MODEL is not None else "not_ready",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "fallback_model_loaded": FALLBACK_MODEL is not None,
        "fallback_model_name": FALLBACK_MODEL_NAME,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, x_api_key: str | None = Header(default=None, alias="x-api-key")) -> PredictResponse:
    _enforce_api_key(x_api_key)

    if not payload.features:
        raise HTTPException(status_code=400, detail="features cannot be empty")

    incoming = payload.features
    missing_count = sum(1 for feat in FEATURE_COLS if feat not in incoming)
    unknown_count = sum(1 for feat in incoming.keys() if feat not in FEATURE_COLS)

    x_vec = np.array([float(incoming.get(feat, 0.0)) for feat in FEATURE_COLS], dtype=float).reshape(1, -1)

    try:
        pred_idx, probs = _infer(MODEL, x_vec, len(CLASS_NAMES))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    served_pred_idx = pred_idx
    served_probs = probs
    served_classes = CLASS_NAMES
    served_model_name = MODEL_NAME
    fallback_used = False
    fallback_reason: str | None = None
    primary_confidence = float(probs[pred_idx])

    if FALLBACK_MODEL is not None and primary_confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        if FALLBACK_FEATURE_COLS is None or FALLBACK_CLASS_NAMES is None or FALLBACK_MODEL_NAME is None:
            LOGGER.warning("Fallback model metadata is incomplete. Serving primary prediction.")
        else:
            x_vec_fallback = np.array(
                [float(incoming.get(feat, 0.0)) for feat in FALLBACK_FEATURE_COLS],
                dtype=float,
            ).reshape(1, -1)
            try:
                fallback_pred_idx, fallback_probs = _infer(FALLBACK_MODEL, x_vec_fallback, len(FALLBACK_CLASS_NAMES))
                served_pred_idx = fallback_pred_idx
                served_probs = fallback_probs
                served_classes = FALLBACK_CLASS_NAMES
                served_model_name = FALLBACK_MODEL_NAME
                fallback_used = True
                fallback_reason = (
                    f"primary_confidence {primary_confidence:.4f} below threshold {FALLBACK_CONFIDENCE_THRESHOLD:.4f}"
                )
            except Exception as exc:
                LOGGER.warning("Fallback inference failed. Serving primary prediction. error=%s", exc)

    top_k = min(payload.top_k, len(served_classes))
    top_idx = np.argsort(served_probs)[::-1][:top_k]
    top_items = [
        PredictionItem(label=served_classes[int(i)], score=float(served_probs[int(i)]))
        for i in top_idx
    ]

    return PredictResponse(
        model_name=served_model_name,
        primary_model_name=MODEL_NAME,
        fallback_model_name=FALLBACK_MODEL_NAME,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        primary_confidence=primary_confidence,
        predicted_label=served_classes[served_pred_idx],
        confidence=float(served_probs[served_pred_idx]),
        top_k=top_items,
        missing_feature_count=missing_count,
        unknown_feature_count=unknown_count,
    )
