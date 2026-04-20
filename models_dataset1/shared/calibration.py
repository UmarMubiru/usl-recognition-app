from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return exp_vals / denom


def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true.astype(int)], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


class TemperatureScaledClassifier:
    """Wrap a classifier and apply temperature scaling to its probability output."""

    def __init__(self, base_model, temperature: float = 1.0) -> None:
        self.base_model = base_model
        self.temperature = float(max(1e-4, temperature))

    @staticmethod
    def _scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        scaled = probs ** (1.0 / max(1e-4, float(temperature)))
        denom = np.clip(np.sum(scaled, axis=1, keepdims=True), 1e-12, None)
        return scaled / denom

    @staticmethod
    def fit_from_validation(
        base_model,
        x_val: np.ndarray,
        y_val: np.ndarray,
        candidates: list[float] | None = None,
    ) -> tuple["TemperatureScaledClassifier", dict[str, float]]:
        if not hasattr(base_model, "predict_proba"):
            return TemperatureScaledClassifier(base_model=base_model, temperature=1.0), {
                "temperature": 1.0,
                "pre_nll": 0.0,
                "post_nll": 0.0,
            }

        y_val = np.asarray(y_val, dtype=np.int64)
        base_probs = np.asarray(base_model.predict_proba(x_val), dtype=np.float64)

        pre_nll = _nll(y_val, base_probs)
        best_t = 1.0
        best_nll = pre_nll

        if candidates is None:
            candidates = [0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2, 3.0]

        for t in candidates:
            scaled = TemperatureScaledClassifier._scale_probs(base_probs, t)
            cur_nll = _nll(y_val, scaled)
            if cur_nll < best_nll:
                best_nll = cur_nll
                best_t = float(t)

        wrapped = TemperatureScaledClassifier(base_model=base_model, temperature=best_t)
        return wrapped, {
            "temperature": float(best_t),
            "pre_nll": float(pre_nll),
            "post_nll": float(best_nll),
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(x)
            return self._scale_probs(probs, self.temperature)

        if hasattr(self.base_model, "decision_function"):
            decision = np.asarray(self.base_model.decision_function(x), dtype=np.float64)
            if decision.ndim == 1:
                decision = np.column_stack([-decision, decision])
            logits = decision / self.temperature
            return _softmax(logits)

        pred = np.asarray(self.base_model.predict(x), dtype=np.int64)
        n_classes = int(np.max(pred)) + 1
        probs = np.zeros((len(pred), n_classes), dtype=np.float64)
        probs[np.arange(len(pred)), pred] = 1.0
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1).astype(int)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        probs = np.clip(self.predict_proba(x), 1e-12, 1.0)
        return np.log(probs)
