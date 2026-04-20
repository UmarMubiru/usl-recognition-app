from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return exp_vals / denom


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


@dataclass
class _BestState:
    val_loss: float
    weights: np.ndarray
    bias: np.ndarray
    epoch: int


@dataclass
class _BestStateMLP:
    val_loss: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    epoch: int


class DistilledSoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """Compact linear softmax classifier trained with blended KD targets."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1800,
        batch_size: int = 128,
        l2: float = 1e-4,
        tol: float = 1e-6,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "l2": self.l2,
            "tol": self.tol,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        out = np.zeros((len(y), n_classes), dtype=np.float64)
        out[np.arange(len(y)), y.astype(int)] = 1.0
        return out

    @staticmethod
    def _cross_entropy(q: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.mean(np.sum(q * np.log(p), axis=1)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        soft_targets: np.ndarray | None = None,
        alpha: float = 0.6,
        x_val: np.ndarray | None = None,
        y_val_blended: np.ndarray | None = None,
        patience: int = 120,
    ):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        y_int = y.astype(int)
        n_samples, n_features = x.shape
        n_classes = int(np.max(y_int)) + 1

        hard = self._one_hot(y_int, n_classes)
        if soft_targets is None:
            soft = hard.copy()
        else:
            soft = np.asarray(soft_targets, dtype=np.float64)
            if soft.shape != hard.shape:
                raise ValueError("soft_targets shape must match one-hot hard targets")
            row_sum = np.clip(soft.sum(axis=1, keepdims=True), 1e-12, None)
            soft = soft / row_sum

        a = float(np.clip(alpha, 0.0, 1.0))
        blended = a * soft + (1.0 - a) * hard

        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0.0, 0.01, size=(n_features, n_classes)).astype(np.float64)
        self.bias_ = np.zeros(n_classes, dtype=np.float64)

        self.classes_ = np.arange(n_classes, dtype=int)
        self.n_features_in_ = n_features
        self.loss_history_ = []
        self.alpha_ = a

        if x_val is not None and y_val_blended is not None:
            x_val = np.asarray(x_val, dtype=np.float64)
            y_val_blended = np.asarray(y_val_blended, dtype=np.float64)
            best = _BestState(
                val_loss=np.inf,
                weights=self.weights_.copy(),
                bias=self.bias_.copy(),
                epoch=0,
            )
            bad_epochs = 0
        else:
            best = None
            bad_epochs = 0

        batch = max(8, min(self.batch_size, n_samples))
        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            x_shuf = x[indices]
            y_shuf = blended[indices]

            for start in range(0, n_samples, batch):
                end = min(start + batch, n_samples)
                xb = x_shuf[start:end]
                yb = y_shuf[start:end]

                logits = xb @ self.weights_ + self.bias_
                probs = _softmax(logits)
                err = probs - yb

                grad_w = (xb.T @ err) / len(xb) + self.l2 * self.weights_
                grad_b = np.mean(err, axis=0)

                self.weights_ -= self.learning_rate * grad_w
                self.bias_ -= self.learning_rate * grad_b

            train_probs = _softmax(x @ self.weights_ + self.bias_)
            train_loss = self._cross_entropy(blended, train_probs) + 0.5 * self.l2 * float(np.sum(self.weights_ ** 2))
            self.loss_history_.append(train_loss)

            if self.verbose and (epoch % 100 == 0 or epoch == self.max_iter - 1):
                print(f"epoch={epoch:04d} train_loss={train_loss:.6f}")

            if len(self.loss_history_) > 1:
                if abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                    break

            if best is not None:
                val_probs = _softmax(x_val @ self.weights_ + self.bias_)
                val_loss = self._cross_entropy(y_val_blended, val_probs)
                if val_loss + self.tol < best.val_loss:
                    best = _BestState(
                        val_loss=val_loss,
                        weights=self.weights_.copy(),
                        bias=self.bias_.copy(),
                        epoch=epoch,
                    )
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best is not None:
            self.weights_ = best.weights
            self.bias_ = best.bias
            self.best_epoch_ = int(best.epoch)
            self.best_val_loss_ = float(best.val_loss)

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x @ self.weights_ + self.bias_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.decision_function(x)
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1).astype(int)


class DistilledMLPClassifier(BaseEstimator, ClassifierMixin):
    """Tiny one-hidden-layer MLP trained with blended KD targets."""

    def __init__(
        self,
        hidden_dim: int = 96,
        learning_rate: float = 0.01,
        max_iter: int = 2200,
        batch_size: int = 64,
        l2: float = 1e-4,
        tol: float = 1e-6,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    @staticmethod
    def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        out = np.zeros((len(y), n_classes), dtype=np.float64)
        out[np.arange(len(y)), y.astype(int)] = 1.0
        return out

    @staticmethod
    def _cross_entropy(q: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.mean(np.sum(q * np.log(p), axis=1)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        soft_targets: np.ndarray | None = None,
        alpha: float = 0.6,
        x_val: np.ndarray | None = None,
        y_val_blended: np.ndarray | None = None,
        patience: int = 180,
    ):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        y_int = y.astype(int)
        n_samples, n_features = x.shape
        n_classes = int(np.max(y_int)) + 1

        hard = self._one_hot(y_int, n_classes)
        if soft_targets is None:
            soft = hard.copy()
        else:
            soft = np.asarray(soft_targets, dtype=np.float64)
            if soft.shape != hard.shape:
                raise ValueError("soft_targets shape must match one-hot hard targets")
            row_sum = np.clip(soft.sum(axis=1, keepdims=True), 1e-12, None)
            soft = soft / row_sum

        a = float(np.clip(alpha, 0.0, 1.0))
        blended = a * soft + (1.0 - a) * hard

        rng = np.random.default_rng(self.random_state)
        hidden_dim = max(8, int(self.hidden_dim))

        # He-like init for ReLU hidden layer, small init for output layer.
        self.w1_ = (rng.standard_normal((n_features, hidden_dim)) * np.sqrt(2.0 / n_features)).astype(np.float64)
        self.b1_ = np.zeros(hidden_dim, dtype=np.float64)
        self.w2_ = (rng.standard_normal((hidden_dim, n_classes)) * 0.01).astype(np.float64)
        self.b2_ = np.zeros(n_classes, dtype=np.float64)

        self.classes_ = np.arange(n_classes, dtype=int)
        self.n_features_in_ = n_features
        self.loss_history_ = []
        self.alpha_ = a

        if x_val is not None and y_val_blended is not None:
            x_val = np.asarray(x_val, dtype=np.float64)
            y_val_blended = np.asarray(y_val_blended, dtype=np.float64)
            best = _BestStateMLP(
                val_loss=np.inf,
                w1=self.w1_.copy(),
                b1=self.b1_.copy(),
                w2=self.w2_.copy(),
                b2=self.b2_.copy(),
                epoch=0,
            )
            bad_epochs = 0
        else:
            best = None
            bad_epochs = 0

        batch = max(8, min(self.batch_size, n_samples))
        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            x_shuf = x[indices]
            y_shuf = blended[indices]

            for start in range(0, n_samples, batch):
                end = min(start + batch, n_samples)
                xb = x_shuf[start:end]
                yb = y_shuf[start:end]

                z1 = xb @ self.w1_ + self.b1_
                h1 = _relu(z1)
                logits = h1 @ self.w2_ + self.b2_
                probs = _softmax(logits)

                d_logits = (probs - yb) / len(xb)
                grad_w2 = h1.T @ d_logits + self.l2 * self.w2_
                grad_b2 = np.sum(d_logits, axis=0)

                d_h1 = d_logits @ self.w2_.T
                d_z1 = d_h1 * (z1 > 0.0)
                grad_w1 = xb.T @ d_z1 + self.l2 * self.w1_
                grad_b1 = np.sum(d_z1, axis=0)

                self.w2_ -= self.learning_rate * grad_w2
                self.b2_ -= self.learning_rate * grad_b2
                self.w1_ -= self.learning_rate * grad_w1
                self.b1_ -= self.learning_rate * grad_b1

            train_probs = self.predict_proba(x)
            train_loss = self._cross_entropy(blended, train_probs) + 0.5 * self.l2 * (
                float(np.sum(self.w1_ ** 2)) + float(np.sum(self.w2_ ** 2))
            )
            self.loss_history_.append(train_loss)

            if self.verbose and (epoch % 100 == 0 or epoch == self.max_iter - 1):
                print(f"epoch={epoch:04d} train_loss={train_loss:.6f}")

            if len(self.loss_history_) > 1:
                if abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                    break

            if best is not None:
                val_probs = self.predict_proba(x_val)
                val_loss = self._cross_entropy(y_val_blended, val_probs)
                if val_loss + self.tol < best.val_loss:
                    best = _BestStateMLP(
                        val_loss=val_loss,
                        w1=self.w1_.copy(),
                        b1=self.b1_.copy(),
                        w2=self.w2_.copy(),
                        b2=self.b2_.copy(),
                        epoch=epoch,
                    )
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best is not None:
            self.w1_ = best.w1
            self.b1_ = best.b1
            self.w2_ = best.w2
            self.b2_ = best.b2
            self.best_epoch_ = int(best.epoch)
            self.best_val_loss_ = float(best.val_loss)

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z1 = x @ self.w1_ + self.b1_
        h1 = _relu(z1)
        return h1 @ self.w2_ + self.b2_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.decision_function(x)
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1).astype(int)
