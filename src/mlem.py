from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLEMResult:
    x: np.ndarray
    chi2: np.ndarray
    relative_l2: np.ndarray | None


def smooth_121(values: np.ndarray) -> np.ndarray:
    """Three-point [0.25, 0.5, 0.25] smoothing with renormalized endpoints."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"smooth_121 expects a 1-D vector, got shape {values.shape}")
    if len(values) < 2:
        return values.copy()

    out = values.copy()
    out[1:-1] = 0.25 * values[:-2] + 0.5 * values[1:-1] + 0.25 * values[2:]
    out[0] = (0.5 * values[0] + 0.25 * values[1]) / 0.75
    out[-1] = (0.5 * values[-1] + 0.25 * values[-2]) / 0.75
    return out


def chi2_reduced(y_obs: np.ndarray, y_pred: np.ndarray, n_params: int = 0) -> float:
    """Reduced chi-square using Poisson variance approximated by predicted counts."""

    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_obs.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_obs={y_obs.shape}, y_pred={y_pred.shape}")

    variance = np.clip(y_pred, 1.0, None)
    residual = (y_obs - y_pred) ** 2 / variance
    dof = max(len(y_obs) - n_params, 1)
    return float(residual.sum() / dof)


def forward_project(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(h, dtype=float) @ np.asarray(x, dtype=float)


def run_mlem(
    y: np.ndarray,
    h: np.ndarray,
    n_iter: int = 300,
    smooth_every: int = 0,
    final_smooth: bool = False,
    x_true: np.ndarray | None = None,
    eps: float = 1e-12,
) -> MLEMResult:
    """Rectangular-safe MLEM / Richardson-Lucy solver for ``y = H x``."""

    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.ndim != 2:
        raise ValueError(f"H must be 2-D, got shape {h.shape}")
    if y.shape != (h.shape[0],):
        raise ValueError(f"y length must match H rows: y={y.shape}, H={h.shape}")
    if np.any(y < 0):
        raise ValueError("MLEM requires nonnegative measured counts")

    if x_true is not None:
        x_true = np.asarray(x_true, dtype=float)
        if x_true.shape != (h.shape[1],):
            raise ValueError(f"x_true length must match H columns: x_true={x_true.shape}, H={h.shape}")

    ht = h.T
    sensitivity = h.sum(axis=0)
    sensitivity = np.where(sensitivity == 0, eps, sensitivity)

    x0 = ht @ y
    peak = float(x0.max()) if len(x0) else 0.0
    floor = peak * 1e-8 if peak > 0 else eps
    x = np.clip(x0, floor, None)

    chi2_history: list[float] = []
    l2_history: list[float] = []

    for iteration in range(1, n_iter + 1):
        y_pred = h @ x
        y_pred = np.where(y_pred == 0, eps, y_pred)
        correction = ht @ (y / y_pred)
        x = x / sensitivity * correction
        x = np.clip(x, 0.0, None)

        if smooth_every and iteration % smooth_every == 0:
            x = smooth_121(x)

        chi2_history.append(chi2_reduced(y, h @ x))
        if x_true is not None:
            l2_history.append(relative_l2(normalize_sum(x), normalize_sum(x_true)))

    if final_smooth:
        x = smooth_121(x)

    return MLEMResult(
        x=x,
        chi2=np.array(chi2_history, dtype=float),
        relative_l2=np.array(l2_history, dtype=float) if x_true is not None else None,
    )


def normalize_sum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total == 0.0:
        return values.copy()
    return values / total


def relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = float(np.linalg.norm(truth))
    if denom == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(estimate - truth) / denom)

