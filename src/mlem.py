from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLEMResult:
    x: np.ndarray
    chi2: np.ndarray
    relative_l2: np.ndarray | None
    n_iter_used: int


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


def narrow_drf(
    h: np.ndarray,
    e_meas: np.ndarray,
    scale: float = 0.7,
) -> np.ndarray:
    """Construct a narrowed DRF by compressing each column around its peak.

    For each column j (response to E_true_j), the measured-energy axis is
    linearly compressed toward the column's peak position by ``scale``
    (default 0.7 = 30% narrower FWHM), then interpolated back onto the
    original grid.  Column sums are preserved by renormalization.

    Parameters
    ----------
    h : (M, N) array
        Original detector response matrix.
    e_meas : (M,) array
        Measured-energy bin centers in keV.
    scale : float
        Compression factor (0.7 = 30% narrower).

    Returns
    -------
    h_narrow : (M, N) array
        Narrowed response matrix, same shape as *h*.
    """
    h = np.asarray(h, dtype=float)
    e_meas = np.asarray(e_meas, dtype=float)
    m, n = h.shape
    h_narrow = np.zeros_like(h)

    for j in range(n):
        col = h[:, j]
        col_sum = col.sum()
        if col_sum == 0:
            continue

        # Peak position for this column
        peak_idx = int(col.argmax())
        peak_e = e_meas[peak_idx]

        # Compressed energy axis: squeeze toward peak
        e_compressed = peak_e + scale * (e_meas - peak_e)

        # Interpolate original column onto compressed grid
        # (values outside original range are zero)
        new_col = np.interp(e_meas, e_compressed, col, left=0.0, right=0.0)

        # Renormalize to preserve column sum (probability conservation)
        new_sum = new_col.sum()
        if new_sum > 0:
            new_col *= col_sum / new_sum

        h_narrow[:, j] = new_col

    return h_narrow


def run_mlem(
    y: np.ndarray,
    h: np.ndarray,
    n_iter: int = 300,
    smooth_every: int = 0,
    final_smooth: bool = False,
    x_true: np.ndarray | None = None,
    h_backward: np.ndarray | None = None,
    chi2_target: float | None = None,
    chi2_window: int = 10,
    eps: float = 1e-12,
) -> MLEMResult:
    """MLEM / Richardson-Lucy solver for ``y = H x``.

    Implements the DeGaSum algorithmic modifications from Khilkevitch (2013):
      1. Non-negativity clipping at every iteration
      2. In-loop periodic [1/4, 1/2, 1/4] smoothing every ``smooth_every`` iters
      3. Optional narrowed DRF for the backward step (``h_backward``)
      4. Final post-deconvolution smoothing (``final_smooth``)
      5. Chi-squared early stopping (``chi2_target``)

    Parameters
    ----------
    y : measured spectrum (M,)
    h : forward DRF matrix (M, N)
    n_iter : max iterations
    smooth_every : apply smoothing every this many iterations (0 = off)
    final_smooth : apply one final smooth after iteration ends
    x_true : ground-truth for tracking relative-L2 error (optional)
    h_backward : narrowed DRF for backward step (M, N); if None, uses ``h``
    chi2_target : stop when reduced chi2 stays near this value; None = off
    chi2_window : number of consecutive iters chi2 must be near target
    eps : numerical floor
    """

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

    # Backward-step operator: narrowed DRF if provided, else same as forward
    if h_backward is not None:
        h_backward = np.asarray(h_backward, dtype=float)
        if h_backward.shape != h.shape:
            raise ValueError(
                f"h_backward shape must match h: h_backward={h_backward.shape}, h={h.shape}"
            )
        ht_back = h_backward.T
    else:
        ht_back = h.T

    sensitivity = h.sum(axis=0)
    sensitivity = np.where(sensitivity == 0, eps, sensitivity)

    # --- Initialization: x⁰ = H^T @ y / ‖H(H^T @ y)‖ ---
    # Paper: x⁰ = y / ‖Hy‖ (adapted for rectangular H: backproject then
    # normalize by the forward-re-projection norm so x starts at a
    # physically meaningful scale)
    x0 = ht_back @ y
    hy_norm = float(np.linalg.norm(h @ x0))
    if hy_norm > 0:
        x = x0 / hy_norm
    else:
        x = np.clip(x0, eps, None)
    x = np.clip(x, 0.0, None)

    chi2_history: list[float] = []
    l2_history: list[float] = []
    n_iter_used = n_iter

    for iteration in range(1, n_iter + 1):
        # Forward project with the physical (full-width) DRF
        y_pred = h @ x
        y_pred = np.where(y_pred == 0, eps, y_pred)

        # Backward correction with (optionally narrowed) DRF
        correction = ht_back @ (y / y_pred)
        x = x / sensitivity * correction

        # DeGaSum trick 1: non-negativity
        x = np.clip(x, 0.0, None)

        # DeGaSum trick 2: in-loop periodic smoothing
        if smooth_every and iteration % smooth_every == 0:
            x = smooth_121(x)

        chi2_val = chi2_reduced(y, h @ x)
        chi2_history.append(chi2_val)
        if x_true is not None:
            l2_history.append(relative_l2(normalize_sum(x), normalize_sum(x_true)))

        # DeGaSum trick 5: chi-squared early stopping
        if chi2_target is not None and len(chi2_history) >= chi2_window:
            recent = chi2_history[-chi2_window:]
            if all(abs(c - chi2_target) < 0.1 * chi2_target for c in recent):
                n_iter_used = iteration
                break

    # DeGaSum trick 4: final post-deconvolution smoothing
    if final_smooth:
        x = smooth_121(x)

    return MLEMResult(
        x=x,
        chi2=np.array(chi2_history, dtype=float),
        relative_l2=np.array(l2_history, dtype=float) if x_true is not None else None,
        n_iter_used=n_iter_used,
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

