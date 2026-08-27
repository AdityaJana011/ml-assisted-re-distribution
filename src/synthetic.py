from __future__ import annotations

import numpy as np


def make_synthetic_distribution(e_true_keV: np.ndarray, kind: str = "bulk_plus_tail") -> np.ndarray:
    """Small validation distributions for the real response operator."""

    e_true_keV = np.asarray(e_true_keV, dtype=float)
    if kind == "bulk_plus_tail":
        bulk = np.exp(-e_true_keV / 300.0)
        tail = 0.15 * np.exp(-((e_true_keV - 2000.0) ** 2) / (2 * 250.0**2))
        x = bulk + tail
    elif kind == "single_exp":
        x = np.exp(-e_true_keV / 400.0)
    elif kind == "delta_2mev":
        x = np.zeros_like(e_true_keV)
        x[int(np.argmin(np.abs(e_true_keV - 2000.0)))] = 1.0
    elif kind == "two_peaks":
        x = (
            np.exp(-((e_true_keV - 900.0) ** 2) / (2 * 120.0**2))
            + 0.65 * np.exp(-((e_true_keV - 2600.0) ** 2) / (2 * 220.0**2))
        )
    else:
        raise ValueError(f"Unknown distribution kind: {kind!r}")

    total = float(x.sum())
    if total == 0.0:
        raise ValueError(f"Distribution {kind!r} has zero area")
    return x / total


def simulate_counts(
    h: np.ndarray,
    x_true: np.ndarray,
    total_counts: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-project ``x_true`` and draw one Poisson measurement realization."""

    y_clean = np.asarray(h, dtype=float) @ np.asarray(x_true, dtype=float)
    total = float(y_clean.sum())
    if total == 0.0:
        raise ValueError("Forward projection produced zero total counts")
    y_clean = y_clean / total * float(total_counts)

    rng = np.random.default_rng(seed)
    y_noisy = rng.poisson(y_clean).astype(float)
    return y_clean, y_noisy

