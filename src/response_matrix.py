from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


_ETRUE_HEADER_RE = re.compile(r"^Etrue_(?P<value>\d+(?:p\d+)?)_keV$")


@dataclass(frozen=True)
class ResponseMatrix:
    """Response operator and its measured/true energy grids."""

    h: np.ndarray
    e_meas_keV: np.ndarray
    e_true_keV: np.ndarray
    source_path: Path

    @property
    def n_measured(self) -> int:
        return int(self.h.shape[0])

    @property
    def n_true(self) -> int:
        return int(self.h.shape[1])


@dataclass(frozen=True)
class MatrixValidation:
    shape: tuple[int, int]
    e_meas_range_keV: tuple[float, float]
    e_true_range_keV: tuple[float, float]
    e_meas_step_first_keV: float | None
    e_true_step_first_keV: float | None
    finite: bool
    min_value: float
    max_value: float
    negative_count: int
    column_sum_min: float
    column_sum_max: float
    column_sum_mean: float
    column_sum_std: float
    dead_columns: int
    dead_rows: int


@dataclass(frozen=True)
class OperatorDiagnosticRow:
    e_true_keV: float
    global_peak_keV: float
    peak_on_diagonal: bool
    photopeak_to_colmax: float
    fraction_above_true_energy: float
    escape_511: bool
    escape_1022: bool


def parse_etrue_header(label: str) -> float:
    """Parse a header like ``Etrue_20p000000_keV`` into a keV value."""

    match = _ETRUE_HEADER_RE.match(str(label))
    if not match:
        raise ValueError(f"Invalid E_true header: {label!r}")
    return float(match.group("value").replace("p", "."))


def load_response_matrix(path: str | Path) -> ResponseMatrix:
    """Load the real rectangular response matrix CSV.

    Expected CSV convention:
    - first column: measured-energy bin centers in keV
    - remaining column headers: true-energy labels such as ``Etrue_20p000000_keV``
    - values: response matrix with shape ``measured x true``
    """

    source_path = Path(path)
    df = pd.read_csv(source_path, index_col=0)
    h = df.to_numpy(dtype=float)
    e_meas_keV = df.index.to_numpy(dtype=float)
    e_true_keV = np.array([parse_etrue_header(col) for col in df.columns], dtype=float)

    if h.shape != (len(e_meas_keV), len(e_true_keV)):
        raise ValueError(
            "Matrix/grid mismatch: "
            f"h={h.shape}, e_meas={len(e_meas_keV)}, e_true={len(e_true_keV)}"
        )

    return ResponseMatrix(
        h=h,
        e_meas_keV=e_meas_keV,
        e_true_keV=e_true_keV,
        source_path=source_path,
    )


def validate_response_matrix(matrix: ResponseMatrix) -> MatrixValidation:
    """Return core numerical validation facts for the response matrix."""

    h = matrix.h
    column_sums = h.sum(axis=0)
    row_sums = h.sum(axis=1)

    return MatrixValidation(
        shape=(int(h.shape[0]), int(h.shape[1])),
        e_meas_range_keV=(float(matrix.e_meas_keV[0]), float(matrix.e_meas_keV[-1])),
        e_true_range_keV=(float(matrix.e_true_keV[0]), float(matrix.e_true_keV[-1])),
        e_meas_step_first_keV=_first_step(matrix.e_meas_keV),
        e_true_step_first_keV=_first_step(matrix.e_true_keV),
        finite=bool(np.isfinite(h).all()),
        min_value=float(np.nanmin(h)),
        max_value=float(np.nanmax(h)),
        negative_count=int((h < 0).sum()),
        column_sum_min=float(column_sums.min()),
        column_sum_max=float(column_sums.max()),
        column_sum_mean=float(column_sums.mean()),
        column_sum_std=float(column_sums.std()),
        dead_columns=int((column_sums == 0).sum()),
        dead_rows=int((row_sums == 0).sum()),
    )


def diagnose_operator_type(
    matrix: ResponseMatrix,
    test_energies_keV: list[float] | None = None,
) -> list[OperatorDiagnosticRow]:
    """Probe columns to infer whether the operator behaves like H_d or H_tot.

    This is a diagnostic only. A detector-only DRF should show a strong feature near
    ``E_meas == E_true``. A full electron-to-detector operator is expected to peak
    at lower measured energies and lack a strong diagonal photopeak.
    """

    if test_energies_keV is None:
        candidates = [matrix.e_true_keV[0], 500.0, 1500.0, 3000.0, 6000.0]
        test_energies_keV = [e for e in candidates if e <= matrix.e_true_keV[-1]]

    rows: list[OperatorDiagnosticRow] = []
    for energy in test_energies_keV:
        j = nearest_index(matrix.e_true_keV, energy)
        col = matrix.h[:, j]
        col_sum = float(col.sum())
        if col_sum == 0.0:
            continue

        global_peak = float(matrix.e_meas_keV[int(col.argmax())])
        window = max(10.0, 0.05 * float(energy))
        lo = nearest_index(matrix.e_meas_keV, float(energy) - window)
        hi = nearest_index(matrix.e_meas_keV, float(energy) + window)
        photopeak = float(col[lo : hi + 1].max()) if hi >= lo else 0.0
        colmax = float(col.max())
        photopeak_to_colmax = photopeak / colmax if colmax > 0 else 0.0

        rows.append(
            OperatorDiagnosticRow(
                e_true_keV=float(matrix.e_true_keV[j]),
                global_peak_keV=global_peak,
                peak_on_diagonal=bool(photopeak_to_colmax >= 0.5),
                photopeak_to_colmax=float(photopeak_to_colmax),
                fraction_above_true_energy=float(
                    col[matrix.e_meas_keV > matrix.e_true_keV[j]].sum() / col_sum
                ),
                escape_511=_has_local_bump(matrix.e_meas_keV, col, matrix.e_true_keV[j] - 511.0),
                escape_1022=_has_local_bump(matrix.e_meas_keV, col, matrix.e_true_keV[j] - 1022.0),
            )
        )
    return rows


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _first_step(grid: np.ndarray) -> float | None:
    if len(grid) < 2:
        return None
    return float(grid[1] - grid[0])


def _has_local_bump(grid: np.ndarray, values: np.ndarray, target: float) -> bool:
    if target < grid[0] or target > grid[-1]:
        return False
    k = nearest_index(grid, target)
    lo = max(k - 10, 0)
    hi = min(k + 10, len(values))
    local_median = float(np.median(values[lo:hi]))
    return bool(values[k] > 1.2 * local_median) if local_median > 0 else bool(values[k] > 0)

