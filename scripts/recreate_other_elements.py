"""Recreate Figure 2 style plots for the remaining calibration isotopes.

Isotopes processed: Ba133, Co60, Cs137, Na22.
For each isotope, we use the baseline MLEM algorithm (no narrowing, no smoothing)
because our prior investigation proved this is the optimal configuration for 
the ADITYA-U pre-computed real DRF.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path so `src` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mlem import run_mlem

# ── Paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DRF_PATH = REPO / "data" / "response_matrix.csv"
DATA_DIR = REPO / "data" / "LaBr3 spectrum 1kb"
OUT_DIR = REPO / "results"

# ── Known isotope gamma lines (keV) ──────────────────────────────────────
CERTIFICATES = {
    "Ba133": [
        (81.0, "81 keV"),
        (276.4, "276 keV"),
        (302.9, "303 keV"),
        (356.0, "356 keV"),
        (383.8, "384 keV"),
    ],
    "Co60": [
        (1173.2, "1173 keV"),
        (1332.5, "1332 keV"),
    ],
    "Cs137": [
        (661.7, "662 keV"),
    ],
    "Na22": [
        (511.0, "511 keV (annihil)"),
        (1274.5, "1275 keV"),
    ],
}

# ── Data loading ─────────────────────────────────────────────────────────
def load_drf():
    print("Loading DRF …")
    drf = pd.read_csv(DRF_PATH, index_col=0)
    h = drf.values.astype(float)                       # (6100, 599)
    e_meas = drf.index.to_numpy(dtype=float)           # keV, 1 keV step
    e_true = np.array(
        [float(c.replace("Etrue_", "").replace("_keV", "").replace("p", "."))
         for c in drf.columns],
        dtype=float,
    )  # 20–6000 keV, 10 keV step
    return h, e_meas, e_true

def load_spectrum(filepath: Path, n_rows_drf: int):
    data = np.loadtxt(filepath)
    e_spectrum = data[:, 0]  # keV
    counts = data[:, 1]      # counts per channel

    # Zero-pad to match DRF rows
    y = np.zeros(n_rows_drf)
    n_copy = min(len(counts), n_rows_drf)
    y[:n_copy] = counts[:n_copy]
    return y, e_spectrum, counts

# ── Peak finding ─────────────────────────────────────────────────────────
def find_nearest_peak(x_recon, e_true_kev, target_kev, window_kev=25.0):
    mask = (e_true_kev >= target_kev - window_kev) & (e_true_kev <= target_kev + window_kev)
    if not mask.any():
        return None, None
    idx_in_window = np.where(mask)[0]
    local_peak_idx = idx_in_window[np.argmax(x_recon[mask])]
    return e_true_kev[local_peak_idx], x_recon[local_peak_idx]

# ── Main processing loop ──────────────────────────────────────────────────
def process_isotope(isotope: str, h, e_meas, e_true):
    filepath = DATA_DIR / f"LaBr3_{isotope}_5mC.txt"
    if not filepath.exists():
        print(f"Skipping {isotope}: file not found at {filepath}")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing {isotope} ...")
    print(f"{'=' * 60}")

    y, e_spectrum, counts = load_spectrum(filepath, h.shape[0])

    # Run baseline MLEM (no narrowing, no smoothing)
    result = run_mlem(y, h, n_iter=300, smooth_every=0, final_smooth=False)
    x_recon = result.x
    
    print(f"  Converged after {result.n_iter_used} iterations, final χ² = {result.chi2[-1]:.4f}")

    # Peak Id Table
    print("\n  PEAK IDENTIFICATION TABLE")
    print(f"  {'Line':>18s}  {'Cert (keV)':>10s}  {'Found (keV)':>11s}  {'Offset':>8s}  {'Amplitude':>11s}")
    print("  " + "-" * 66)
    for cert_kev, name in CERTIFICATES[isotope]:
        found_e, amp = find_nearest_peak(x_recon, e_true, cert_kev)
        
        if found_e is not None:
            offset = found_e - cert_kev
            print(f"  {name:>18s}  {cert_kev:10.1f}  {found_e:11.1f}  {offset:+8.1f}  {amp:11.1f}")
        else:
            print(f"  {name:>18s}  {cert_kev:10.1f}  {'—':>11s}  {'—':>8s}  {'—':>11s}")

    # Create Dual-Axis Plot (Figure 2 Style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    e_meas_mev = e_meas / 1000.0
    e_true_mev = e_true / 1000.0
    e_spec_mev = e_spectrum / 1000.0

    # Determine plotting limits based on active spectrum
    max_e_signal = (e_spec_mev[counts > 0].max() if counts.sum() > 0 else 1.8) + 0.2
    max_e_plot = min(max_e_signal, 3.0) 

    # Left axis — measured spectrum (dashed)
    ax1.plot(e_spec_mev, counts,
             color="#2c7bb6", linestyle="--", linewidth=0.8, alpha=0.7,
             label="Recorded spectrum")
    ax1.set_xlabel("Energy (MeV)", fontsize=12)
    ax1.set_ylabel("Recorded spectrum\nN, counts per channel", fontsize=11, color="#2c7bb6")
    ax1.tick_params(axis="y", labelcolor="#2c7bb6")

    # Right axis — reconstructed spectrum (solid)
    ax2.plot(e_true_mev, x_recon,
             color="#d7191c", linestyle="-", linewidth=1.2,
             label="Deconvoluted spectrum (MLEM)")
    ax2.set_ylabel("Deconvoluted spectrum\nN (reconstructed)", fontsize=11, color="#d7191c")
    ax2.tick_params(axis="y", labelcolor="#d7191c")

    # Known lines as light vertical markers
    for cert_kev, name in CERTIFICATES[isotope]:
        ax1.axvline(cert_kev / 1000.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)

    ax1.set_xlim(0, max_e_plot)
    ax1.set_title(
        f"Reconstructed radiation spectrum of the $^{{{isotope}}}$ source\n"
        "(ADITYA-U LaBr$_3$ detector, standard MLEM deconvolution)",
        fontsize=13,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    fig.tight_layout()
    out_fig = OUT_DIR / f"fig2_{isotope}_recreation.png"
    fig.savefig(out_fig, dpi=200)
    print(f"\n  Saved plot → {out_fig}")
    plt.close(fig)


def main():
    h, e_meas, e_true = load_drf()
    for isotope in CERTIFICATES.keys():
        process_isotope(isotope, h, e_meas, e_true)
    print("\nAll isotopes processed successfully.")


if __name__ == "__main__":
    main()
