"""Recreate Figure 2 from Khilkevitch et al. (2013).

Dual-axis plot of the Eu152 calibration source:
  - Dashed line (left y-axis):  measured / recorded spectrum  (counts per channel)
  - Solid  line (right y-axis): MLEM-reconstructed spectrum

Uses the ADITYA-U LaBr3(Ce) real detector response matrix (6100 x 599)
and the 1 keV aligned Eu152 measured spectrum.

Note on the narrowed-DRF trick (DeGaSum trick #3):
  The paper's 30% narrower DRF is designed for *parametric* DRFs that can be
  regenerated at a different FWHM.  For our pre-computed real DRF, the column
  peaks at near-zero E_meas (a known artifact — see open_physics_questions.md)
  make compression-based narrowing destructive.  We therefore use the standard
  DRF for both forward and backward steps.  Once a parametric DRF model or a
  narrowed DRF from MCNP is available, `narrow_drf()` in src/mlem.py can be
  used.
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
EU152_PATH = REPO / "data" / "LaBr3 spectrum 1kb" / "LaBr3_Eu152_5mC.txt"
OUT_DIR = REPO / "results"

# ── Known Eu152 gamma lines (keV) ───────────────────────────────────────
EU152_CERT = [
    (121.8,  "122 keV"),
    (244.7,  "245 keV"),
    (344.3,  "344 keV"),
    (411.1,  "411 keV"),
    (443.9,  "444 keV"),
    (563.2,  "563 keV"),
    (586.3,  "586 keV"),
    (688.7,  "689 keV"),
    (778.9,  "779 keV"),
    (867.4,  "867 keV"),
    (964.1,  "964 keV"),
    (1085.8, "1086 keV"),
    (1112.1, "1112 keV"),
    (1212.9, "1213 keV"),
    (1299.1, "1299 keV"),
    (1408.0, "1408 keV"),
]

# ── Data loading ─────────────────────────────────────────────────────────
def load_data():
    print("Loading DRF …")
    drf = pd.read_csv(DRF_PATH, index_col=0)
    h = drf.values.astype(float)                       # (6100, 599)
    e_meas = drf.index.to_numpy(dtype=float)           # keV, 1 keV step
    e_true = np.array(
        [float(c.replace("Etrue_", "").replace("_keV", "").replace("p", "."))
         for c in drf.columns],
        dtype=float,
    )  # 20–6000 keV, 10 keV step
    print(f"  DRF shape: {h.shape}, E_meas: {e_meas[0]:.1f}–{e_meas[-1]:.1f} keV, "
          f"E_true: {e_true[0]:.0f}–{e_true[-1]:.0f} keV")

    print("Loading Eu152 spectrum …")
    data = np.loadtxt(EU152_PATH)
    e_spectrum = data[:, 0]  # keV
    counts = data[:, 1]      # counts per channel
    print(f"  Spectrum: {len(counts)} channels, "
          f"E: {e_spectrum[0]:.1f}–{e_spectrum[-1]:.1f} keV, "
          f"total counts: {counts.sum():.0f}")

    # Zero-pad to match DRF rows
    y = np.zeros(h.shape[0])
    y[:len(counts)] = counts
    print(f"  Padded y to {len(y)} channels")

    return h, e_meas, e_true, y, e_spectrum, counts


# ── Peak finding ─────────────────────────────────────────────────────────
def find_nearest_peak(x_recon, e_true_kev, target_kev, window_kev=25.0):
    """Find the peak in x_recon nearest to target_kev within ±window_kev."""
    mask = (e_true_kev >= target_kev - window_kev) & (e_true_kev <= target_kev + window_kev)
    if not mask.any():
        return None, None
    idx_in_window = np.where(mask)[0]
    local_peak_idx = idx_in_window[np.argmax(x_recon[mask])]
    return e_true_kev[local_peak_idx], x_recon[local_peak_idx]


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    h, e_meas, e_true, y, e_spectrum, counts = load_data()

    # ── Run MLEM: unsmoothed (user-confirmed best with current DRF) ──
    print("\n[A] Running MLEM — no smoothing (300 iters) …")
    result_nosmooth = run_mlem(y, h, n_iter=300, smooth_every=0, final_smooth=False)
    print(f"    Done: {result_nosmooth.n_iter_used} iters, "
          f"final χ² = {result_nosmooth.chi2[-1]:.4f}")

    # ── Run MLEM: with smoothing for comparison ──
    print("[B] Running MLEM — smooth_every=10, final_smooth (300 iters) …")
    result_smooth = run_mlem(y, h, n_iter=300, smooth_every=10, final_smooth=True)
    print(f"    Done: {result_smooth.n_iter_used} iters, "
          f"final χ² = {result_smooth.chi2[-1]:.4f}")

    # Primary reconstruction for Figure 2 — use whichever is better
    # (user reported unsmoothed is better for this DRF)
    x_primary = result_nosmooth.x
    x_alt = result_smooth.x

    # ── Peak identification table ──
    print("\n" + "=" * 80)
    print("PEAK IDENTIFICATION TABLE")
    print("=" * 80)
    print(f"{'Line':>8s}  {'Cert (keV)':>10s}  "
          f"{'NoSmooth':>12s}  {'Smooth':>12s}  "
          f"{'Amp (NoSm)':>11s}  {'Amp (Sm)':>11s}")
    print("-" * 80)
    for cert_kev, name in EU152_CERT:
        e_ns, a_ns = find_nearest_peak(x_primary, e_true, cert_kev)
        e_sm, a_sm = find_nearest_peak(x_alt, e_true, cert_kev)
        ns_str = f"{e_ns:.0f}" if e_ns is not None else "—"
        sm_str = f"{e_sm:.0f}" if e_sm is not None else "—"
        ans_str = f"{a_ns:.1f}" if a_ns is not None else "—"
        asm_str = f"{a_sm:.1f}" if a_sm is not None else "—"
        print(f"{name:>8s}  {cert_kev:10.1f}  {ns_str:>12s}  {sm_str:>12s}  "
              f"{ans_str:>11s}  {asm_str:>11s}")

    # ── Figure 2: dual-axis plot (paper style) ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    e_meas_mev = e_meas / 1000.0
    e_true_mev = e_true / 1000.0
    e_spec_mev = e_spectrum / 1000.0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Left axis — measured spectrum (dashed)
    ax1.plot(e_spec_mev, counts,
             color="#2c7bb6", linestyle="--", linewidth=0.8, alpha=0.7,
             label="Recorded spectrum")
    ax1.set_xlabel("Energy (MeV)", fontsize=12)
    ax1.set_ylabel("Recorded spectrum\nN, counts per channel", fontsize=11, color="#2c7bb6")
    ax1.tick_params(axis="y", labelcolor="#2c7bb6")

    # Right axis — reconstructed spectrum (solid)
    ax2.plot(e_true_mev, x_primary,
             color="#d7191c", linestyle="-", linewidth=1.2,
             label="Deconvoluted spectrum (MLEM)")
    ax2.set_ylabel("Deconvoluted spectrum\nN (reconstructed)", fontsize=11, color="#d7191c")
    ax2.tick_params(axis="y", labelcolor="#d7191c")

    # Known Eu152 lines as light vertical markers
    for cert_kev, name in EU152_CERT:
        ax1.axvline(cert_kev / 1000.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    ax1.set_xlim(0, 1.8)
    ax1.set_title(
        "Reconstructed radiation spectrum of the $^{152}$Eu source\n"
        "(ADITYA-U LaBr$_3$ detector, MLEM deconvolution)",
        fontsize=13,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    fig.tight_layout()
    out_fig2 = OUT_DIR / "fig2_eu152_recreation.png"
    fig.savefig(out_fig2, dpi=200)
    print(f"\nSaved Figure 2 → {out_fig2}")
    plt.close(fig)

    # ── Comparison: smoothed vs unsmoothed ──
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(e_true_mev, x_primary,
            alpha=0.7, linewidth=0.9, label=f"No smoothing (χ²={result_nosmooth.chi2[-1]:.1f})")
    ax.plot(e_true_mev, x_alt,
            alpha=0.7, linewidth=0.9, label=f"Smooth every 10 + final (χ²={result_smooth.chi2[-1]:.1f})")
    for cert_kev, _ in EU152_CERT:
        ax.axvline(cert_kev / 1000.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, 1.8)
    ax.set_xlabel("E_true (MeV)")
    ax.set_ylabel("Reconstructed x")
    ax.legend(fontsize=9)
    ax.set_title("MLEM Eu152 — smoothed vs unsmoothed comparison")
    fig2.tight_layout()
    out_cmp = OUT_DIR / "fig2_comparison_smooth.png"
    fig2.savefig(out_cmp, dpi=200)
    print(f"Saved comparison → {out_cmp}")
    plt.close(fig2)

    # ── Chi-squared convergence ──
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(result_nosmooth.chi2, label=f"No smoothing ({result_nosmooth.n_iter_used} iters)")
    ax.semilogy(result_smooth.chi2, label=f"Smooth (every 10 + final, {result_smooth.n_iter_used} iters)")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="χ²_red = 1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reduced χ²")
    ax.legend(fontsize=9)
    ax.set_title("Chi-squared convergence")
    fig3.tight_layout()
    out_chi2 = OUT_DIR / "fig2_chi2_convergence.png"
    fig3.savefig(out_chi2, dpi=200)
    print(f"Saved χ² plot → {out_chi2}")
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
