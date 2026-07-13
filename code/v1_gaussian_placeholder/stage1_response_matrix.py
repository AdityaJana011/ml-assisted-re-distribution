"""
STAGE 1: Build the total response matrix H_tot

PHYSICS RECAP (from our earlier discussion):
  H_tot = H_d @ H_e

  H_e[k, i] = probability that an electron with TRUE energy E_i produces
              a bremsstrahlung photon with energy E_k.
              (Bethe-Heitler cross-section, "electron -> photon" layer)

  H_d[j, k] = probability that a photon with energy E_k is RECORDED by
              the detector in channel E_j.
              (Gaussian smearing with energy-dependent resolution,
               "photon -> detector channel" layer)

  H_tot[j, i] = probability that an electron with true energy E_i
                ends up contributing a count in detector channel E_j.

Both H_e and H_d are built so that EACH COLUMN sums to 1 (each column is
a probability distribution: "if I have ONE electron/photon at this energy,
how does it distribute across the next stage?").
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display, just save figures
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# STEP 0: Define the energy grid
# -----------------------------------------------------------------------
# We use the SAME grid for: true electron energy, photon energy, and
# detector channel energy. This keeps indexing simple (square matrices).
N_BINS = 50
E_MIN = 0.0   # MeV
E_MAX = 15.0  # MeV

# bin edges and bin centers
E_EDGES = np.linspace(E_MIN, E_MAX, N_BINS + 1)
E_CENTERS = 0.5 * (E_EDGES[:-1] + E_EDGES[1:])
DE = E_EDGES[1] - E_EDGES[0]  # bin width (MeV)

print(f"Energy grid: {N_BINS} bins from {E_MIN} to {E_MAX} MeV, bin width = {DE:.3f} MeV")


# -----------------------------------------------------------------------
# STEP 1: H_e -- electron -> photon (Bethe-Heitler bremsstrahlung)
# -----------------------------------------------------------------------
# The (non-relativistic-corrected, simplified) Bethe-Heitler differential
# cross-section for bremsstrahlung has the well-known approximate form:
#
#     dsigma/dk  ~  (1/k) * [ some slowly varying function of k/E ]
#
# The key physical facts we need to capture (this is the "shape", not an
# exact cross-section -- good enough for building/testing our MLEM code):
#
#   1. A photon can ONLY have energy k <= E (electron energy)
#      -> hard cutoff at k = E
#   2. The spectrum dN/dk diverges roughly as 1/k for small k
#      (lots of low-energy "soft" photons)
#   3. The spectrum falls off as k approaches E (high-energy photons
#      are less likely, but not zero)
#
# We use a commonly-used simplified form:
#     dN/dk  ~ (1/k) * (1 - k/E + (3/4)*(k/E)^2)   for 0 < k <= E
#            = 0                                    for k > E
#
# This is the simplified Bethe-Heitler shape (thin-target, high-energy
# approximation) used in many textbook treatments. It captures the 1/k
# divergence and the falloff toward k=E.

def bethe_heitler_spectrum(E_electron, k_array):
    """
    Compute dN/dk (un-normalized) for an electron of energy E_electron,
    evaluated at each photon energy in k_array.

    Returns an array the same shape as k_array.
    Values for k > E_electron or k <= 0 are set to 0 (physically forbidden).
    """
    spectrum = np.zeros_like(k_array)

    # only consider photon energies strictly between 0 and E_electron
    valid = (k_array > 0) & (k_array < E_electron)

    k = k_array[valid]
    E = E_electron
    x = k / E  # fraction of electron energy carried by the photon

    # dN/dk ~ (1/k) * (1 - x + 0.75 * x^2)
    spectrum[valid] = (1.0 / k) * (1.0 - x + 0.75 * x**2)

    return spectrum


def build_H_e(E_centers, E_edges):
    """
    Build the electron -> photon matrix H_e.

    H_e[k_idx, i_idx] = probability that an electron in true-energy bin
                        i_idx produces a photon landing in photon-energy
                        bin k_idx.

    Each COLUMN i_idx is normalized to sum to 1 (it's a probability
    distribution over photon energy, for a fixed electron energy).

    Special case: if an electron's energy is too low to produce any
    photon in our grid (e.g. E_electron is in the lowest bin, so almost
    no valid k < E_electron), we just put all probability into the
    lowest photon bin (electron deposits ~its own energy as one photon).
    """
    n = len(E_centers)
    H_e = np.zeros((n, n))

    for i, E_electron in enumerate(E_centers):
        # evaluate the bremsstrahlung shape at each photon-bin center
        spectrum = bethe_heitler_spectrum(E_electron, E_centers)

        total = spectrum.sum()

        if total > 0:
            H_e[:, i] = spectrum / total  # normalize column to sum=1
        else:
            # electron too low-energy to produce a photon within our grid
            # -> treat as: all its energy goes into the lowest bin
            H_e[0, i] = 1.0

    return H_e


H_e = build_H_e(E_centers=E_CENTERS, E_edges=E_EDGES)

print(f"H_e shape: {H_e.shape}")
print(f"H_e column sums (should all be 1.0): "
      f"min={H_e.sum(axis=0).min():.4f}, max={H_e.sum(axis=0).max():.4f}")


# -----------------------------------------------------------------------
# STEP 2: H_d -- photon -> detector channel (Gaussian resolution)
# -----------------------------------------------------------------------
# A monoenergetic photon of energy E does not land in a single detector
# channel -- it gets spread into a Gaussian shape centered at E, with a
# width (FWHM) that depends on E.
#
# Resolution formula (from the papers):
#     FWHM(E) = a + b*sqrt(E) + c*E^2
#
# We pick placeholder a, b, c values in the same ballpark as the NaI(Tl)
# detectors discussed in the papers (11.5% resolution at 661.6 keV).
# These are PLACEHOLDERS -- to be replaced with ADITYA-U-specific values
# once available.
#
# Gaussian: FWHM = 2*sqrt(2*ln(2)) * sigma  ~=  2.3548 * sigma
#   -> sigma = FWHM / 2.3548

# Placeholder resolution parameters (units: MeV, MeV^0.5, 1/MeV)
A_RES = 0.03   # MeV   (constant term -- electronic noise floor)
B_RES = 0.05   # MeV^0.5 (dominant term -- statistical/scintillation noise)
C_RES = 0.001  # 1/MeV  (small term -- nonlinearity at high energy)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def fwhm_at_energy(E):
    """FWHM(E) = a + b*sqrt(E) + c*E^2, with a floor to avoid FWHM=0 at E=0."""
    fwhm = A_RES + B_RES * np.sqrt(np.abs(E)) + C_RES * E**2
    return np.maximum(fwhm, 0.01)  # floor: never let resolution go to zero


def build_H_d(E_centers):
    """
    Build the photon -> detector-channel matrix H_d.

    H_d[j_idx, k_idx] = probability that a photon in photon-energy bin
                        k_idx is recorded in detector channel j_idx.

    Each COLUMN k_idx is a Gaussian centered at E_centers[k_idx], with
    sigma = FWHM(E_centers[k_idx]) * FWHM_TO_SIGMA, evaluated at all
    detector-channel centers, then normalized to sum to 1.
    """
    n = len(E_centers)
    H_d = np.zeros((n, n))

    for k_idx, E_photon in enumerate(E_centers):
        sigma = fwhm_at_energy(E_photon) * FWHM_TO_SIGMA

        # Gaussian evaluated at every detector-channel center
        gauss = np.exp(-0.5 * ((E_centers - E_photon) / sigma) ** 2)

        H_d[:, k_idx] = gauss / gauss.sum()  # normalize column to sum=1

    return H_d


H_d = build_H_d(E_centers=E_CENTERS)

print(f"H_d shape: {H_d.shape}")
print(f"H_d column sums (should all be 1.0): "
      f"min={H_d.sum(axis=0).min():.4f}, max={H_d.sum(axis=0).max():.4f}")


# -----------------------------------------------------------------------
# STEP 3: Combine -- H_tot = H_d @ H_e
# -----------------------------------------------------------------------
# This is the matrix-multiplication composition we discussed:
# "chain the two probability transformations through the intermediate
# photon-energy variable".
H_tot = H_d @ H_e

print(f"H_tot shape: {H_tot.shape}")
print(f"H_tot column sums (should all be ~1.0): "
      f"min={H_tot.sum(axis=0).min():.4f}, max={H_tot.sum(axis=0).max():.4f}")


# -----------------------------------------------------------------------
# STEP 4: Save matrices for use in later stages
# -----------------------------------------------------------------------
np.savez("response_matrices.npz",
         E_centers=E_CENTERS,
         E_edges=E_EDGES,
         H_e=H_e,
         H_d=H_d,
         H_tot=H_tot)
print("\nSaved H_e, H_d, H_tot, and energy grid to response_matrices.npz")


# -----------------------------------------------------------------------
# STEP 5: Visualize
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- (0,0) H_e heatmap ---
im0 = axes[0, 0].imshow(H_e, origin="lower", aspect="auto",
                         extent=[E_MIN, E_MAX, E_MIN, E_MAX], cmap="viridis")
axes[0, 0].set_title("H_e: electron -> photon\n(Bethe-Heitler)")
axes[0, 0].set_xlabel("True electron energy (MeV)")
axes[0, 0].set_ylabel("Photon energy (MeV)")
plt.colorbar(im0, ax=axes[0, 0], label="probability")

# --- (0,1) H_d heatmap ---
im1 = axes[0, 1].imshow(H_d, origin="lower", aspect="auto",
                         extent=[E_MIN, E_MAX, E_MIN, E_MAX], cmap="viridis")
axes[0, 1].set_title("H_d: photon -> detector channel\n(Gaussian resolution)")
axes[0, 1].set_xlabel("Photon energy (MeV)")
axes[0, 1].set_ylabel("Detector channel energy (MeV)")
plt.colorbar(im1, ax=axes[0, 1], label="probability")

# --- (0,2) H_tot heatmap ---
im2 = axes[0, 2].imshow(H_tot, origin="lower", aspect="auto",
                         extent=[E_MIN, E_MAX, E_MIN, E_MAX], cmap="viridis")
axes[0, 2].set_title("H_tot = H_d @ H_e\n(electron -> detector channel)")
axes[0, 2].set_xlabel("True electron energy (MeV)")
axes[0, 2].set_ylabel("Detector channel energy (MeV)")
plt.colorbar(im2, ax=axes[0, 2], label="probability")

# --- (1,0): individual columns of H_e (response to single-energy electrons) ---
for E_test in [2, 6, 10]:  # MeV
    i_idx = np.argmin(np.abs(E_CENTERS - E_test))
    axes[1, 0].plot(E_CENTERS, H_e[:, i_idx], label=f"e- at {E_CENTERS[i_idx]:.1f} MeV")
axes[1, 0].set_title("H_e columns:\nphoton spectrum from one electron energy")
axes[1, 0].set_xlabel("Photon energy (MeV)")
axes[1, 0].set_ylabel("probability")
axes[1, 0].legend()

# --- (1,1): individual columns of H_d (detector response to single-energy photons) ---
for E_test in [2, 6, 10]:  # MeV
    k_idx = np.argmin(np.abs(E_CENTERS - E_test))
    axes[1, 1].plot(E_CENTERS, H_d[:, k_idx], label=f"photon at {E_CENTERS[k_idx]:.1f} MeV")
axes[1, 1].set_title("H_d columns:\ndetector response to one photon energy")
axes[1, 1].set_xlabel("Detector channel energy (MeV)")
axes[1, 1].set_ylabel("probability")
axes[1, 1].legend()

# --- (1,2): individual columns of H_tot (full response to single-energy electrons) ---
for E_test in [2, 6, 10]:  # MeV
    i_idx = np.argmin(np.abs(E_CENTERS - E_test))
    axes[1, 2].plot(E_CENTERS, H_tot[:, i_idx], label=f"e- at {E_CENTERS[i_idx]:.1f} MeV")
axes[1, 2].set_title("H_tot columns:\nfull detector response to one electron energy")
axes[1, 2].set_xlabel("Detector channel energy (MeV)")
axes[1, 2].set_ylabel("probability")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("stage1_response_matrix.png", dpi=120)
print("\nSaved plot to stage1_response_matrix.png")
