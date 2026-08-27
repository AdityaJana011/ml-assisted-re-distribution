# DeGaSum Physics & Deconvolution Reference Guide

This document synthesizes all physical principles, mathematical formulations, and algorithmic modifications presented in **Khilkevitch et al. (2013)**, **Shevelev et al. (2013)**, and **Purohit et al. (2020)**. It serves as an audit checklist to ensure that the computational deconvolution and machine learning pipelines maintain fidelity to nuclear physics and tokamak diagnostics.

  

## 1. Mathematical Formulation of the Forward Problem

### 1.1 Single-Layer (Detector-Only) Formulation

For monoenergetic or discrete gamma-ray spectra where electron-to-photon conversion is not explicitly modeled, the recorded spectrum $y(\epsilon)$ is related to the true photon energy distribution $x(\epsilon')$ by a Fredholm integral equation of the first kind:


$$y(\epsilon) = \int_0^{+\infty} x(\epsilon') h(\epsilon, \epsilon') d\epsilon' + n(\epsilon)$$

In discretized matrix form:

$$y = H x + n$$

- $x$: True photon spectrum (vector of dimension $N_{\text{fine}}$ or $N_{\text{coarse}}$).
- $y$: Measured spectrum recorded across $M$ energy channels.
- $H$: Detector Response Function (DRF) matrix of size $M \times N$.
- $n$: Poisson counting noise vector.
### 1.2 Two-Layer (RE Bremsstrahlung + Detector) Formulation

For Runaway Electron (RE) diagnostics, the measured Hard X-ray (HXR) spectrum $y(\epsilon)$ is produced via a two-step physical process (electron $\rightarrow$ Bremsstrahlung photon $\rightarrow$ detector output):

$$y(\epsilon) = \int_0^\infty d\epsilon' h_d(\epsilon, \epsilon') \int_0^\infty d\epsilon'' h_e(\epsilon', \epsilon'') f(\epsilon'') + n(\epsilon) = \int_0^\infty d\epsilon'' f(\epsilon'') h_{\text{tot}}(\epsilon, \epsilon'') + n(\epsilon)$$

In matrix operator form:

$$y = H_{\text{tot}} f + n \quad \text{where} \quad H_{\text{tot}} = H_d @ H_e$$

- $f(\epsilon'')$: True Runaway Electron energy distribution function.

- $H_e(\epsilon', \epsilon'')$: Bremsstrahlung emission kernel (probability that an electron with energy $\epsilon''$ produces a photon of energy $\epsilon'$ directed toward the detector).

- $H_d(\epsilon, \epsilon')$: Detector instrumental response matrix (probability that a photon of energy $\epsilon'$ is registered in detector channel $\epsilon$).

- $H_{\text{tot}}$: Combined total response operator.

## 2. Physical Considerations for Response Matrices ($H_e$, $H_d$, $H_{\text{tot}}$)

### 2.1 Detector Response Function ($H_d$)

A realistic DRF $H_d$ cannot be modeled purely as a symmetrical Gaussian; it contains distinct spectral features:

- **Full Energy Peak (Photopeak):** Represents total photo-absorption in the crystal.

- **Compton Continuum & Shelf:** Photons that undergo Compton scattering and escape the crystal without depositing their full energy.

- **Escape Peaks:** Single and double escape peaks resulting from pair production ($\epsilon > 1.022\text{ MeV}$) followed by annihilation photon escape.

- **Energy-Dependent Resolution:** The detector line-broadening Full-Width at Half-Maximum ($\text{FWHM}$) scales non-linearly with photon energy:
$$\text{FWHM}(E) = a + b\sqrt{E} + c E^2$$

### 2.2 Bremsstrahlung & Transport Kernel ($H_e$)

- **Target Nuclei $Z^2$ Scaling:** Bremsstrahlung yield scales with $Z^2$ of the target nuclei (e.g., plasma hydrogen/deuterium $Z=1$, impurities like Carbon $Z=6$, Beryllium $Z=4$, Argon $Z=18$, or High-$Z$ limiter materials).

- **Thick-Target vs. Thin-Target Emission:**
    
    - _Thin-target (Plasma core):_ REs interact with thermal ions/impurities; emissions are strongly forward-peaked along the electron pitch angle.
    - _Thick-target (Limiter/PFC impact):_ REs strike the solid limiter (e.g., ADITYA-U graphite limiter); electrons deposit all remaining energy, dominating the total HXR yield.

- **Line-of-Sight Attenuation:** Physical materials along the diagnostic sightline (vacuum windows, port walls, collimator structures, or dedicated $77\text{ cm}$ Polyethylene or $\text{LiH}$ neutron filters) attenuate low-energy photons and must be explicitly folded into $H_e$ or $H_d$.

## 3. The 5 Algorithmic Modifications (Khilkevitch / DeGaSum Tricks)

Standard MLEM (Richardson-Lucy) updates are computed as:

$$x_i^p = x_i^{p-1} \sum_j h_{j,i} \frac{y_j}{\sum_k h_{j,k} x_k^{p-1}}$$

To prevent instability, noise amplification, and unphysical oscillations on continuous spectra, **five specific modifications** must be implemented:

```
+-----------------------------------------------------------------------------------+
|                           5 DEGASUM ALGORITHMIC TRICKS                             |
+-----------------------------------------------------------------------------------+
|  1. Non-Negativity Enforcer   -->  x_i^p = max(x_i^p, 0) at each iteration step    |
|  2. In-Loop Periodic Smooth   -->  Convolve x with kernel every 'j' iterations    |
|  3. 30% Narrower DRF (H_narrow)-->  Scale DRF FWHM by 0.7 for backward step        |
|  4. Channel Refinement        -->  Interpolate E-grid to finer bins before MLEM   |
|  5. Final Post-Smooth         -->  3-point linear smooth after iteration ends     |
+-----------------------------------------------------------------------------------+
```

### Detail Breakdown

1. **Non-Negativity Clipping:** Enforce $x_i^p = \max(x_i^p, 0)$ at every iteration step to prohibit unphysical negative photon or electron densities.

2. **In-Loop Periodic Smoothing:** Perform spatial/channel smoothing on $x$ after every $j$ iterations:
    - For continuous Bremsstrahlung HXR spectra: $j = 5 \text{--} 10$ iterations.
    - For discrete nuclear gamma lines: $j = 10 \text{--} 20$ iterations.

3. **30% Narrower Deconvolution DRF ($H_{\text{narrow}}$):** Compute $H_{\text{narrow}}$ with a resolution/FWHM reduced by $30\%$ ($\text{FWHM}_{\text{decon}} = 0.7 \times \text{FWHM}_{\text{physical}}$) specifically for the backward iteration matrix. This mathematically suppresses peak splitting and high-frequency ringing.
    
4. **Channel Refinement via Interpolation:** Interpolate measured spectrum channels onto a finer energy grid ($N_{\text{fine}} = \text{REFINE} \times N_{\text{coarse}}$) prior to running MLEM. This makes $H$ rectangular ($M \times N_{\text{fine}}$) and enhances energy resolution.
    
5. **Final Post-Deconvolution Smoothing:** Apply a 3-point moving average filter ($\left[\frac{1}{4}, \frac{1}{2}, \frac{1}{4}\right]$ or similar) once after the iterative loop completes to eliminate residual high-frequency artifacts.

### Stopping Criterion & Convergence

- **Initial Approximations:** $x^0 = y / \Vert{}H y\Vert{}$.
- **Chi-Squared Tracking:** Monitor the reduced chi-squared $\chi^2_{\text{red}}$ between forward-projected estimates $\hat{y} = H x^p$ and measured data $y$.
- **Semi-Convergence Trap:** Because MLEM fits noise at high iteration counts, stop iterations when the residual approaches the noise floor ($\chi^2_{\text{red}} \approx 1.0$) or when the residual variation stabilizes:

$$\Vert{}y - H x^{p-1}\Vert{} - \Vert{}y - H x^p\Vert{} < \epsilon$$

## 4. Diagnostics & Machine-Specific Details (ADITYA-U & JET)

### 4.1 ADITYA-U Tokamak Parameters (Purohit 2020)

- **Detector:** $1.5'' \times 1.5''$ $\text{LaBr}_3(\text{Ce})$ scintillator (BrillanCe-380) with 10-stage Hamamatsu R6231 PMT and Canberra Osprey MCA.
    
- **Spectrometer Specs:** Light yield $\approx 61,000\text{ photons/MeV}$, decay time $16\text{ ns}$, energy resolution $\approx 2.8\text{--}3.0\%$ at $662\text{ keV}$.
    
- **Line-of-Sight Geometry:** Tangential view through the equatorial plane terminating on the inboard/outboard poloidal graphite limiters ($\approx 4\text{ m}$ distance).
    
- **Spectral Features:** Measured HXR spectra range from $75\text{--}90\text{ keV}$ up to $\sim 3\text{ MeV}$.
    
- **Bi-Maxwellian Feature:** HXR spectra show two distinct temperature slopes:
    

$$I(h\nu) = \frac{n_b e^{-h\nu/T_b}}{\sqrt{T_b}} + \frac{n_t e^{-h\nu/T_t}}{\sqrt{T_t}}$$

- _Bulk RE Temperature ($T_{\text{RE}}$):_ $100\text{--}600\text{ keV}$ (measured below $700\text{--}800\text{ keV}$).
    
- _Tail RE Component:_ Extends above $800\text{ keV}$.
    
- _Compton Region:_ Above $700\text{--}800\text{ keV}$, Compton scattering dominates over photoelectric absorption in $\text{LaBr}_3$, distorting raw spectral slopes unless deconvoluted via DRF.
    
- **Empirical Confinement Scaling ($\tau_{\text{RE}}$):**
    
$$\epsilon_r = (n_e^{-0.15} T_e^{-0.06} Z_{\text{eff}}^{-0.03}) (V_p^{1.02} \tau_{\text{RE}}^{1.1})$$

Estimated $\tau_{\text{RE}} = 1\text{--}6\text{ ms}$ for ADITYA-U.

## 5. Verification Checklist for Pipeline Implementation

|**Category**|**Requirement / Feature**|**Physics / Algorithmic Reason**|**Status / Implementation Check**|
|---|---|---|---|
|**Forward Model**|Two-layer operator $H_{\text{tot}} = H_d @ H_e$|Separates electron-to-photon physics ($H_e$) from detector blurring ($H_d$).|Required for RE electron spectrum $f(E)$ reconstruction.|
|**Noise Model**|Poisson counting statistics $n \sim \text{Poisson}(y)$|MLEM is mathematically derived assuming Poisson likelihood.|Add Poisson noise to synthetic $y$; do not use Gaussian noise.|
|**Deconvolution**|Non-negativity constraint $\max(x, 0)$|Prevents unphysical negative particle or photon densities.|Enforce `np.clip(x, 0, None)` in every iteration.|
|**Deconvolution**|$30\%$ Narrower DRF ($0.7 \times \text{FWHM}$)|Mechanically suppresses reconstruction oscillations and peak splitting.|Pass $H_{\text{narrow}}$ (width scale = 0.7) to backward step.|
|**Regularization**|In-loop periodic smoothing every $j$ steps|Prevents MLEM from over-fitting high-frequency Poisson noise.|Apply 3-point smooth every $j=5\text{--}10$ iterations.|
|**Regularization**|Post-deconvolution 3-point smoothing|Cleans residual iteration artifacts after loop termination.|Apply 3-point moving average once after MLEM completes.|
|**Resolution**|Channel refinement via $E$-grid interpolation|Improves energy resolution and peak location precision.|Interpolate $E_{\text{fine}}$ to make $H$ rectangular ($M \times N_{\text{fine}}$).|
|**Stopping Rule**|$\chi^2_{\text{red}}$ tracking & discrepancy principle|Avoids semi-convergence trap where error grows at high iterations.|Halt when $\Vert{}y - H x^p\Vert{} \approx \text{noise floor}$.|
|**Edge Artifacts**|Preserve endpoints during convolution|Zero-padding in standard convolution artificially pulls 10 MeV tail down.|Ensure boundary padding mode does not drag high-energy counts to 0.|
