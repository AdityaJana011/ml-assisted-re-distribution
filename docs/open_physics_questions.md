# Open Physics Questions

These questions must be answered before treating the real-response reconstruction as physically final.

## 1. What exactly is `data/response_matrix.csv`?

Current code-level diagnostic:

- Matrix shape is `6100 x 599`.
- Rows are measured-energy bins from `0.5` to `6099.5 keV`.
- Columns are true-energy bins from `20` to `6000 keV`.
- Columns are nonnegative and approximately column-stochastic.
- Tested columns peak at very low measured energy, not near `E_meas = E_true`.
- No strong diagonal photopeak or 511/1022 keV escape-peak pattern appears.

Working interpretation:

- This behaves more like a full forward operator `H_tot` than a detector-only response `H_d`.
- If true, the reconstructed vector `x` should be interpreted as the source/RE-side distribution represented by the matrix columns.

Still needed:

- Confirm from the matrix source/model whether this CSV already includes electron-to-HXR generation plus detector response, or only detector response.
- Confirm what physical quantity each column maps from: RE energy distribution, photon source spectrum, or another simulated source basis.

## 2. What are the units and normalization conventions?

Current code treats each column as a probability-like response because column sums are approximately one.

Still needed:

- Confirm whether column normalization was applied intentionally.
- Confirm whether bin widths are already included in the matrix values.
- Confirm the intended units of the reconstructed distribution.
- Confirm whether absolute count intensity should be preserved or whether only distribution shape is meaningful.

## 3. What should stop MLEM at runtime?

Current validation shows that chi-square near one does not necessarily match the minimum distribution-space error.

Example from `bulk_plus_tail` validation:

- Noisy best relative-L2 iteration: `30`.
- Chi-square closest to one: iteration `4`.

Still needed:

- Decide whether runtime stopping should use reduced chi-square, relative change in `x`, semiconvergence/U-curve calibration, a fixed iteration count, or a hybrid rule.
- Define the reduced chi-square convention formally, including variance floor and degrees of freedom.

## 4. How should resolution limits be handled?

The current delta validation recovers the correct peak location but has notable residual shape error.

Example from `delta_2mev` validation:

- Noiseless relative L2 after 300 iterations: about `0.110`.
- Noisy best iteration hit the current cap at `300`.

Still needed:

- Decide whether this is acceptable physical broadening, an iteration-limit issue, or a sign that additional regularization/narrowing is required.
- Confirm whether the narrowed-response trick is appropriate for this real matrix and how it should be constructed.

