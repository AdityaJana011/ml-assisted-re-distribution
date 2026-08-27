# Open Physics Questions

These questions must be answered before treating the real-response reconstruction as physically final.

## 1. What exactly is `data/response_matrix.csv`?

Code-level diagnostic (full matrix):

- Matrix shape is `6100 x 599`.
- Rows are measured-energy bins from `0.5` to `6099.5 keV`.
- Columns are true-energy bins from `20` to `6000 keV`.
- Columns are nonnegative and approximately column-stochastic (sums 0.999-1.0).
- For `E_true` up to ~200-300 keV, each column peaks at `E_meas ~= E_true`
  (photopeak / diagonal ridge) with a Compton tail below.
- Above ~500 keV the full-energy peak collapses: columns peak in the lowest
  measured bin and the centroid falls well below `E_true`
  (e.g. `E_true = 6000` -> centroid ~2070 keV).

Working interpretation:

- Low-energy photopeak ridge + high-energy photofraction collapse is standard
  detector-response behavior. This looks like a detector-only response `H_d`,
  consistent with the earlier corner-slice conclusion.
- If so, the reconstructed vector `x` is a photon spectrum. An electron-to-HXR
  operator `H_e` (Bethe-Heitler) is still required to form `H_tot = H_d @ H_e`
  and reach the RE energy distribution.
- The earlier "more like `H_tot`" note came from high-energy columns / the
  geometrically mismatched `drf_corner.csv` and missed the low-energy diagonal.

Still needed:

- Confirm from the matrix source/model that this is detector-only `H_d`.
- Obtain or construct `H_e` (electron -> HXR generation).
- Confirm units, and what physical quantity each column maps from.

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

