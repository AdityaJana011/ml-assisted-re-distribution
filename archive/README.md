# Archive

Superseded material, kept for reference only. **Do not build on anything in here.**
Everything is also in git history; this folder just keeps it out of the active tree.

## v1_gaussian_placeholder/
Earliest toy pipeline using fake 100x100 Gaussian response matrices.
Not valid for the real ADITYA-U 6100x599 response path — learning history only.

## Stage1_RealData_H_and_validation - Copy.ipynb
Older reference-paper DRF benchmark (digitized graph_a/b/c curves), pre-real-matrix.
Superseded by the src/ + scripts/ validation path. Has a stale error in its final cell.

## v2_exploratory_plots/
Standalone plots from early real/reference-matrix exploration (DRF overlay, bin-size
tradeoff, first-pass check, ML-EM U-curve / semiconvergence). No captions, not wired
into any notebook. Kept in case the exploration context is useful.

Current mainline: `src/`, `scripts/`, `tests/`,
`v2_real_H_benchmark/Stage2_RealDRF_MLEM.ipynb` (see `docs/repo_map.md`).
