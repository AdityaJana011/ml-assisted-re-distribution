# Repository Map

This branch currently contains several generations of the project. Use this map to avoid mixing fake, reference, and current real-response work.

## Status (2026-08-28)

- `data/response_matrix.csv` reads as a detector-only response `H_d` (6100 x 599),
  not the full operator `H_tot`. See `docs/open_physics_questions.md` Q1.
- No electron-to-HXR operator `H_e` yet, so current reconstructions are photon
  spectra, not runaway-electron distributions.
- `docs/considerations.md` (paper physics checklist) drives the pre-`H_e` work.

## Current Mainline

These are the files to use for the current ADITYA-U real-response path.

- `src/`
  - Reusable current code.
  - Handles real response-matrix loading, validation, synthetic checks, and MLEM.
- `scripts/validate_stage2_real_response.py`
  - Main command-line validation runner for the real response matrix.
- `tests/`
  - Small sanity tests for parser and core math.
- `docs/`
  - Handoff notes, data setup, and open physics questions.
- `v2_real_H_benchmark/Stage2_RealDRF_MLEM.ipynb`
  - Current research notebook for the supplied real response matrix.

## Local Data

- `data/response_matrix.csv`
  - Supplied real response matrix.
  - Ignored by git because it is large.
  - Required locally to run the real-response validation script.
- `data/drf_corner.csv`
  - Debug/sample cutout from the response matrix.
  - Not canonical.

## Reference Paper Data

- `data/raw/graph_a*.csv`
- `data/raw/graph_b*.csv`
- `data/raw/graph_c*.csv`

These are digitized reference-paper curves. They are useful for comparison and background understanding, but they are not the supplied real `6100 x 599` response matrix.

## Archive

- `archive/` — superseded generations, reference only, do not build on these:
  - `v1_gaussian_placeholder/` — earliest toy pipeline, fake `100 x 100` matrices
  - `Stage1_RealData_H_and_validation - Copy.ipynb` — old paper-DRF benchmark, stale final-cell error
  - `v2_exploratory_plots/` — early validation plots, no captions
  - see `archive/README.md`

## ML Scaffold

- `v3_ml/PINN_Transformer_scaffold.ipynb` — runnable patch-transformer + physics-informed
  loss on the real `H_d`. Starting point only: underfits (posterior-mean collapse,
  val relative-L2 ~0.69). Caveats + next steps in the notebook's first and last cells.

## Branches

- `aditya` (this) — real `6100 x 599` `H_d` validation + ML scaffold; no `H_e`
- `master` — older `100 x 100` lineage (`main.py`, `src/drf.py`, `src/model.py`)
- `origin/trial/-transformer` — `SpectralTransformer1D` + attention plots; `master` layout, placeholder `H`
- `origin/trial/-PINN` — physics-loss (`src/losses.py`) + supervised CNN; `master` layout, placeholder `H`
- `origin/trial/-1d-cnn-tricks` — CNN tuning
- Trial branches predate this branch's real-response work and have diverged; porting a
  model onto the real-response line is a task, not a merge.

## External Reference

- `falak-reference-code/`
  - Falak's original/reference DeGaSum-style code.
  - Uses the old `100 x 100` response/data setup.
  - Good for ideas and comparison, but not directly compatible with the current `6100 -> 599` real-response path.

## Generated Output

- `results/`
  - Generated validation metrics and figures.
  - Ignored by git.
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
  - Generated runtime/cache files.
  - Ignored by git.

