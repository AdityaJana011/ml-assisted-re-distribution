# Repository Map

This branch currently contains several generations of the project. Use this map to avoid mixing fake, reference, and current real-response work.

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

## Real-H Notebook Area

- `v2_real_H_benchmark/Stage1_RealData_H_and_validation - Copy.ipynb`
  - Older reference-paper DRF benchmark notebook.
  - Not the current real-response mainline.
  - Contains a stale final-cell error.
- `v2_real_H_benchmark/*.png`
  - Exploratory plots from real/reference validation.
  - Treat as generated or discussion artifacts unless explicitly needed.

## Fake-H Placeholder Area

- `v1_gaussian_placeholder/`
  - Early toy pipeline using fake `100 x 100` matrices.
  - Useful only as learning/archive material.
  - Not valid for the current real-response pipeline.

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

