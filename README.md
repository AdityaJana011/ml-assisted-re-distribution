# ml-assisted-re-distribution

ML-assisted deconvolution of gamma-ray energy spectra for the ADITYA-U real detector
response matrix (`6100 x 599`). Combines classical iterative reconstruction (ML-EM) with
data-driven approaches to recover the true incident spectrum from blurred, noisy
detector measurements.

Based on the diagnostic methodology for fast ions and runaway electrons in tokamaks
described by **A.E. Shevelev et al. (2013)**.

## Scientific context

The forward process is a Fredholm integral equation of the first kind:

$$y(\varepsilon) = \int_{0}^{+\infty} x(\varepsilon')\,h(\varepsilon, \varepsilon')\,d\varepsilon' + n(\varepsilon)$$

- $\varepsilon$ &mdash; measured energy deposited in the detector (channels)
- $\varepsilon'$ &mdash; true energy of the incoming gamma quanta
- $x(\varepsilon')$ &mdash; true incident spectrum (the deconvolution target)
- $h(\varepsilon, \varepsilon')$ &mdash; detector response function (DRF) matrix
- $n(\varepsilon)$ &mdash; Poisson measurement noise
- $y(\varepsilon)$ &mdash; measured (blurred, noisy) spectrum

## Current status

The actively worked branch is `aditya` (`origin/aditya`). Recent history shows the
current notebook-led work is in
`v2_real_H_benchmark/Stage2_LaBr3_Benchmark_EDA.ipynb`, added in commit `5af5746`
on 2026-09-08 and modified further in the current worktree.

`v2_real_H_benchmark/Stage2_RealDRF_MLEM.ipynb` is an earlier real-DRF ML-EM
benchmark notebook. It is still useful background, but it is not the newest current
working notebook.

The `v3_ml/` notebooks are recent ML experiments:

- `v3_ml/PINN_Transformer_scaffold.ipynb` &mdash; scaffold experiment on the real response matrix
- `v3_ml/MLP_baseline.ipynb` &mdash; newer baseline following the scaffold

The project is currently notebook-led with supporting Python utilities. The top-level
`src/`, `scripts/`, and `tests/` directories are not dead pre-reset code: the old
July `src/` was removed on 2026-08-06, and a smaller current `src/` was added on
2026-08-27. These modules are imported by the validation script, tests, the LaBr3 EDA
notebook, and the v3 ML notebooks. They are not yet a complete packaged pipeline.

## Layout

- `v2_real_H_benchmark/Stage2_LaBr3_Benchmark_EDA.ipynb` &mdash; current working notebook for LaBr3 calibration/data EDA, response-matrix baseline validation, and rebinning work
- `v2_real_H_benchmark/Stage2_RealDRF_MLEM.ipynb` &mdash; earlier real-DRF ML-EM benchmark notebook, now background/reference for the newer LaBr3 EDA work
- `v3_ml/` &mdash; recent ML notebooks; exploratory, not a packaged training pipeline
- `src/` &mdash; current support modules for response-matrix loading/validation, synthetic counts, and ML-EM utilities
- `scripts/validate_stage2_real_response.py` &mdash; CLI validation runner using the current `src/` modules
- `tests/test_core_math.py` &mdash; sanity tests for parser and core math utilities
- `data/raw/` and `data/processed/` &mdash; tracked legacy/reference data from the earlier DeGaSum/reference path
- `data/response_matrix.csv`, `data/drf_corner.csv`, and `data/LaBr3 spectrum data/` &mdash; local ignored data inputs used by the current notebooks/scripts when present
- `archive/` &mdash; superseded notebooks and exploratory plots, kept for reference only; note that `archive/README.md` still names `Stage2_RealDRF_MLEM.ipynb` as the mainline, which is stale relative to the current branch state
- `falak-reference-code/` &mdash; older external/reference DeGaSum implementation with its own code, model, notes, and outputs; not the active top-level workflow

There is no `docs/` directory in the current worktree. Older notebook references to
`docs/...` are stale unless those files are restored from git history.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

> **Note:** `.gitattributes` declares the `nbstripout` filter but not the tool itself.
> After cloning, run:
>
> ```bash
> pip install nbstripout
> nbstripout --install
> ```
>
> to enable output-stripping on this machine.

The main real response matrix is local/ignored. Place it at `data/response_matrix.csv`,
then run:

```bash
python scripts/validate_stage2_real_response.py
```

Validation metrics and figures are written under `results/` (git-ignored).

## Start here

Open `v2_real_H_benchmark/Stage2_LaBr3_Benchmark_EDA.ipynb` for the current working
analysis path. Use `scripts/validate_stage2_real_response.py` for the scripted
response-matrix validation baseline, and `tests/test_core_math.py` for the focused
sanity tests around shared utilities.
