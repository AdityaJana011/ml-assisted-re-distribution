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

## Layout

- `src/` &mdash; reusable code: real response-matrix loading, validation, synthetic checks, ML-EM
- `scripts/validate_stage2_real_response.py` &mdash; CLI validation runner for the real response matrix
- `tests/` &mdash; sanity tests for the parser and core math
- `docs/` &mdash; handoff notes, data setup, open physics questions, repo map
- `v2_real_H_benchmark/Stage2_RealDRF_MLEM.ipynb` &mdash; current research notebook
- `archive/` &mdash; superseded earlier generations, kept for reference only
- `falak-reference-code/` &mdash; external reference implementation (older `100 x 100` setup)

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Large data files are not tracked. Place the supplied response matrix at
`data/response_matrix.csv` (conventions in `docs/data_setup.md`), then run:

```bash
python scripts/validate_stage2_real_response.py
```

Validation metrics and figures are written under `results/` (git-ignored).

## Start here

`docs/repo_map.md` &mdash; maps every part of the tree and flags what is current vs. superseded.
