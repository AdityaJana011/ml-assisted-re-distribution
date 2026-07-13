# DeGaSum: Deconvolution of Gamma-Ray Spectra

**DeGaSum** is a computational physics and deep learning package designed for the deconvolution of gamma-ray energy spectra in thermonuclear fusion plasmas. It implements forward modeling, classical iterative reconstruction (Maximum Likelihood Expectation Maximization), and a data-driven 1D Convolutional Neural Network (1D CNN) to reconstruct the true incident gamma-ray energy spectra from blurred, noisy detector measurements.

This project is based on the diagnostic methodologies for fast ions and runaway electrons in tokamak devices (like ITER) described by **A.E. Shevelev et al. (2013)**.

---

## 1. Scientific Context

In fusion plasmas, gamma-ray detectors record a blurred and noise-contaminated energy spectrum due to the detector's **Response Function (DRF)**. This forward process is modeled by the Fredholm integral equation of the first kind:

$$y(\varepsilon) = \int_{0}^{+\infty} x(\varepsilon')h(\varepsilon, \varepsilon')d\varepsilon' + n(\varepsilon)$$

Where:
*   $\varepsilon$: Measured energy deposited inside the detector (Channels).
*   $\varepsilon'$: Initial true energy of incoming gamma quanta from the plasma.
*   $x(\varepsilon')$: True initial spectrum of gamma quanta (the target for deconvolution).
*   $h(\varepsilon, \varepsilon')$: Detector Response Function (DRF) matrix.
*   $n(\varepsilon)$: Statistical Poisson measurement noise.
*   $y(\varepsilon)$: Final noisy spectrum measured by the detector (the blurred input).

---

## 2. Directory Structure

```
DeGaSum/
├── data/
│   ├── raw/                  # Original digitized CSV files (graph_a, graph_b, graph_c)
│   └── processed/            # Generated data (response matrix, training datasets)
├── models/
│   └── degas_cnn_weights.pth # Trained 1D CNN PyTorch weights
├── src/                      # Refactored Python source code modules
│   ├── __init__.py
│   ├── drf.py            # DRF scaling & continuous matrix construction
│   ├── data_generator.py # Randomized synthetic profile generator
│   ├── mlem.py           # Classical stabilized ML-EM solver
│   ├── model.py          # 1D CNN architecture in PyTorch
│   ├── train.py          # Model training, validation, and saving loop
│   └── utils.py          # Plotting, CSV data loading, and visualization helpers
├── main.py               # Main runner script to orchestrate the entire pipeline
├── outputs/              # Visual outputs, plots, and figures
│   ├── figures/          # Reproduced paper plots (recon_figure_1a, 1b, 1c)
│   └── diagnostics/      # Diagnostic plots (Sample_pair, loss curves, stress tests)
├── notes/                # Project documentation and reference papers
│   ├── project_overview.md # Detailed project overview
│   ├── restructure_plan.md # Restructuring plan and CLI guidelines
│   └── papers/           # Original reference research PDFs
└── requirements.txt      # Project dependencies
```

---

## 3. Installation & Dependencies

To run the project, ensure you have the required packages installed. You can install them using:

```bash
pip install -r requirements.txt
```

Core dependencies:
*   `numpy` (Numerical operations)
*   `scipy` (Interpolation and filters)
*   `pandas` (CSV data cleaning)
*   `torch` (Deep learning model training)
*   `matplotlib` (Scientific plotting)
*   `scikit-learn` (Dataset train/test splits)

---

## 4. Execution Outline

The central entry point is [main.py](main.py), which provides a command-line interface (CLI) to run any step of the pipeline or the entire process end-to-end.

*   **Run all steps sequentially (DRF -> Data Gen -> Train -> ML-EM -> Plot):**
    ```bash
    python3 main.py --step all
    ```
*   **Build the Continuous DRF Matrix:**
    ```bash
    python3 main.py --step drf
    ```
*   **Generate Synthetic Dataset:**
    ```bash
    python3 main.py --step data_gen
    ```
*   **Train the PyTorch 1D CNN Model:**
    ```bash
    python3 main.py --step train
    ```
*   **Perform Classical ML-EM Inversion:**
    ```bash
    python3 main.py --step mlem
    ```
*   **Reproduce Reference Paper Plots:**
    ```bash
    python3 main.py --step plot
    ```

---

## 5. Module Details

*   **`src/drf.py`**: Reads raw digitized DRF files (3, 6, 8 MeV) and uses physics-based *Relative Fractional Energy Scaling* to construct a continuous $100 \times 100$ response matrix $H$, normalized to preserve probability. Saved in `data/processed/detector_response_matrix.npy`.
*   **`src/data_generator.py`**: Generates 10,000 randomized synthetic spectra (exponential backgrounds with 1–3 peaks), convolves them with the response matrix, normalizes them, and adds Poisson counting noise to create a scale-invariant ML training dataset. Saved in `data/processed/degas_ml_training_data.npz`.
*   **`src/mlem.py`**: Solves the inverse problem using the classical Maximum Likelihood Expectation Maximization (ML-EM) algorithm. Implements two stabilization rules: row-wise Gaussian matrix blurring and post-processing 3-point moving average smoothing.
*   **`src/model.py`**: Defines a 1D Convolutional Neural Network (1D CNN) in PyTorch to map scale-invariant noisy measured spectrum inputs to target true initial spectra.
*   **`src/train.py`**: Trains the PyTorch model using MSE loss and Adam optimizer. Automatically checks for and utilizes hardware acceleration (`mps` on macOS Apple Silicon, `cuda` on Nvidia GPUs, or CPU fallback). Saves calibrated weights to `models/degas_cnn_weights.pth`.
*   **`src/utils.py`**: Includes helper functions to read and clean raw digitized files, and to reproduce panels (a), (b), and (c) from Figure 1 of the reference paper.
