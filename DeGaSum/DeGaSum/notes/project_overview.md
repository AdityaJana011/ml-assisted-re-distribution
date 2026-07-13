# Workspace Overview: Minor Project (DeGaSum)

The [DeGaSum](file:///Users/falak/Documents/Falak/College/Minor_Project/DeGaSum) project has been successfully restructured and refactored from a flat directory of Jupyter Notebooks into a clean, modular Python codebase. The notebooks have been converted into reusable Python scripts under `src/` and backed up to the parent directory.

---

## 1. Directory Structure

The project now follows a standard structure:

```
Minor_Project/
├── notebooks/                # Archived original Jupyter Notebooks (outside DeGaSum)
│   ├── DRF_gen.ipynb
│   ├── Data_Gen.ipynb
│   ├── DeGaSum.ipynb
│   ├── DL.ipynb
│   └── plot.ipynb
└── DeGaSum/                  # Main project folder
    ├── data/
    │   ├── raw/              # Original digitized CSV files (graph_a, graph_b, graph_c)
    │   └── processed/        # Generated data (detector_response_matrix.npy, degas_ml_training_data.npz)
    ├── models/
    │   └── degas_cnn_weights.pth # Trained 1D CNN PyTorch weights
    ├── src/                  # Refactored Python source code modules
    │   ├── __init__.py
    │   ├── drf.py            # DRF scaling & continuous matrix construction (from DRF_gen.ipynb)
    │   ├── data_generator.py # Randomized synthetic profile generator (from Data_Gen.ipynb)
    │   ├── mlem.py           # Classical stabilized ML-EM solver (from DeGaSum.ipynb)
    │   ├── model.py          # 1D CNN architecture in PyTorch (from DL.ipynb)
    │   ├── train.py          # Model training, validation, and saving loop (from DL.ipynb)
    │   └── utils.py          # Plotting, CSV data loading, and visualization helpers (from plot.ipynb)
    ├── main.py               # Main runner script to orchestrate the entire pipeline
    ├── outputs/              # Visual outputs, plots, and figures
    │   ├── figures/          # Reproduced paper plots (recon_figure_1a, 1b, 1c)
    │   └── diagnostics/      # Diagnostic plots (Sample_pair, loss curves, stress tests)
    ├── notes/                # Project documentation and papers
    │   ├── project_overview.md # Detailed project overview
    │   ├── restructure_plan.md # Restructuring plan and CLI guidelines
    │   └── papers/           # Original reference research PDFs
    │       ├── Shevelev_2013_Nucl._Fusion_53_123004.pdf
    │       ├── S1063785013010161.pdf
    │       └── ACFrOgBQoul...pdf
    └── requirements.txt      # Project dependencies (numpy, scipy, torch, matplotlib, pandas, scikit-learn)
```

---

## 2. Main Orchestrator (`main.py`)

A central entry point [main.py](file:///Users/falak/Documents/Falak/College/Minor_Project/DeGaSum/main.py) has been provided to run any step of the pipeline or the entire process end-to-end.

To run the pipeline, use the following commands:
*   **Run all steps sequentially:**
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
*   **Train the PyTorch 1D CNN:**
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

## 3. Execution Verification Results

All steps of the refactored codebase have been executed and validated:

1.  **DRF Matrix Construction:** Created `data/processed/detector_response_matrix.npy` (100x100 matrix).
2.  **Synthetic Dataset Generation:** Generated 10,000 synthetic pairs and saved them to `data/processed/degas_ml_training_data.npz`. A diagnostic plot was saved to `outputs/diagnostics/Sample_pair.png`.
3.  **1D CNN Training:** Trained the PyTorch model using MPS (Apple Silicon GPU acceleration) for 40 epochs. Final Normalized Train MSE: `0.000012` | Val MSE: `0.000010`. Saved weights to `models/degas_cnn_weights.pth`.
4.  **Classical ML-EM Inversion:** Solved a sample test case using stabilized ML-EM and saved the reconstruction verification plot to `outputs/diagnostics/degas_reconstruction_performance.png`.
5.  **Figure Reconstruction:** Plotted and cleaned the raw CSV data to reproduce Panels (a), (b), and (c) from Figure 1 of the reference paper, saving them under `outputs/figures/recon_figure_1*.png`.
