# Plan: Restructuring DeGaSum Project

Below is the proposed directory structure to reorganize [DeGaSum](file:///Users/falak/Documents/Falak/College/Minor_Project/DeGaSum) into a clean, modular Python project.

## Proposed Directory Structure

```
DeGaSum/
├── data/
│   ├── raw/                  # Original digitized CSV files (graph_a, graph_b, graph_c)
│   └── processed/            # Generated data (detector_response_matrix.npy, degas_ml_training_data.npz)
├── models/
│   └── degas_cnn_weights.pth # Trained 1D CNN PyTorch weights
├── src/                      # Refactored Python source code modules
│   ├── __init__.py
│   ├── drf.py                # DRF scaling & continuous matrix construction (from DRF_gen.ipynb)
│   ├── data_generator.py     # Randomized synthetic profile generator (from Data_Gen.ipynb)
│   ├── mlem.py               # Classical stabilized ML-EM solver (from DeGaSum.ipynb)
│   ├── model.py              # 1D CNN architecture in PyTorch (from DL.ipynb)
│   ├── train.py              # Model training, validation, and saving loop (from DL.ipynb)
│   └── utils.py              # Plotting, CSV data loading, and visualization helpers (from plot.ipynb)
├── main.py                   # Main runner script to orchestrate the entire pipeline
├── outputs/                  # Visual outputs, plots, and figures
│   ├── figures/              # Reproduced paper plots (recon_figure_1a, 1b, 1c)
│   └── diagnostics/          # Diagnostic plots (Sample_pair, loss curves, stress tests)
├── notebooks/                # Archive of the original Jupyter Notebooks (optional)
└── requirements.txt          # Project dependencies (numpy, scipy, torch, matplotlib, pandas, scikit-learn)
```

---

## Refactoring Strategy & Execution Steps

### Step 1: Directory Setup
Create the new folder structure: `data/raw/`, `data/processed/`, `models/`, `src/`, and `outputs/` (with subfolders).

### Step 2: Relocate Data & Outputs
*   Move raw CSV files from `data/` to `data/raw/`.
*   Move existing generated data files (`.npy`, `.npz`) to `data/processed/`.
*   Move model weights (`degas_cnn_weights.pth`) to `models/`.
*   Move existing images and figures to `outputs/figures/` or `outputs/diagnostics/`.

### Step 3: Convert Notebooks to Python Modules
Extract and refactor the code from the Jupyter notebooks into structured `.py` files inside the `src/` folder:
1.  **`src/drf.py`**: Functionalize DRF loading, scaling, and matrix construction.
2.  **`src/data_generator.py`**: Create a configurable dataset generation class/function.
3.  **`src/mlem.py`**: Functionalize ML-EM iteration loop with adjustable stabilization params.
4.  **`src/model.py`** & **`src/train.py`**: Define PyTorch classes and train/val pipelines.
5.  **`src/utils.py`**: Code for plotting, digitizer cleaning, and file saving.

### Step 4: Create orchestrator `main.py`
Provide a central command-line interface (CLI) to run the steps (DRF generation, data generation, training, deconvolution, and plotting) either individually or end-to-end.

### Step 5: Archive original Notebooks
Move original `.ipynb` files into a `notebooks/` backup folder.

---

> [!NOTE]
> All paths in the refactored code will be updated to use relative paths starting from the project root directory.
