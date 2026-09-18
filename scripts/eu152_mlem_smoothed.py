    # scripts/eu152_mlem_smoothed.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.mlem import run_mlem

drf = pd.read_csv("data/response_matrix.csv", index_col=0)
h = drf.values
e_true = 20 + 10 * np.arange(h.shape[1])

data = np.loadtxt("data/LaBr3 spectrum 1kb/LaBr3_Eu152_5mC.txt")
counts_file = data[:, 1]

y = np.zeros(h.shape[0])
y[:len(counts_file)] = counts_file

# Only change from last run: smoothing turned on
result = run_mlem(y, h, n_iter=300, smooth_every=10, final_smooth=True)

top_idx = np.sort(np.argsort(result.x)[::-1][:15])
print("Top reconstructed peaks (smoothed):")
for i in top_idx:
    print(f"  E_true = {e_true[i]:5.0f} keV -> x = {result.x[i]:.2f}")