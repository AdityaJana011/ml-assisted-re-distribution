# Logic Flow:
# 1. Load real DRF (H) and Eu152 measured spectrum (y)
# 2. Pad y with zeros to match H's full 6100-row E_meas axis (Eu152 emits nothing beyond its file's range)
# 3. Run current run_mlem() as-is -- sanity check, not the final rigorous run
# 4. Print the 15 largest reconstructed peaks and their E_true, compare against known Eu152 lines by eye

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path, so `src` imports work

import numpy as np
import pandas as pd
from src.mlem import run_mlem

drf = pd.read_csv("data/response_matrix.csv", index_col=0)
h = drf.values                                         # shape (6100, 599)
e_true = 20 + 10 * np.arange(h.shape[1])               # known grid: 20-6000 keV, 10 keV steps

data = np.loadtxt("data/LaBr3 spectrum 1kb/LaBr3_Eu152_5mC.txt")
counts_file = data[:, 1]

y = np.zeros(h.shape[0])
y[:len(counts_file)] = counts_file
print(f"y: {len(counts_file)} real points padded to {h.shape[0]} to match H")

result = run_mlem(y, h, n_iter=300)

top_idx = np.sort(np.argsort(result.x)[::-1][:15])
print("\nTop reconstructed peaks:")
for i in top_idx:
    print(f"  E_true = {e_true[i]:5.0f} keV -> x = {result.x[i]:.2f}")