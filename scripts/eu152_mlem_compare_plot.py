# scripts/eu152_mlem_compare_plot.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.mlem import run_mlem

drf = pd.read_csv("data/response_matrix.csv", index_col=0)
h = drf.values
e_true = 20 + 10 * np.arange(h.shape[1])

data = np.loadtxt("data/LaBr3 spectrum 1kb/LaBr3_Eu152_5mC.txt")
counts_file = data[:, 1]
y = np.zeros(h.shape[0])
y[:len(counts_file)] = counts_file

result_raw = run_mlem(y, h, n_iter=300)
result_smooth = run_mlem(y, h, n_iter=300, smooth_every=10, final_smooth=True)

known_lines = [122, 245, 344, 411, 444, 563, 586, 688, 778, 867, 964, 1086, 1112, 1213, 1299, 1408]

plt.figure(figsize=(12, 6))
plt.plot(e_true, result_raw.x, label="unsmoothed", alpha=0.7)
plt.plot(e_true, result_smooth.x, label="smoothed (every 10, final)", alpha=0.7)
for line in known_lines:
    plt.axvline(line, color="gray", linestyle=":", linewidth=0.5)
plt.xlim(0, 1500)
plt.xlabel("E_true (keV)")
plt.ylabel("Reconstructed x")
plt.legend()
plt.title("Eu152 MLEM: unsmoothed vs smoothed")
plt.savefig("results/eu152_mlem_comparison.png")
print("saved results/eu152_mlem_comparison.png")