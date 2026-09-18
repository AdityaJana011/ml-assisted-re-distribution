import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/LaBr3 spectrum 1kb/LaBr3_Eu152_5mC.txt")
energy, counts = data[:, 0], data[:, 1]

plt.figure(figsize=(10, 5))
plt.semilogy(energy, np.clip(counts, 1, None))  # log scale, clip zeros so log doesn't break
plt.xlabel("Energy (keV)")
plt.ylabel("Counts (log scale)")
plt.title("Eu152 - raw spectrum")
plt.savefig("results/eu152_raw_check.png")
print(f"Min nonzero count: {counts[counts>0].min()}")
print(f"Max count (peak):  {counts.max()}")
print(f"Count in a quiet region (e.g. 1900-2000 keV, above highest Eu152 line 1408 keV): {counts[(energy>=1900)&(energy<2000)].mean():.2f}")