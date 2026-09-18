import numpy as np
import pandas as pd
from pathlib import Path

drf = pd.read_csv("data/response_matrix.csv", index_col=0)
drf_energy = drf.index.values

folder = Path("data/LaBr3 spectrum 1kb")
for f in sorted(folder.glob("*.txt")):
    data = np.loadtxt(f)          # no skiprows — line 0 is real data
    energy = data[:, 0]
    step = energy[1] - energy[0]
    aligned = np.isclose(energy[0], drf_energy[0]) and np.isclose(step, drf_energy[1]-drf_energy[0])
    print(f"{f.name}: starts at {energy[0]}, step {step}, aligned: {aligned}")