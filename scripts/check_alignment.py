import numpy as np
import pandas as pd

new_data = np.loadtxt("data/LaBr3 spectrum 1kb/LaBr3_Ba133_5mC.txt", skiprows=1)
new_energy = new_data[:, 0]

drf = pd.read_csv("data/response_matrix.csv", index_col=0)
drf_energy = drf.index.values

print(f"new data:  starts at {new_energy[0]}, step {new_energy[1]-new_energy[0]}")
print(f"DRF axis:  starts at {drf_energy[0]}, step {drf_energy[1]-drf_energy[0]}")
print(f"aligned: {np.isclose(new_energy[0], drf_energy[0])}")