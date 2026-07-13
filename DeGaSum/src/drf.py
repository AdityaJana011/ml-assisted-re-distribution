import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def construct_drf_matrix(raw_data_dir='data/raw', processed_data_dir='data/processed', n_channels=100):
    epsilon = np.linspace(0.01, 10.0, n_channels)
    epsilon_prime = np.linspace(0.01, 10.0, n_channels)
    
    ref_epsilon_prime = np.array([3.0, 6.0, 8.0])
    files = [
        os.path.join(raw_data_dir, 'graph_a(3MeV).csv'),
        os.path.join(raw_data_dir, 'graph_a(6MeV).csv'),
        os.path.join(raw_data_dir, 'graph_a(8MeV).csv')
    ]
    
    eta_grid = np.linspace(0, 1.2, 200)
    aligned_shapes = []
    
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}.")
        
        df = pd.read_csv(file, header=None)
        df.columns = ["epsilon", "intensity"]
        
        # Clean micro-noise from digitization by grouping duplicates and sorting
        df = df.groupby("epsilon").mean().reset_index()
        df = df.sort_values(by="epsilon")
        
        # Find which reference true energy \varepsilon' this file corresponds to
        peak_idx = df["intensity"].idxmax()
        approx_true = ref_epsilon_prime[
            np.argmin(np.abs(ref_epsilon_prime - df["epsilon"].iloc[peak_idx]))
        ]
        
        # Convert coordinates to relative scale \eta
        df["eta"] = df["epsilon"] / approx_true
        
        # Interpolate digitized points onto the standard uniform eta_grid
        f_interp = interp1d(
            df["eta"], df["intensity"], bounds_error=False, fill_value=0.0
        )
        aligned_shapes.append(f_interp(eta_grid))
        
    aligned_shapes = np.array(aligned_shapes)
    
    h_matrix = np.zeros((n_channels, n_channels))
    shape_interpolator = interp1d(ref_epsilon_prime, aligned_shapes, axis=0)
    
    for j, ep in enumerate(epsilon_prime):
        # Explicitly clip ep to remain within [3.0, 8.0] MeV bounds.
        ep_clipped = np.clip(ep, ref_epsilon_prime.min(), ref_epsilon_prime.max())
        
        # Safely interpolate the relative shape profile
        interpolated_shape_eta = shape_interpolator(ep_clipped)
        
        # Convert relative \eta scale back to absolute measured scale \varepsilon
        epsilon_mapped = eta_grid * ep
        f_absolute = interp1d(
            epsilon_mapped, interpolated_shape_eta, bounds_error=False, fill_value=0.0
        )
        
        column_response = f_absolute(epsilon)
        
        # Physics Normalization: Normalize column area to preserve total event probability
        if np.sum(column_response) > 0:
            column_response /= np.sum(column_response)
            
        h_matrix[:, j] = column_response
        
    os.makedirs(processed_data_dir, exist_ok=True)
    out_path = os.path.join(processed_data_dir, 'detector_response_matrix.npy')
    np.save(out_path, h_matrix)
    print(f"Successfully generated Continuous DRF Matrix h with shape: {h_matrix.shape}")
    print(f"Saved to {out_path}")
    return h_matrix
