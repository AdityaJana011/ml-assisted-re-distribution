import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def construct_drf_matrix(raw_data_dir='data/raw', processed_data_dir='data/processed', 
                         n_channels_measured=100, n_channels_true=100, resolution_scale=1.0):
    epsilon = np.linspace(0.01, 10.0, n_channels_measured)
    epsilon_prime = np.linspace(0.01, 10.0, n_channels_true)
    
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
        
        # Find which reference true energy \epsilon' this file corresponds to
        peak_idx = df["intensity"].idxmax()
        approx_true = ref_epsilon_prime[
            np.argmin(np.abs(ref_epsilon_prime - df["epsilon"].iloc[peak_idx]))
        ]
        
        # Convert coordinates to relative scale \eta
        df["eta"] = df["epsilon"] / approx_true
        
        # Peak-narrowing trick (Trick 3): narrow peak around its center
        if resolution_scale != 1.0:
            eta_peak = df["eta"].iloc[peak_idx]
            df["eta"] = eta_peak + (df["eta"] - eta_peak) * resolution_scale
            
        # Interpolate digitized points onto the standard uniform eta_grid
        f_interp = interp1d(
            df["eta"], df["intensity"], bounds_error=False, fill_value=0.0
        )
        aligned_shapes.append(f_interp(eta_grid))
        
    aligned_shapes = np.array(aligned_shapes)
    
    h_matrix = np.zeros((n_channels_measured, n_channels_true))
    shape_interpolator = interp1d(ref_epsilon_prime, aligned_shapes, axis=0)
    
    for j, ep in enumerate(epsilon_prime):
        # Explicitly clip ep to remain within [3.0, 8.0] MeV bounds.
        ep_clipped = np.clip(ep, ref_epsilon_prime.min(), ref_epsilon_prime.max())
        
        # Safely interpolate the relative shape profile
        interpolated_shape_eta = shape_interpolator(ep_clipped)
        
        # Convert relative \eta scale back to absolute measured scale \epsilon
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
    out_filename = 'detector_response_matrix.npy' if resolution_scale == 1.0 else 'detector_response_matrix_narrow.npy'
    out_path = os.path.join(processed_data_dir, out_filename)
    np.save(out_path, h_matrix)
    print(f"Successfully generated Continuous DRF Matrix h (scale={resolution_scale}) with shape: {h_matrix.shape}")
    print(f"Saved to {out_path}")
    return h_matrix

def bethe_heitler_spectrum(E_electron, k_array):
    """
    Compute dN/dk (un-normalized) for an electron of energy E_electron,
    evaluated at each photon energy in k_array.
    Returns an array the same shape as k_array.
    Values for k >= E_electron or k <= 0 are set to 0 (physically forbidden).
    """
    spectrum = np.zeros_like(k_array)
    valid = (k_array > 0) & (k_array < E_electron)
    k = k_array[valid]
    E = E_electron
    x = k / E
    # dN/dk ~ (1/k) * (1 - x + 0.75 * x^2)
    spectrum[valid] = (1.0 / k) * (1.0 - x + 0.75 * x**2)
    return spectrum

def construct_bremsstrahlung_kernel(n_channels_true=100, n_channels_photon=100, processed_data_dir='data/processed'):
    """
    Build the electron -> photon matrix H_e using Bethe-Heitler cross-section.
    Each column is normalized to sum to 1.
    """
    E_electron_grid = np.linspace(0.01, 10.0, n_channels_true)
    E_photon_grid = np.linspace(0.01, 10.0, n_channels_photon)
    
    h_e = np.zeros((n_channels_photon, n_channels_true))
    
    for i, E_electron in enumerate(E_electron_grid):
        spectrum = bethe_heitler_spectrum(E_electron, E_photon_grid)
        total = spectrum.sum()
        if total > 0:
            h_e[:, i] = spectrum / total  # normalize column to sum=1
        else:
            # Low energy electron -> energy deposited in the lowest bin
            h_e[0, i] = 1.0
            
    os.makedirs(processed_data_dir, exist_ok=True)
    out_path = os.path.join(processed_data_dir, 'bremsstrahlung_emission_kernel.npy')
    np.save(out_path, h_e)
    print(f"Successfully generated Bremsstrahlung Kernel H_e with shape: {h_e.shape}")
    print(f"Saved to {out_path}")
    return h_e

def construct_total_response_matrix(h_d_path='data/processed/detector_response_matrix.npy',
                                    h_e_path='data/processed/bremsstrahlung_emission_kernel.npy',
                                    out_path='data/processed/total_response_matrix.npy'):
    """
    Compute total response matrix H_tot = H_d @ H_e to couple
    the electron distribution directly to the detector outputs.
    """
    if not os.path.exists(h_d_path):
        raise FileNotFoundError(f"Missing H_d file at {h_d_path}")
    if not os.path.exists(h_e_path):
        raise FileNotFoundError(f"Missing H_e file at {h_e_path}")
        
    h_d = np.load(h_d_path)
    h_e = np.load(h_e_path)
    
    h_tot = np.dot(h_d, h_e)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, h_tot)
    print(f"Successfully generated Total Response Matrix H_tot = H_d @ H_e with shape: {h_tot.shape}")
    print(f"Saved to {out_path}")
    return h_tot
