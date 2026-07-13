import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

def run_mlem_inversion(y_measured, h_matrix, num_iterations=50, sigma=1.2, smooth_size=3):
    # Stabilization Rule 1: Damping the deconvolution matrix resolution
    h_deconvolve = gaussian_filter1d(h_matrix, sigma=sigma, axis=0)
    
    # Re-normalize columns to preserve particle count probability physics
    for col in range(h_deconvolve.shape[1]):
        c_sum = np.sum(h_deconvolve[:, col])
        if c_sum > 0:
            h_deconvolve[:, col] /= c_sum
            
    # Iterative Reconstruction Loop
    n_channels = h_matrix.shape[1]
    # Initialize guess for x with a flat, strictly positive average baseline
    x_recon = np.ones(n_channels) * np.mean(y_measured)
    
    for p in range(num_iterations):
        # Calculate forward projection guess
        forward_projection = np.dot(h_deconvolve, x_recon)
        forward_projection = np.where(forward_projection == 0, 1e-10, forward_projection)
        
        # Compute the correction ratio (y / forward projection)
        ratio = y_measured / forward_projection
        
        # Back-project errors into initial energy domain and update estimate
        x_recon = x_recon * np.dot(h_deconvolve.T, ratio)
        
        # Non-negativity physical constraint: max(x, 0)
        x_recon = np.maximum(x_recon, 0)
        
    # Stabilization Rule 2: Post-processing linear smoothing filter
    x_smoothed = uniform_filter1d(x_recon, size=smooth_size)
    
    return x_recon, x_smoothed
