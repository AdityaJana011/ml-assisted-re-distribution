import os
import numpy as np

def three_point_average(v, kernel=np.array([0.25, 0.5, 0.25])):
    """
    Apply a 3-point moving average filter with boundary-preserving padding.
    Default kernel is [0.25, 0.5, 0.25] (binomial filter).
    """
    v_padded = np.pad(v, 1, mode='edge')
    return np.convolve(v_padded, kernel, mode='valid')

def calculate_reduced_chi_squared(y_measured, y_projected):
    """
    Calculate the reduced chi-squared statistic assuming Poisson statistics.
    """
    y_proj_safe = np.where(y_projected == 0, 1e-12, y_projected)
    return np.mean(((y_measured - y_projected) ** 2) / y_proj_safe)

def run_mlem_inversion(y_measured, h_matrix, h_backward=None, num_iterations=50, 
                       smooth_every=5, init_mode='shevelev', final_smooth=True,
                       tolerance=1e-6):
    """
    Classical Stabilized ML-EM (Richardson-Lucy) solver for spectrum deconvolution.
    Implements the 5 DeGaSum tricks:
    1. Non-negativity constraint.
    2. In-loop periodic smoothing.
    3. 30% narrower deconvolution DRF for backward step.
    4. Channel refinement support (via rectangular matrix).
    5. Final post-deconvolution smoothing.
    6. Semi-convergence tracking (stopping criterion).
    """
    n_measured = h_matrix.shape[0]
    n_true = h_matrix.shape[1]
    
    # Load H_narrow if not provided and file exists
    if h_backward is None:
        narrow_path = 'data/processed/detector_response_matrix_narrow.npy'
        if os.path.exists(narrow_path):
            h_backward = np.load(narrow_path)
            # Ensure it has matching dimensions
            if h_backward.shape != h_matrix.shape:
                print(f"[Warning] Narrow matrix shape {h_backward.shape} mismatch with forward matrix {h_matrix.shape}. Defaulting to forward matrix.")
                h_backward = h_matrix
        else:
            h_backward = h_matrix

    # Re-normalize columns to preserve particle count probability physics
    H_forward = h_matrix.copy()
    for col in range(H_forward.shape[1]):
        c_sum = np.sum(H_forward[:, col])
        if c_sum > 0:
            H_forward[:, col] /= c_sum
            
    H_backward = h_backward.copy()
    for col in range(H_backward.shape[1]):
        c_sum = np.sum(H_backward[:, col])
        if c_sum > 0:
            H_backward[:, col] /= c_sum
            
    # Calculate backward column sums
    col_sum_backward = H_backward.sum(axis=0)
    col_sum_backward[col_sum_backward == 0] = 1e-12

    # Initialization (Trick 1 addition / Shevelev)
    if init_mode == 'shevelev':
        # x_0 = y * (sum(y) / sum(H_forward @ y)) to preserve total counts
        y_projected_init = np.dot(H_forward, y_measured)
        sum_proj = np.sum(y_projected_init)
        scale = np.sum(y_measured) / sum_proj if sum_proj > 0 else 1.0
        x_recon = y_measured * scale
    else:
        # Flat average baseline
        x_recon = np.full(n_true, np.sum(y_measured) / n_true)

    prev_chi2 = float('inf')
    
    # Iterative Loop
    for p in range(1, num_iterations + 1):
        # 1. Forward projection
        y_bar = np.dot(H_forward, x_recon)
        y_bar[y_bar == 0] = 1e-12
        
        # Track reduced chi-squared (Semi-convergence / stopping rule)
        chi2 = calculate_reduced_chi_squared(y_measured, y_bar)
        if np.abs(prev_chi2 - chi2) < tolerance or chi2 <= 1.0:
            # Stop early
            break
        prev_chi2 = chi2
        
        # 2. Correction ratio
        ratio = y_measured / y_bar
        
        # 3. Back-projection of correction using H_backward (Trick 3)
        correction = np.dot(H_backward.T, ratio)
        
        # 4. Multiplicative update
        x_recon = x_recon / col_sum_backward * correction
        
        # 5. Non-negativity constraint (Trick 1)
        x_recon = np.clip(x_recon, 0.0, None)
        
        # 6. In-loop periodic smoothing (Trick 2)
        if smooth_every > 0 and p % smooth_every == 0:
            x_recon = three_point_average(x_recon)
            
    # Final post-deconvolution smoothing (Trick 5)
    x_smoothed = three_point_average(x_recon) if final_smooth else x_recon.copy()
    
    return x_recon, x_smoothed

def run_mlem_uncertainty_propagation(y_measured, h_matrix, h_backward=None, num_iterations=50,
                                     smooth_every=5, init_mode='shevelev', final_smooth=True,
                                     tolerance=1e-6, num_samples=100):
    """
    Perform Monte Carlo uncertainty propagation to compute mean and standard deviation
    error bars for the MLEM deconvolution.
    """
    x_samples = []
    # Seed for reproducibility of MC sampling
    rng = np.random.default_rng(42)
    
    for _ in range(num_samples):
        # Generate Poisson sample of measured counts
        y_sampled = rng.poisson(y_measured)
        # Run deconvolution
        _, x_smoothed = run_mlem_inversion(
            y_sampled, h_matrix, h_backward=h_backward, num_iterations=num_iterations,
            smooth_every=smooth_every, init_mode=init_mode, final_smooth=final_smooth,
            tolerance=tolerance
        )
        x_samples.append(x_smoothed)
        
    x_samples = np.array(x_samples)
    mean_x = np.mean(x_samples, axis=0)
    std_x = np.std(x_samples, axis=0)
    
    return mean_x, std_x
