import os
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_dataset(h_matrix_path='data/processed/detector_response_matrix.npy',
                               out_npz_path='data/processed/degas_ml_training_data.npz',
                               num_samples=10000,
                               total_counts=10000,
                               seed=42,
                               plot_sample=False,
                               plot_out_path='outputs/diagnostics/Sample_pair.png'):
    if not os.path.exists(h_matrix_path):
        raise FileNotFoundError(f"Response matrix not found at {h_matrix_path}. Run DRF construction first.")
        
    h_matrix = np.load(h_matrix_path)
    n_channels = h_matrix.shape[0]
    
    epsilon_prime = np.linspace(0.01, 10.0, n_channels)
    d_epsilon_prime = epsilon_prime[1] - epsilon_prime[0]
    
    X_targets = []
    Y_inputs = []
    
    np.random.seed(seed)
    print(f"Starting corrected generation of {num_samples} training pairs...")
    
    for i in range(num_samples):
        # Generate random physical background (exponential decay)
        decay_constant = np.random.uniform(2.0, 5.0)
        bg_amplitude = np.random.uniform(50, 150)
        x_background = np.exp(-epsilon_prime / decay_constant) * bg_amplitude
        
        # Inject a random number of physical peaks (1 to 3 peaks)
        num_peaks = np.random.randint(1, 4)
        x_peaks = np.zeros_like(epsilon_prime)
        
        for _ in range(num_peaks):
            center = np.random.uniform(2.0, 8.0)
            amplitude = np.random.uniform(100, 600)
            width = np.random.uniform(0.12, 0.28)
            x_peaks += amplitude * np.exp(-(epsilon_prime - center)**2 / (2 * width**2))
            
        x_raw = x_background + x_peaks
        y_ideal_raw = np.dot(h_matrix, x_raw) * d_epsilon_prime
        
        # Synchronized Physical Normalization
        if np.sum(y_ideal_raw) > 0:
            scale_factor = total_counts / np.sum(y_ideal_raw)
            x_true = x_raw * scale_factor
            y_ideal = y_ideal_raw * scale_factor
        else:
            x_true = x_raw
            y_ideal = y_ideal_raw
            
        # Superimpose Poisson statistical counting noise
        y_noisy = np.random.poisson(y_ideal)
        
        X_targets.append(x_true)
        Y_inputs.append(y_noisy)
        
    X_targets = np.array(X_targets)
    Y_inputs = np.array(Y_inputs)
    
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, inputs=Y_inputs, targets=X_targets)
    print(f"Success! Saved dataset as '{out_npz_path}'. Shape: {Y_inputs.shape}")
    
    if plot_sample:
        os.makedirs(os.path.dirname(plot_out_path), exist_ok=True)
        sample_idx = np.random.randint(0, num_samples)
        plt.figure(figsize=(7, 4.5))
        plt.plot(epsilon_prime, X_targets[sample_idx], label="Target: True Plasma $x(\\epsilon')$", color="black", alpha=0.5)
        plt.step(epsilon_prime, Y_inputs[sample_idx], label="Input: Noisy Detector $y(\\epsilon)$", color="crimson", where="mid")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.title(f"Visualizing Sample Data Pair #{sample_idx}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(plot_out_path, dpi=300)
        plt.close()
        print(f"Quality control plot saved to {plot_out_path}")
        
    return Y_inputs, X_targets
