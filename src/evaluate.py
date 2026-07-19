import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.model import SpectralDeconvoluter1D

def evaluate_and_compare_weights(
    dataset_path='data/processed/degas_ml_training_data.npz',
    h_matrix_path='data/processed/detector_response_matrix.npy',
    old_weights_path='models/degas_cnn_weights_pure_supervised.pth', # Rename your old file to this to test
    new_weights_path='models/degas_cnn_weights.pth',
    seed=42
):
    # 1. Device detection setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Load the physical response matrix H
    H = torch.tensor(np.load(h_matrix_path).astype(np.float32)).to(device)
    
    # 3. Load dataset and isolate the holdout Test Set (matching train.py splits)
    data = np.load(dataset_path)
    X_data = data['targets'].astype(np.float32) # Clean ground truth
    Y_data = data['inputs'].astype(np.float32)  # Raw blurry inputs
    
    _, Y_temp, _, X_temp = train_test_split(Y_data, X_data, test_size=0.20, random_state=seed)
    _, Y_test, _, X_test = train_test_split(Y_temp, X_temp, test_size=0.50, random_state=seed)
    
    # Area-Normalize inputs and targets matching your operational pipeline
    Y_test_norm = Y_test / np.sum(Y_test, axis=1, keepdims=True)
    X_test_norm = X_test / np.sum(X_test, axis=1, keepdims=True)
    
    # Cast test vectors to PyTorch Tensors
    y_test_tensor = torch.tensor(Y_test_norm).unsqueeze(1).to(device) # (B, 1, 100)
    x_test_tensor = torch.tensor(X_test_norm).to(device)              # (B, 100)
    
    # 4. Initialize the CNN container
    model = SpectralDeconvoluter1D().to(device)
    model.eval()
    
    def compute_metrics(weights_path, name):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        with torch.no_grad():
            x_pred = model(y_test_tensor) # Model evaluation pass
            
            # Metric A: Standard Supervised Data MSE
            mse_data = torch.mean((x_pred - x_test_tensor) ** 2).item()
            
            # Metric B: Physics Residual Error ||H * x_pred - y||^2
            y_simulated = torch.matmul(x_pred, H.T)
            mse_physics = torch.mean((y_simulated - y_test_tensor.squeeze(1)) ** 2).item()
            
            # Metric C: Percentage of unphysical negative channel predictions
            neg_violations = torch.mean((x_pred < 0).float()).item() * 100
            
            # Metric D: Total Variation (Smoothness indicator)
            total_variation = torch.mean(torch.abs(x_pred[:, 1:] - x_pred[:, :-1])).item()
            
        print(f"\n=== Performance Report: {name} ===")
        print(f"1. Target Data MSE:      {mse_data:.7f}  (Lower = closer to ground truth)")
        print(f"2. Physics Residual MSE: {mse_physics:.7f}  (Lower = obeys matrix inversion)")
        print(f"3. Negative Violations:   {neg_violations:.2f}%   (Lower = realistic counts)")
        print(f"4. Total Variation Code:  {total_variation:.5f}  (Lower = fewer noise ripples)")
        return x_pred.cpu().numpy()

    # Run comparative benchmarking
    try:
        x_pred_old = compute_metrics(old_weights_path, "Purely Supervised CNN Baseline")
    except FileNotFoundError:
        print(f"\n[Warning] Old weights file not found at {old_weights_path}. Skipping baseline.")
        x_pred_old = None
        
    x_pred_new = compute_metrics(new_weights_path, "Physics-Informed PINN CNN (Current)")
    
    # 5. Visual Check: Save a qualitative plot tracking a sample profile comparison
    if x_pred_old is not None:
        import matplotlib.pyplot as plt
        idx = 0 # Look at the first entry of the test set
        plt.figure(figsize=(8, 4.5))
        plt.plot(X_test_norm[idx], label="Ground Truth (x)", color="black", linestyle="--")
        plt.plot(x_pred_old[idx], label="Old Supervised Baseline", color="red", alpha=0.6)
        plt.plot(x_pred_new[idx], label="New Physics-Informed (PINN)", color="blue", linewidth=1.5)
        plt.title("Holdout Test Set Qualitative Evaluation")
        plt.xlabel("Energy Channels")
        plt.ylabel("Normalized Yield")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig("outputs/diagnostics/pinn_vs_supervised_comparison.png", dpi=300)
        print("\nComparative graphic saved successfully to 'outputs/diagnostics/pinn_vs_supervised_comparison.png'!")

if __name__ == "__main__":
    evaluate_and_compare_weights()