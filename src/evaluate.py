import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.model import SpectralDeconvoluter1D, SpectralTransformer1D

def evaluate_all_models(
    dataset_path='data/processed/degas_ml_training_data.npz',
    h_matrix_path='data/processed/detector_response_matrix.npy',
    baseline_weights='models/degas_cnn_weights_pure_supervised.pth',
    pinn_cnn_weights='models/degas_cnn_weights.pth',
    pinn_transformer_weights='models/degas_transformer_weights.pth',
    seed=42
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    H = torch.tensor(np.load(h_matrix_path).astype(np.float32)).to(device)
    
    data = np.load(dataset_path)
    X_data = data['targets'].astype(np.float32)
    Y_data = data['inputs'].astype(np.float32)
    
    _, Y_temp, _, X_temp = train_test_split(Y_data, X_data, test_size=0.20, random_state=seed)
    _, Y_test, _, X_test = train_test_split(Y_temp, X_temp, test_size=0.50, random_state=seed)
    
    Y_test_norm = Y_test / np.sum(Y_test, axis=1, keepdims=True)
    X_test_norm = X_test / np.sum(X_test, axis=1, keepdims=True)
    
    y_test_tensor = torch.tensor(Y_test_norm).unsqueeze(1).to(device)
    x_test_tensor = torch.tensor(X_test_norm).to(device)
    
    def eval_weights(model, weights_path, name):
        if not os.path.exists(weights_path):
            print(f"\n[Warning] {weights_path} not found. Skipping {name}.")
            return None
        
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            x_pred = model(y_test_tensor)
            
            mse_data = torch.mean((x_pred - x_test_tensor) ** 2).item()
            y_simulated = torch.matmul(x_pred, H.T)
            mse_physics = torch.mean((y_simulated - y_test_tensor.squeeze(1)) ** 2).item()
            neg_violations = torch.mean((x_pred < 0).float()).item() * 100
            total_variation = torch.mean(torch.abs(x_pred[:, 1:] - x_pred[:, :-1])).item()
            
        print(f"\n=== Performance Report: {name} ===")
        print(f"1. Target Data MSE:      {mse_data:.7f}")
        print(f"2. Physics Residual MSE: {mse_physics:.7f}")
        print(f"3. Negative Violations:   {neg_violations:.2f}%")
        print(f"4. Total Variation Code:  {total_variation:.5f}")
        return x_pred.cpu().numpy()

    # Instantiate models
    cnn_model = SpectralDeconvoluter1D().to(device)
    trans_model = SpectralTransformer1D().to(device)

    # Run checks
    x_base = eval_weights(cnn_model, baseline_weights, "Purely Supervised CNN Baseline")
    x_pinn_cnn = eval_weights(cnn_model, pinn_cnn_weights, "Physics-Informed PINN CNN")
    x_pinn_trans = eval_weights(trans_model, pinn_transformer_weights, "Physics-Informed Transformer")

    # Plot Comparison
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(9, 5))
    plt.plot(X_test_norm[idx], label="Ground Truth (x)", color="black", linestyle="--")
    if x_base is not None:
        plt.plot(x_base[idx], label="Supervised CNN Baseline", color="red", alpha=0.5)
    if x_pinn_cnn is not None:
        plt.plot(x_pinn_cnn[idx], label="PINN 1D CNN", color="blue", linewidth=1.5)
    if x_pinn_trans is not None:
        plt.plot(x_pinn_trans[idx], label="PINN Transformer", color="green", linewidth=1.5)

    plt.title("Holdout Test Set Model Comparison")
    plt.xlabel("Energy Channels")
    plt.ylabel("Normalized Yield")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("outputs/diagnostics/model_benchmark_comparison.png", dpi=300)
    print("\nBenchmark comparison plot saved to 'outputs/diagnostics/model_benchmark_comparison.png'!")

if __name__ == "__main__":
    evaluate_all_models()