import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import SpectralTransformer1D

def visualize_attention_heads(
    dataset_path='data/processed/degas_ml_training_data.npz',
    weights_path='models/degas_transformer_weights.pth',
    sample_idx=0
):
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model = SpectralTransformer1D(num_channels=100, d_model=64, nhead=4).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("Transformer weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Load a Single Sample
    data = np.load(dataset_path)
    Y_data = data['inputs'].astype(np.float32)
    
    # Area-normalize the input just like in training
    y_sample = Y_data[sample_idx] / np.sum(Y_data[sample_idx])
    y_tensor = torch.tensor(y_sample).unsqueeze(0).unsqueeze(1).to(device) # Shape: (1, 1, 100)

    # 4. Push data through the embedding layers (Manual Forward Pass)
    with torch.no_grad():
        x_seq = y_tensor.permute(0, 2, 1)
        tokens = model.input_projection(x_seq)
        tokens = model.pos_encoder(tokens)
        
        # Isolate the first Transformer layer
        first_layer = model.transformer_encoder.layers[0]
        
        # Extract attention weights without averaging across the 4 heads
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len) -> (1, 4, 100, 100)
        _, attn_weights = first_layer.self_attn(
            tokens, tokens, tokens, 
            need_weights=True, 
            average_attn_weights=False
        )

    # Convert to NumPy and drop the batch dimension -> Shape: (4, 100, 100)
    attn_matrices = attn_weights.squeeze(0).cpu().numpy()

    # 5. Plot the 4 Heads
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        # Plot heatmap
        im = ax.imshow(attn_matrices[i], cmap='viridis', origin='lower', extent=[0, 10, 0, 10])
        ax.set_title(f'Attention Head {i+1}')
        ax.set_xlabel('Key Channel (Source Energy / MeV)')
        ax.set_ylabel('Query Channel (Scattered Energy / MeV)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Multi-Head Self-Attention Maps (Sample {sample_idx})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = 'outputs/diagnostics/transformer_heads.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Attention map plotted and saved to '{out_path}'")

if __name__ == "__main__":
    visualize_attention_heads(sample_idx=5) # You can change this to look at different spectra