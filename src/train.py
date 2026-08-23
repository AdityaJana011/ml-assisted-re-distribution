import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.model import SpectralDeconvoluter1D, SpectralTransformer1D
from src.losses import PhysicsInformedDeconvolutionLoss

def train_deconvoluter(dataset_path='data/processed/degas_ml_training_data.npz',
                       weights_save_path=None,
                       h_matrix_path='data/processed/detector_response_matrix.npy',
                       model_type='transformer',
                       epochs=40,
                       batch_size=64,
                       learning_rate=0.001,
                       seed=42):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Success: Apple Silicon GPU Acceleration (MPS) activated!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Success: NVIDIA GPU Acceleration (CUDA) activated!")
    else:
        device = torch.device("cpu")
        print("Device defaulting to CPU.")

    # Set default save paths based on model type
    if weights_save_path is None:
        weights_save_path = f"models/degas_{model_type}_weights.pth"

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Generate data first.")
        
    data = np.load(dataset_path)
    X_data = data['targets'].astype(np.float32)
    Y_data = data['inputs'].astype(np.float32)

    Y_train, Y_temp, X_train, X_temp = train_test_split(Y_data, X_data, test_size=0.20, random_state=seed)
    Y_val, Y_test, X_val, X_test = train_test_split(Y_temp, X_temp, test_size=0.50, random_state=seed)

    Y_train_norm = Y_train / np.sum(Y_train, axis=1, keepdims=True)
    X_train_norm = X_train / np.sum(X_train, axis=1, keepdims=True)
    Y_val_norm = Y_val / np.sum(Y_val, axis=1, keepdims=True)
    X_val_norm = X_val / np.sum(X_val, axis=1, keepdims=True)

    Y_train_t = torch.tensor(Y_train_norm).unsqueeze(1)
    X_train_t = torch.tensor(X_train_norm)
    Y_val_t = torch.tensor(Y_val_norm).unsqueeze(1)
    X_val_t = torch.tensor(X_val_norm)

    # Prevent O(N^2) memory explosion in Transformer self-attention for N=6100
    if model_type == 'transformer' and Y_train_t.shape[2] > 100:
        Y_train_t = torch.nn.functional.interpolate(Y_train_t, size=100, mode='linear', align_corners=False)
        X_train_t = torch.nn.functional.interpolate(X_train_t.unsqueeze(1), size=100, mode='linear', align_corners=False).squeeze(1)
        Y_val_t = torch.nn.functional.interpolate(Y_val_t, size=100, mode='linear', align_corners=False)
        X_val_t = torch.nn.functional.interpolate(X_val_t.unsqueeze(1), size=100, mode='linear', align_corners=False).squeeze(1)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Y_val_t, X_val_t), batch_size=batch_size, shuffle=False)

    # Model instantiation
    if model_type == 'transformer':
        print("Initializing 1D Transformer Encoder...")
        model = SpectralTransformer1D(num_channels=Y_train_t.shape[2]).to(device)
    else:
        print("Initializing 1D CNN Deconvoluter...")
        model = SpectralDeconvoluter1D().to(device)

    criterion = PhysicsInformedDeconvolutionLoss(h_matrix_path=h_matrix_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting PINN training ({model_type.upper()}) for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, inputs)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
    torch.save(model.state_dict(), weights_save_path)
    print(f"Weights saved securely as '{weights_save_path}'!")
    return model