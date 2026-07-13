import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.model import SpectralDeconvoluter1D

def train_deconvoluter(dataset_path='data/processed/degas_ml_training_data.npz',
                       weights_save_path='models/degas_cnn_weights.pth',
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

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Generate data first.")
        
    data = np.load(dataset_path)
    X_data = data['targets'].astype(np.float32)
    Y_data = data['inputs'].astype(np.float32)

    # Train-Validation-Test Split
    Y_train, Y_temp, X_train, X_temp = train_test_split(Y_data, X_data, test_size=0.20, random_state=seed)
    Y_val, Y_test, X_val, X_test = train_test_split(Y_temp, X_temp, test_size=0.50, random_state=seed)

    # Convert raw absolute vectors to pure area-normalized shapes (Sum to 1.0)
    Y_train_norm = Y_train / np.sum(Y_train, axis=1, keepdims=True)
    X_train_norm = X_train / np.sum(X_train, axis=1, keepdims=True)
    Y_val_norm = Y_val / np.sum(Y_val, axis=1, keepdims=True)
    X_val_norm = X_val / np.sum(X_val, axis=1, keepdims=True)

    # Shape for 1D CNN layout
    Y_train_t = torch.tensor(Y_train_norm).unsqueeze(1)
    X_train_t = torch.tensor(X_train_norm)
    Y_val_t = torch.tensor(Y_val_norm).unsqueeze(1)
    X_val_t = torch.tensor(X_val_norm)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Y_val_t, X_val_t), batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SpectralDeconvoluter1D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Normalized Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")

    # Save model weights
    os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
    torch.save(model.state_dict(), weights_save_path)
    print(f"Scale-invariant weights saved securely as '{weights_save_path}'!")
    return model
