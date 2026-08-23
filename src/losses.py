import os
import torch
import torch.nn as nn
import numpy as np

class PhysicsInformedDeconvolutionLoss(nn.Module):
    def __init__(self, h_matrix_path='data/processed/detector_response_matrix.npy', 
                 lambda_residual=1.0, lambda_smoothness=1e-4, lambda_positivity=1e-2):
        super(PhysicsInformedDeconvolutionLoss, self).__init__()
        
        if not os.path.exists(h_matrix_path):
            raise FileNotFoundError(f"Missing response matrix at '{h_matrix_path}'. Run DRF construction first.")
            
        h_matrix = np.load(h_matrix_path).astype(np.float32)
        # Normalize columns so each column sums to 1.0
        col_sums = h_matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        h_matrix /= col_sums
        
        self.register_buffer('H', torch.tensor(h_matrix))
        
        self.lambda_res = lambda_residual
        self.lambda_smooth = lambda_smoothness
        self.lambda_pos = lambda_positivity
        
        self.supervised_criterion = nn.MSELoss()

    def forward(self, x_pred, x_true, y_measured):
        """
        x_pred: Model output shape -> (batch_size, N_true)
        x_true: Ground truth target shape -> (batch_size, N_true)
        y_measured: Raw detector input shape -> (batch_size, 1, M_meas) or (batch_size, M_meas)
        """
        H_device = self.H.to(x_pred.device)
        
        if y_measured.dim() == 3:
            y_measured = y_measured.squeeze(1)
            
        # Dynamically resample x_pred if model output length differs from target true energy grid
        if x_pred.shape[1] != x_true.shape[1]:
            x_pred = nn.functional.interpolate(
                x_pred.unsqueeze(1), size=x_true.shape[1], mode='linear', align_corners=False
            ).squeeze(1)

        # 1. Supervised Data Loss
        loss_supervised = self.supervised_criterion(x_pred, x_true)
        
        # 2. Physics Residual Loss: ||x_pred @ H.T - y||^2
        y_simulated = torch.matmul(x_pred, H_device.T)
        loss_residual = torch.mean((y_simulated - y_measured) ** 2)
        
        # 3. Smoothness Penalty (1D Total Variation)
        loss_smoothness = torch.mean(torch.abs(x_pred[:, 1:] - x_pred[:, :-1]))
        
        # 4. Positivity Constraint: Penalize negative values
        loss_positivity = torch.mean(torch.relu(-x_pred) ** 2)
        
        # Combined objective optimization
        total_loss = (loss_supervised + 
                      self.lambda_res * loss_residual + 
                      self.lambda_smooth * loss_smoothness + 
                      self.lambda_pos * loss_positivity)
                      
        return total_loss