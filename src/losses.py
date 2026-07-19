import torch
import torch.nn as nn

class PhysicsInformedDeconvolutionLoss(nn.Module):
    def __init__(self, h_matrix_path='data/processed/detector_response_matrix.npy', 
                 lambda_residual=1.0, lambda_smoothness=1e-4, lambda_positivity=1e-2):
        super(PhysicsInformedDeconvolutionLoss, self).__init__()
        
        # Load your physical response matrix and register it as a non-trainable buffer
        import numpy as np
        h_matrix = np.load(h_matrix_path).astype(np.float32)
        self.register_buffer('H', torch.tensor(h_matrix))
        
        # Weights for the multi-objective loss optimization
        self.lambda_res = lambda_residual
        self.lambda_smooth = lambda_smoothness
        self.lambda_pos = lambda_positivity
        
        # Supervised component baseline matching your old framework
        self.supervised_criterion = nn.MSELoss()

    def forward(self, x_pred, x_true, y_measured):
        """
        x_pred: Model output shape -> (batch_size, 100)
        x_true: Ground truth target shape -> (batch_size, 100)
        y_measured: Raw detector input shape -> (batch_size, 1, 100)
        """

        self.H = self.H.to(x_pred.device)
        
        # Remove the channel dimension from the input spectrum to get (batch_size, 100)
        y_measured = y_measured.squeeze(1)
        
        # ----------------------------------------------------
        # 1. Supervised Data Loss (Your original metric)
        # ----------------------------------------------------
        loss_supervised = self.supervised_criterion(x_pred, x_true)
        
        # ----------------------------------------------------
        # 2. Physics Residual Loss: ||H * x_pred - y||^2
        # ----------------------------------------------------
        # Because x_pred is (batch_size, 100), we compute x_pred @ H.T 
        # to get the simulated detector measurement.
        y_simulated = torch.matmul(x_pred, self.H.T)
        loss_residual = torch.mean((y_simulated - y_measured) ** 2)
        
        # ----------------------------------------------------
        # 3. Smoothness Penalty (Total Variation Loss)
        # ----------------------------------------------------
        # Enforces that neighboring energy channels fluctuate continuously, 
        # smoothing out high-frequency noise oscillations.
        loss_smoothness = torch.mean(torch.abs(x_pred[:, 1:] - x_pred[:, :-1]))
        
        # ----------------------------------------------------
        # 4. Positivity Constraint: Penalize negative photon counts
        # ----------------------------------------------------
        # ReLU of (-x_pred) isolates negative predictions and squares them.
        loss_positivity = torch.mean(torch.relu(-x_pred) ** 2)
        
        # Combined objective optimization
        total_loss = (loss_supervised + 
                      self.lambda_res * loss_residual + 
                      self.lambda_smooth * loss_smoothness + 
                      self.lambda_pos * loss_positivity)
                      
        return total_loss