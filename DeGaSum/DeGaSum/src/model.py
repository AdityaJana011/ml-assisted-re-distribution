import torch
import torch.nn as nn

class SpectralDeconvoluter1D(nn.Module):
    def __init__(self):
        super(SpectralDeconvoluter1D, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        )
        
    def forward(self, x):
        out = self.network(x)
        return out.squeeze(1)
