import math
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


class PositionalEncoding1D(nn.Module):
    """
    1D Sinusoidal Positional Encoding to preserve absolute energy channel scale identity (MeV).
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding1D, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class SpectralTransformer1D(nn.Module):
    """
    1D Bidirectional Transformer Encoder for Gamma-Ray Spectrum Deconvolution.
    Provides O(1) global self-attention across energy channels.
    """
    def __init__(
        self, 
        num_channels: int = 100, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 3, 
        dim_feedforward: int = 128, 
        dropout: float = 0.1
    ):
        super(SpectralTransformer1D, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding1D(d_model=d_model, max_len=num_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Enforces physical positivity (counts >= 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 1, 100) or (batch_size, 100)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_seq = x.permute(0, 2, 1)  # (batch_size, 100, 1)
        tokens = self.input_projection(x_seq)  # (batch_size, 100, d_model)
        tokens = self.pos_encoder(tokens)
        
        encoded_tokens = self.transformer_encoder(tokens)  # (batch_size, 100, d_model)
        out = self.regression_head(encoded_tokens)  # (batch_size, 100, 1)
        
        return out.squeeze(-1)  # (batch_size, 100)