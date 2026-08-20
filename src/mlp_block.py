import torch
import torch.nn as nn
from src.swiglu import SwiGLU

class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim, n_layers):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d_model, self.d_model), SwiGLU(d_model, mult=4, dropout=0.0), nn.Linear(self.d_model, self.d_model)) for i in range(self.n_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x