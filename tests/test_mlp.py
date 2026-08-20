import torch

from src.mlp_block import MLP


def test_mlp_preserves_shape():
    model = MLP(d_model=16, hidden_dim=32, n_layers=1)
    x = torch.randn(2, 5, 16)

    y = model(x)

    assert y.shape == x.shape
