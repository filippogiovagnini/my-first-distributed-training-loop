import torch

from src.rsmnorm import RMSNorm
from src.swiglu import SwiGLU
from src.top_k_top_p import top_k_top_p_filtering


def test_rmsnorm_preserves_shape():
    x = torch.randn(2, 4, 16)

    y = RMSNorm(16)(x)

    assert y.shape == x.shape


def test_swiglu_preserves_shape():
    x = torch.randn(2, 4, 16)

    y = SwiGLU(dim=16, mult=2, dropout=0.0)(x)

    assert y.shape == x.shape


def test_top_k_keeps_at_most_k_logits_per_row():
    logits = torch.randn(2, 10)

    filtered = top_k_top_p_filtering(logits, top_k=3, top_p=None)

    assert torch.isfinite(filtered).sum(dim=-1).le(3).all()
