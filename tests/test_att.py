import torch
from src.model import *
from src.attn_block import attention, Attention

def test_dot_attention():
    B, Tk, Tq, H, D = 10, 15, 30, 20, 25

    k, v = torch.unbind(torch.randn(2, B, H, Tk, D), dim=0)
    q = torch.randn(B, H, Tq, D)

    scores, out = attention(q, k, v)

    assert out.shape == q.shape

def test_attention_block():
    B, T = 10, 15

    d_model, n_heads, n_kv, sliding_window, attention_sink = 100, 20, 5, 3, 3 

    att_fn = Attention(d_model, n_heads, n_kv, sliding_window, attention_sink)
    
    x = torch.randn(B, T, d_model)

    y, _ = att_fn(x)

    assert y.shape == x.shape
