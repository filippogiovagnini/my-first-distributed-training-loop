import torch
import torch.nn as nn
from src.rsmnorm import RMSNorm
from src.attn_block import Attention
from src.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, sliding_window, attention_sink, n_kv):
        super().__init__()
        Norm = RMSNorm
        self.ln1 = Norm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv, sliding_window, attention_sink)
        self.ln2 = Norm(d_model)
        self.ffn = SwiGLU(d_model, mult=4, dropout=dropout)
        
    def forward(self, x, kv_cache=None):
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache
