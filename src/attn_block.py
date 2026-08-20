import torch
import torch.nn as nn
from src.kvcache import KVCache

def attention(q, k, v):
    # q [B, H, Tq, D], k [B, H, Tk, D], v [B, H, Tk, D]
    Tq, Tk = q.shape[-2], k.shape[-2]
    att = q @ k.transpose(-2, -1)
    
    key_positions = torch.arange(Tk, device=q.device)
    query_positions = torch.arange(Tk - Tq, Tk, device=q.device)
    mask = key_positions[None, :] > query_positions[:, None]

    scores = torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1)

    return scores, scores @ v # [B, H, Tq, D]

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv, sliding_window, attention_sink):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.d_head = self.d_model // self.n_heads
        self.group_size = self.n_heads // self.n_kv # for each group we will use the same k and v for all the different q
        self.sliding = sliding_window
        self.sink = attention_sink

        self.wq = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv * self.d_head, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv * self.d_head, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x, kv_cache=None):
        # x has shape [B, T, H*D]
        B, T, C = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # [B, H, T, D]
        k = self.wk(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2) # [B, n_kv, T, D]
        v = self.wv(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2) # [B, n_kv, T, D]

        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        if self.sliding is not None and k_all.size(2) > self.sliding + self.sink:
            s = self.sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding:, :]], dim=2) # [B, n_kv, attention_sink + sliding_window, D]
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding:, :]], dim=2) # [B, n_kv, attention_sink + sliding_window, D]

        if self.n_kv != self.n_heads:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1) # [B, n_kv, attention_sink + sliding_window, D] -> [B, H, attention_sink + sliding_window, D]
            v_attn = v_all.repeat_interleave(self.group_size, dim=1) # [B, n_kv, attention_sink + sliding_window, D] -> [B, H, attention_sink + sliding_window, D]
        else:
            k_attn, v_attn = k_all, v_all # [B, H, attention_sink + sliding_window, D]

        _, y = attention(q, k_attn, v_attn) # ( q: [B, H, T, D] ) x ( k^T: [B, H, D, attention_sink + sliding_window] ) x ( v: [B, H, attention_sink + sliding_window, D] ) -> ( y: [B, H, T, D] )

        y = y.transpose(1, 2).reshape(B, T, C) # back to [B, T, H*D]

        y = self.proj(y)

        return y, KVCache(k=k_all, v=v_all)
