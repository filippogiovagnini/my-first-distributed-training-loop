import torch
import torch.nn as nn
from src.transformer_block import TransformerBlock
from src.rsmnorm import RMSNorm
from src.top_k_top_p import top_k_top_p_filtering

class GPT(nn.Module):
    def __init__(self, vocab_size, context_window, n_layer, n_heads, d_model, dropout, sliding_window, attention_sink, n_kv):
        super().__init__()
        self.context_window = context_window
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, sliding_window=sliding_window, attention_sink=attention_sink, n_kv=n_kv)
            for _ in range(n_layer)
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None, kv_cache_list=None):
        _, T = x.shape
        x = self.drop(self.tok_emb(x))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = block(x, kv_cache=cache)
            new_caches.append(cache)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_caches

    @torch.no_grad() # this is for inference time
    def generate(self, x, max_new_tokens, temperature, top_k, top_p, eos_id):
        self.eval()
        idx = x
        kvs = [None] * len(self.blocks) # initialize the cache

        for _ in range(max_new_tokens):

            first_pass = kvs[0] is None
            idx_cond = idx[:, -self.context_window:] if first_pass else idx[:, -1:] # if the current input exceeds the context window we crop it

            logits, _, kvs = self(idx_cond, kv_cache_list=kvs)

            next_logits = logits[:, -1, :]
            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-6)
                next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
                next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), 1) # sample from the probability distribution obtained

            idx = torch.cat([idx, next_id], dim=1)

            # this is for end-of-sequence special token
            if eos_id is not None and (next_id == eos_id).all():
                break

        return idx
