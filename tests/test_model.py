import torch

from src.model import GPT


def test_gpt_forward_smoke():
    model = GPT(
        vocab_size=32,
        context_window=8,
        n_layer=1,
        n_heads=4,
        d_model=16,
        dropout=0.0,
        sliding_window=None,
        attention_sink=0,
        n_kv=4,
    )
    tokens = torch.randint(32, (2, 8))

    logits, loss, caches = model(tokens, targets=tokens)

    assert logits.shape == (2, 8, 32)
    assert loss.ndim == 0
    assert len(caches) == 1
