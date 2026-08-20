# the-tiniest-transformer

A small PyTorch implementation of Transformer building blocks and a GPT-style language model.

This repository is meant to be read like a book. Most Transformer implementations are
highly engineered and surrounded by many tools and abstractions. Here, the code is
simplified to the bone so that the core mechanisms remain visible and easy to follow.

## Current components

- `src/attn_block.py` — causal grouped-query attention with optional KV caching.
- `src/transformer_block.py` — RMSNorm, attention, residual connections, and SwiGLU feed-forward block.
- `src/model.py` — GPT-style token embedding, Transformer blocks, normalization, and output head.
- `src/swiglu.py` — SwiGLU feed-forward layer.
- `src/rsmnorm.py` — RMSNorm.
- `src/top_k_top_p.py` — top-k and top-p logit filtering for generation.
- `src/kvcache.py` — key/value cache container.
- `src/config.py` — default training configuration.
- `src/train.py` — TinyStories training entry point.

## Run tests

From the project root:

```bash
./myenv/bin/python -m pytest
```

The tests are lightweight smoke tests for tensor shapes, filtering, and a GPT forward pass.

## Train

Install the project and its dependencies:

```bash
pip install -e .
```

Then run:

```bash
./myenv/bin/python -m src.train
```

Training defaults are defined in `src/config.py`. They can be overridden when calling `train()`:

```python
from src.train import train

train(n_layer=4, batch_size=32)
```

The training script uses the `roneneldan/TinyStories` dataset and the GPT-2 tokenizer.

## References

- Transformer attention and decoder architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- SwiGLU feed-forward layers: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
- RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
- Shared key/value heads and KV-cache motivation: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150).
- Grouped-query attention: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
- Nucleus/top-p sampling: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).
- GPT-style language modeling: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Dataset: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).

These references describe the ideas implemented here; this repository is a small educational implementation and is not intended to reproduce the full systems or experiments from those papers.
