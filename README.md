# the-tiniest-transformer

A tiny implementation of a GPT model. Modern machine-learning codebases are often large, complex, and highly engineered, making it difficult to understand the core ideas behind today’s models. I’m building the tiniest model I can to learn those ideas from the ground up, and to explore how fundamental understanding can remain at the core of research, regardless of how powerful our tools become.

## Structure

- `src/attn_block.py` — causal grouped-query attention with optional KV caching.
- `src/transformer_block.py` — RMSNorm, attention, residual connections, and SwiGLU feed-forward block.
- `src/model.py` — GPT-style token embedding, Transformer blocks, normalization, and output head.
- `src/swiglu.py` — SwiGLU feed-forward layer.
- `src/rsmnorm.py` — RMSNorm.
- `src/top_k_top_p.py` — top-k and top-p logit filtering for generation.
- `src/kvcache.py` — key/value cache container.
- `src/config.py` — default training configuration.
- `src/train.py` — TinyStories training entry point.

## Install the project and its dependencies:

```bash
pip install -e .
```

Training defaults are defined in `src/config.py`

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
