import pytest

from torchfeather.components.tokenizer import (
    ByteTokenizer, DeepSeekV3Tokenizer, validate_tokenizer_vocab,
)
from torchfeather.config.default_configs import get_deepseek_v3_model_args


def test_vocab_validation_includes_sparse_added_token_ids() -> None:
    tokenizer = object.__new__(DeepSeekV3Tokenizer)
    tokenizer.get_vocab = lambda: {"a": 0, "added_token": 128814}
    assert tokenizer.required_vocab_size == 128815
    with pytest.raises(ValueError, match="at least 128815"):
        validate_tokenizer_vocab(tokenizer, 102400)
    validate_tokenizer_vocab(tokenizer, 128815)
    validate_tokenizer_vocab(tokenizer, get_deepseek_v3_model_args().vocab_size)


def test_byte_vocab_validation_includes_special_tokens() -> None:
    with pytest.raises(ValueError, match="at least 258"):
        validate_tokenizer_vocab(ByteTokenizer(), 257)
    validate_tokenizer_vocab(ByteTokenizer(), 258)


@pytest.mark.parametrize("text", ["", "Hello, world!", "café 中文 🌍", "\x00\n\t"])
def test_byte_tokenizer_round_trip_and_special_tokens(text: str) -> None:
    tokenizer = ByteTokenizer()
    raw_tokens = tokenizer.encode(text)
    assert raw_tokens == list(text.encode("utf-8"))
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokens == [tokenizer.bos_id, *raw_tokens, tokenizer.eos_id]
    assert all(0 <= token < tokenizer.vocab_size for token in tokens)
    assert tokenizer.decode(tokens) == text


def test_byte_tokenizer_decodes_partial_unicode() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("🌍")[:-1]) == "�"
