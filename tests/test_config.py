import torch

from torchfeather.components.tokenizer import ByteTokenizer
from torchfeather.config.default_configs import get_config
from torchfeather.datasets import DatasetConfig
from torchfeather.model.model import DeepSeekV3Model


def test_tiny_baselines_fit_parameter_budget_and_tokenizer() -> None:
    for name in ("ddp1", "ddp4"):
        config = get_config(name)
        with torch.device("meta"):
            model = DeepSeekV3Model(config.model.args)
        assert 450_000 <= sum(p.numel() for p in model.parameters()) <= 550_000
        assert config.model.tokenizer == "byte"
        assert config.model.args.vocab_size == ByteTokenizer.vocab_size


def test_ddp4_config_uses_four_data_parallel_ranks() -> None:
    config = get_config("ddp4")

    assert config.parallelism.data_parallel_replicate_degree == 4
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.training.seq_len == 512
    assert config.training.steps == 100


def test_dataset_config_is_constructible() -> None:
    config = DatasetConfig(
        path="example/dataset",
        loader=lambda path: path,
        sample_processor=lambda sample: sample,
    )

    assert config.path == "example/dataset"
    assert config.loader("example/dataset") == "example/dataset"
    assert config.sample_processor("sample") == "sample"
