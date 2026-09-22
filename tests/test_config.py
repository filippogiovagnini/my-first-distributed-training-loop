from torchfeather.config.default_configs import get_config
from torchfeather.datasets import DatasetConfig


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
