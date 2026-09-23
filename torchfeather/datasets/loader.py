from torchfeather.components.dataloader import ParallelAwareDataloader
from torchfeather.components.tokenizer import ByteTokenizer, DeepSeekV3Tokenizer
from torchfeather.config import JobConfig


def build_dataloader(
    dp_world_size: int, dp_rank: int,
    tokenizer: ByteTokenizer | DeepSeekV3Tokenizer, job_config: JobConfig,
    split: str = "train", infinite: bool = True,
) -> ParallelAwareDataloader:
    if split != "train" or not infinite:
        if job_config.training.dataset != "local_tokens":
            raise ValueError("Finite validation splits currently require local_tokens")
    if job_config.training.dataset != "local_tokens":
        from torchfeather.datasets.hf_datasets import build_hf_dataloader

        return build_hf_dataloader(dp_world_size, dp_rank, tokenizer, job_config)

    from torchfeather.datasets.local_tokens import LocalTokenDataset

    if not isinstance(tokenizer, ByteTokenizer):
        raise ValueError("Prepared byte tokens require model.tokenizer='byte'")
    if not job_config.training.dataset_path:
        raise ValueError("local_tokens requires training.dataset_path")
    dataset = LocalTokenDataset(
        job_config.training.dataset_path, job_config.training.seq_len,
        dp_rank=dp_rank, dp_world_size=dp_world_size, split=split,
        infinite=infinite,
    )
    return ParallelAwareDataloader(
        dataset=dataset, dp_rank=dp_rank, dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
    )
