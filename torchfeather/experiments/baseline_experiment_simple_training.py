"""Launch the small baselines or scaled four-GPU byte-token experiments.

Prepare the shuffled local-token dataset once from the repository root. For the
150M/1B-token profile:

    python -m scripts.prepare_fineweb_tokens \
        --output data/fineweb-byte-1b --max-tokens 1000000000

Then launch the 150M/1B-token profile on four GPUs:

    TORCHFEATHER_CONFIG=ddp4_150m_1b \
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        --module torchfeather.experiments.baseline_experiment_simple_training

The earlier 50M/300M-token profile remains available as `ddp4_50m_300m` and
uses `data/fineweb-byte-100m` by default.
The dataset is prepared separately so tokenization and network I/O happen before
GPU training. Set TORCHFEATHER_DATA_PATH if it lives somewhere else.
"""

import math
import os

import torch

from torchfeather.config.default_configs import get_config
from torchfeather.config.job_config import JobConfig
from torchfeather.train import Trainer


# This profile uses the existing four-GPU benchmark as its architectural base.
SCALED_CONFIG_NAME = "ddp4_50m_300m"
LARGE_CONFIG_NAME = "ddp4_150m_1b"
BASE_CONFIG_NAME = "ddp4_benchmark"

# Dataset preparation settings. The uint16 token file is about 200 MB on disk.
PREPARED_DATASET_TOKEN_COUNT = 100_000_000
DEFAULT_DATASET_PATH = "./data/fineweb-byte-100m"

# Model settings. With the benchmark's other model dimensions unchanged, an
# MoE intermediate width of 1,152 gives exactly 51,105,024 total parameters.
MODEL_TARGET_PARAMETERS = 50_000_000
MODEL_MOE_INTERMEDIATE_SIZE = 1_152

# Four DDP ranks each process 8 sequences of 512 byte tokens per optimizer step.
DATA_PARALLEL_DEGREE = 4
LOCAL_BATCH_SIZE = 8
SEQUENCE_LENGTH = 512
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * DATA_PARALLEL_DEGREE  # 32 sequences
TOKENS_PER_OPTIMIZER_STEP = GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH  # 16,384

# Round up so training reaches the 300M byte-token target.
TARGET_TRAINING_TOKENS = 300_000_000
TRAINING_STEPS = math.ceil(
    TARGET_TRAINING_TOKENS / TOKENS_PER_OPTIMIZER_STEP
)  # 18,311 steps, 300,007,424 tokens

# Evaluate periodically, save checkpoints, and generate a short sample at the end.
VALIDATION_INTERVAL_STEPS = 500
VALIDATION_STEPS = 5
CHECKPOINT_INTERVAL_STEPS = 500
GENERATION_MAX_NEW_TOKENS = 128

# Keep the base benchmark's warmup-stable-cosine schedule explicit in this
# experiment: warm up for 50 steps, hold through about 20% of the run, then decay.
LR_WARMUP_STEPS = 50
LR_DECAY_RATIO = 0.8
LR_DECAY_TYPE = "cosine"
LR_MIN_FACTOR = 0.1

# Larger profile: one billion prepared byte tokens and a roughly 150M model.
# A width of 3,728 gives 150,023,424 total parameters with this base model.
LARGE_PREPARED_DATASET_TOKEN_COUNT = 1_000_000_000
LARGE_DEFAULT_DATASET_PATH = "./data/fineweb-byte-1b"
LARGE_MODEL_TARGET_PARAMETERS = 150_000_000
LARGE_MODEL_MOE_INTERMEDIATE_SIZE = 3_728
LARGE_TARGET_TRAINING_TOKENS = 1_000_000_000
LARGE_TRAINING_STEPS = math.ceil(
    LARGE_TARGET_TRAINING_TOKENS / TOKENS_PER_OPTIMIZER_STEP
)  # 61,036 steps, 1,000,013,824 tokens


def _build_scaled_config(
    *,
    model_moe_intermediate_size: int,
    training_steps: int,
    dataset_path: str,
) -> JobConfig:
    """Build a four-GPU local-byte-token profile from the shared benchmark base."""
    config = get_config(BASE_CONFIG_NAME)

    # Keep the byte vocabulary and all other architecture dimensions from the
    # benchmark profile; grow only each routed/shared expert's hidden width.
    config.model.tokenizer = "byte"
    config.model.hf_assets_path = ""
    config.model.args.moe_inter_dim = model_moe_intermediate_size
    config.model.args.max_seq_len = SEQUENCE_LENGTH

    # Spell out the DDP layout and batch math so the effective token budget is clear.
    config.parallelism.data_parallel_replicate_degree = DATA_PARALLEL_DEGREE
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.training.dataset = "local_tokens"
    config.training.dataset_path = os.environ.get(
        "TORCHFEATHER_DATA_PATH", dataset_path
    )
    config.training.local_batch_size = LOCAL_BATCH_SIZE
    config.training.global_batch_size = GLOBAL_BATCH_SIZE
    config.training.seq_len = SEQUENCE_LENGTH
    config.training.steps = training_steps
    config.lr_scheduler.warmup_steps = LR_WARMUP_STEPS
    config.lr_scheduler.decay_ratio = LR_DECAY_RATIO
    config.lr_scheduler.decay_type = LR_DECAY_TYPE
    config.lr_scheduler.min_lr_factor = LR_MIN_FACTOR

    config.training.validation_interval_steps = VALIDATION_INTERVAL_STEPS
    config.training.validation_steps = VALIDATION_STEPS
    config.training.generation_max_new_tokens = GENERATION_MAX_NEW_TOKENS
    config.checkpoint.interval = CHECKPOINT_INTERVAL_STEPS
    return config


def get_scaled_config() -> JobConfig:
    """Build the four-GPU, ~50M-parameter, ~300M-token profile."""
    return _build_scaled_config(
        model_moe_intermediate_size=MODEL_MOE_INTERMEDIATE_SIZE,
        training_steps=TRAINING_STEPS,
        dataset_path=DEFAULT_DATASET_PATH,
    )


def get_large_config() -> JobConfig:
    """Build the four-GPU, ~150M-parameter, ~1B-token profile."""
    return _build_scaled_config(
        model_moe_intermediate_size=LARGE_MODEL_MOE_INTERMEDIATE_SIZE,
        training_steps=LARGE_TRAINING_STEPS,
        dataset_path=LARGE_DEFAULT_DATASET_PATH,
    )


def main() -> None:
    config_name = os.environ.get("TORCHFEATHER_CONFIG", "ddp4")

    if config_name == SCALED_CONFIG_NAME:
        config = get_scaled_config()
    elif config_name == LARGE_CONFIG_NAME:
        config = get_large_config()
    else:
        config = get_config(config_name)

    # This override is useful for short smoke runs without editing the profile.
    steps_override = os.environ.get("TORCHFEATHER_STEPS")
    if steps_override is not None:
        config.training.steps = int(steps_override)
        if config.training.steps < 1:
            raise ValueError("TORCHFEATHER_STEPS must be a positive integer")

    config.job.dump_folder = os.environ.get(
        "TORCHFEATHER_OUTPUT_DIR", f"./outputs/{config_name}"
    )

    trainer = None

    try:
        trainer = Trainer(config)
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
