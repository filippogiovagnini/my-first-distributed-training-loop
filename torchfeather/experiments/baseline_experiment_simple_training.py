"""Launch the small baselines or the documented 50M/300M-token DDP experiment.

Prepare its shuffled local-token dataset once from the repository root:

    python -m scripts.prepare_fineweb_tokens \
        --output data/fineweb-byte-100m --max-tokens 100000000

Then launch the scaled profile on four GPUs:

    TORCHFEATHER_CONFIG=ddp4_50m_300m \
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        --module torchfeather.experiments.baseline_experiment_simple_training

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


def get_scaled_config() -> JobConfig:
    """Build the named four-GPU, ~50M-parameter, ~300M-token profile."""
    config = get_config(BASE_CONFIG_NAME)

    # Keep the byte vocabulary and all other architecture dimensions from the
    # benchmark profile; grow only each routed/shared expert's hidden width.
    config.model.tokenizer = "byte"
    config.model.hf_assets_path = ""
    config.model.args.moe_inter_dim = MODEL_MOE_INTERMEDIATE_SIZE
    config.model.args.max_seq_len = SEQUENCE_LENGTH

    # Spell out the DDP layout and batch math so the effective token budget is clear.
    config.parallelism.data_parallel_replicate_degree = DATA_PARALLEL_DEGREE
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.training.dataset = "local_tokens"
    config.training.dataset_path = os.environ.get(
        "TORCHFEATHER_DATA_PATH", DEFAULT_DATASET_PATH
    )
    config.training.local_batch_size = LOCAL_BATCH_SIZE
    config.training.global_batch_size = GLOBAL_BATCH_SIZE
    config.training.seq_len = SEQUENCE_LENGTH
    config.training.steps = TRAINING_STEPS

    config.training.validation_interval_steps = VALIDATION_INTERVAL_STEPS
    config.training.validation_steps = VALIDATION_STEPS
    config.training.generation_max_new_tokens = GENERATION_MAX_NEW_TOKENS
    config.checkpoint.interval = CHECKPOINT_INTERVAL_STEPS
    return config


def main() -> None:
    config_name = os.environ.get("TORCHFEATHER_CONFIG", "ddp4")

    if config_name == SCALED_CONFIG_NAME:
        config = get_scaled_config()
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
