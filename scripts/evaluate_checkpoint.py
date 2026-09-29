"""Evaluate a saved checkpoint on the held-out local-token split.

Example, from the repository root, using four GPUs:

    TORCHFEATHER_CONFIG=ddp4_150m_1b \
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        --module scripts.evaluate_checkpoint --step 61000 --validation-steps 50

The script loads the requested checkpoint (or the latest checkpoint by default),
runs validation only, and logs validation loss/perplexity to the configured
metrics destination. It does not perform optimizer steps or alter checkpoints.
"""

import argparse
import math
import os

import torch

from torchfeather.config.default_configs import get_config
from torchfeather.experiments.baseline_experiment_simple_training import (
    LARGE_CONFIG_NAME,
    SCALED_CONFIG_NAME,
    get_large_config,
    get_scaled_config,
)
from torchfeather.train import Trainer


def build_config(config_name: str):
    if config_name == LARGE_CONFIG_NAME:
        return get_large_config()
    if config_name == SCALED_CONFIG_NAME:
        return get_scaled_config()
    return get_config(config_name)


def main() -> None:
    default_config = os.environ.get("TORCHFEATHER_CONFIG", LARGE_CONFIG_NAME)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=default_config,
        help="training config/profile used to create the checkpoint",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="run directory containing the checkpoint (defaults to the training run directory)",
    )
    parser.add_argument(
        "--step", type=int, default=-1,
        help="checkpoint step to evaluate; -1 selects the latest checkpoint",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=50,
        help="number of local validation batches per rank",
    )
    args = parser.parse_args()
    if args.step < -1:
        parser.error("--step must be -1 (latest) or a non-negative step")
    if args.validation_steps < 1:
        parser.error("--validation-steps must be positive")

    config = build_config(args.config)
    config.training.validation_steps = args.validation_steps
    config.job.dump_folder = args.output_dir or os.environ.get(
        "TORCHFEATHER_OUTPUT_DIR", f"./outputs/{args.config}"
    )

    trainer = None
    try:
        trainer = Trainer(config)
        if not trainer.checkpointer.load(step=args.step):
            raise FileNotFoundError(
                f"No checkpoint found under {config.job.dump_folder!r}"
            )
        validation_loss = trainer.validate()
        if torch.distributed.get_rank() == 0:
            print(
                f"checkpoint_step={trainer.step} "
                f"validation_loss={validation_loss:.6f} "
                f"validation_perplexity={math.exp(validation_loss):.4f}"
            )
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
