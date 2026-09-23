import os

import torch

from torchfeather.config.default_configs import get_config
from torchfeather.train import Trainer


def main() -> None:
    config_name = os.environ.get("TORCHFEATHER_CONFIG", "ddp4")

    config = get_config(config_name)
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
