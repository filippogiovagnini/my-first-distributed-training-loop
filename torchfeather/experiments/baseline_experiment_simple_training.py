import os

import torch

from torchfeather.config.default_configs import get_config
from torchfeather.train import Trainer


def main() -> None:
    config_name = os.environ.get("TORCHFEATHER_CONFIG", "ddp4")

    config = get_config(config_name)
    config.job.dump_folder = f"./outputs/{config_name}"

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
