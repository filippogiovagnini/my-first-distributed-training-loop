from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor
