"""Prepared byte tokens, read locally with deterministic data-parallel sharding."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info


def token_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # hashlib.file_digest is only available in Python 3.11+.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalTokenDataset(IterableDataset, Stateful):
    def __init__(
        self, path: str, seq_len: int, dp_rank: int = 0,
        dp_world_size: int = 1, infinite: bool = True,
        split: str = "train", validation_fraction: float = 0.1,
    ):
        super().__init__()
        if seq_len < 1 or dp_world_size < 1 or not 0 <= dp_rank < dp_world_size:
            raise ValueError("Invalid sequence length or data-parallel rank/size")
        folder = Path(path)
        if not (folder / "metadata.json").is_file():
            raise FileNotFoundError(
                f"Prepared dataset missing at {folder}. Run "
                "python -m scripts.prepare_fineweb_tokens --output <path> first."
            )
        metadata = json.loads((folder / "metadata.json").read_text())
        if (metadata.get("format_version"), metadata.get("tokenizer"), metadata.get("dtype")) != (
            1, "byte-v1", "<u2"
        ):
            raise ValueError("Unsupported token dataset format or tokenizer")
        token_path = folder / "tokens.bin"
        count = metadata["token_count"]
        if count <= 0 or token_path.stat().st_size != count * 2:
            raise ValueError("Token file size does not match metadata")
        self.digest = token_file_sha256(token_path)
        if self.digest != metadata["sha256"]:
            raise ValueError("Token file checksum mismatch")
        self.tokens = np.memmap(token_path, dtype="<u2", mode="r", shape=(count,))
        if self.tokens.max() >= 258:
            raise ValueError("Token file contains IDs outside the byte vocabulary")
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        # Every rank gets the same number of disjoint (seq_len + 1)-token blocks.
        # Drop any incomplete block and fewer than dp_world_size leftover blocks.
        if split not in ("train", "validation"):
            raise ValueError(f"Unknown local token split: {split}")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        source = metadata.get("source", {})
        if split == "validation" and source.get("selection") != "seeded-buffer-shuffle":
            raise ValueError(
                "Validation requires a seeded-shuffled dataset; regenerate it with "
                "scripts.prepare_fineweb_tokens.py"
            )
        usable_samples = count // (seq_len + 1) // dp_world_size * dp_world_size
        validation_samples = (
            int(usable_samples * validation_fraction // dp_world_size)
            * dp_world_size
        )
        train_samples = usable_samples - validation_samples
        self.split_offset = train_samples if split == "validation" else 0
        self.split = split
        self.samples_per_rank = (
            validation_samples if split == "validation" else train_samples
        ) // dp_world_size
        if self.samples_per_rank == 0:
            raise ValueError("Dataset needs at least one complete sequence per DP rank")
        self.position = 0

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("LocalTokenDataset currently requires num_workers=0")
        while self.infinite or self.position < self.samples_per_rank:
            index = (
                self.split_offset
                + (self.position % self.samples_per_rank) * self.dp_world_size
                + self.dp_rank
            )
            start = index * (self.seq_len + 1)
            block = torch.from_numpy(
                self.tokens[start:start + self.seq_len + 1].astype(np.int64)
            )
            self.position += 1
            yield {"input": block[:-1]}, block[1:]

    def state_dict(self):
        return {
            "sha256": self.digest, "seq_len": self.seq_len,
            "dp_rank": self.dp_rank, "dp_world_size": self.dp_world_size,
            "split": self.split, "position": self.position,
        }

    def load_state_dict(self, state_dict):
        current = self.state_dict()
        for key in ("sha256", "seq_len", "dp_rank", "dp_world_size", "split"):
            if state_dict.get(key) != current[key]:
                if key == "split":
                    raise ValueError(
                        "Checkpoint uses a different local-token split layout "
                        "(likely created before held-out validation was added). "
                        "Start a fresh run with a new TORCHFEATHER_OUTPUT_DIR."
                    )
                raise ValueError(f"Cannot resume local dataset: {key} changed")
        position = state_dict["position"]
        if not isinstance(position, int) or position < 0:
            raise ValueError("Invalid dataset checkpoint position")
        self.position = position
