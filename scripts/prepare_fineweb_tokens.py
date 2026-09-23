"""Run from the repo root: python -m scripts.prepare_fineweb_tokens."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from torchfeather.components.tokenizer import ByteTokenizer
from torchfeather.datasets.local_tokens import token_file_sha256


def prepare_tokens(records, output: Path, max_tokens: int, source: dict) -> dict:
    """Write a bounded prefix; publish the directory only after successful completion."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    tokenizer = ByteTokenizer()
    count = documents = 0
    try:
        with (staging / "tokens.bin").open("wb") as stream:
            for record in records:
                tokens = tokenizer.encode(record["text"], add_bos=True, add_eos=True)
                tokens = tokens[:max_tokens - count]
                stream.write(np.asarray(tokens, dtype="<u2").tobytes())
                count += len(tokens)
                documents += 1
                if count == max_tokens:
                    break
        if count < max_tokens:
            raise ValueError(f"Source exhausted after {count} tokens; requested {max_tokens}")
        metadata = {
            "format_version": 1, "tokenizer": "byte-v1", "dtype": "<u2",
            "token_count": count, "documents_read": documents,
            "sha256": token_file_sha256(staging / "tokens.bin"), "source": source,
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(output)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description="Prepare a fixed FineWeb byte-token subset.")
    parser.add_argument("--output", type=Path, default=Path("data/fineweb-byte"))
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--revision", default="main", help="HF dataset commit or branch")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a fresh directory")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = "HuggingFaceFW/fineweb"
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve dataset revision")
    print(f"Preparing {args.max_tokens:,} byte tokens from {repo}@{revision}", flush=True)
    records = load_dataset(
        repo, name="default", split="train", revision=revision, streaming=True
    ).shuffle(seed=args.shuffle_seed, buffer_size=10_000)
    metadata = prepare_tokens(records, args.output, args.max_tokens, {
        "repo": repo, "config": "default", "split": "train", "revision": revision,
        "selection": "seeded-buffer-shuffle", "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer_size": 10_000,
    })
    print(f"Saved {metadata['token_count']:,} tokens to {args.output} (uint16, 2 bytes/token)")


if __name__ == "__main__":
    main()
