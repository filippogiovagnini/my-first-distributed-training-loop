import copy
import json

import numpy as np
import pytest
import torch

from scripts.prepare_fineweb_tokens import prepare_tokens
from torchfeather.components.tokenizer import ByteTokenizer
from torchfeather.config.default_configs import get_config
from torchfeather.datasets.loader import build_dataloader
from torchfeather.datasets.local_tokens import LocalTokenDataset


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "tokens"
    records = [{"text": "abcdefghijklmnopqrstuvwxyz 🌍" * 30}]
    prepare_tokens(records, path, 803, {"fixture": True})
    return path


def test_preparation_is_bounded_and_preserves_token_ids(prepared):
    expected = ByteTokenizer().encode(
        "abcdefghijklmnopqrstuvwxyz 🌍" * 30, add_bos=True, add_eos=True
    )[:803]
    actual = np.fromfile(prepared / "tokens.bin", dtype="<u2")
    assert actual.tolist() == expected
    assert json.loads((prepared / "metadata.json").read_text())["token_count"] == 803
    with pytest.raises(FileExistsError):
        prepare_tokens([], prepared, 1, {})


def test_preparation_failure_does_not_publish_partial_data(tmp_path):
    target = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_tokens([{"text": "hi"}], target, 100, {})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_four_ranks_partition_blocks_and_shift_labels(prepared):
    tokens = np.fromfile(prepared / "tokens.bin", dtype="<u2")
    seen = []
    for rank in range(4):
        dataset = LocalTokenDataset(str(prepared), 7, rank, 4, infinite=False)
        samples = list(dataset)
        assert len(samples) == 25
        for position, (inputs, labels) in enumerate(samples):
            index = position * 4 + rank
            seen.append(index)
            expected = torch.tensor(tokens[index * 8:index * 8 + 8].astype(np.int64))
            torch.testing.assert_close(inputs["input"], expected[:-1])
            torch.testing.assert_close(labels, expected[1:])
    assert sorted(seen) == list(range(100))


def test_loader_resume_is_exact_across_dataset_wrap(prepared, monkeypatch):
    monkeypatch.setenv("TORCHFEATHER_DATA_PATH", str(prepared))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: pytest.fail("Network loader called"))
    config = get_config("ddp4")
    config.training.seq_len = 7
    config.training.local_batch_size = 2

    def build():
        return build_dataloader(4, 1, ByteTokenizer(), config)

    loader = build()
    iterator = iter(loader)
    for _ in range(12):
        next(iterator)  # 24 of 25 samples; next batch crosses the epoch boundary
    state = copy.deepcopy(loader.state_dict())
    expected = [next(iterator) for _ in range(3)]
    resumed = build()
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    for expected_inputs, expected_labels in expected:
        inputs, labels = next(resumed_iterator)
        torch.testing.assert_close(inputs["input"], expected_inputs["input"])
        torch.testing.assert_close(labels, expected_labels)


def test_resume_rejects_changed_layout(prepared):
    state = LocalTokenDataset(str(prepared), 7, 0, 4).state_dict()
    with pytest.raises(ValueError, match="seq_len changed"):
        LocalTokenDataset(str(prepared), 8, 0, 4).load_state_dict(state)
    with pytest.raises(ValueError, match="dp_world_size changed"):
        LocalTokenDataset(str(prepared), 7, 0, 2).load_state_dict(state)


def test_missing_too_small_and_corrupt_data_fail_early(prepared, tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_fineweb_tokens"):
        LocalTokenDataset(str(tmp_path / "missing"), 7)
    with pytest.raises(ValueError, match="one complete sequence"):
        LocalTokenDataset(str(prepared), 512, 0, 4)
    with (prepared / "tokens.bin").open("r+b") as stream:
        stream.write(b"\xff\xff")
    with pytest.raises(ValueError, match="checksum"):
        LocalTokenDataset(str(prepared), 7)
