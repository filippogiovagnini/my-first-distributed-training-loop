import contextlib
import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchfeather.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchfeather.distributed import ParallelDims
from torchfeather.distributed.model_parallel import apply_ddp
from torchfeather.train import Trainer


def _check_gradients(
    rank: int, world_size: int, rendezvous: str, device_type: str
) -> None:
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=rendezvous, rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(5, 7).to(device)
        reference = copy.deepcopy(model)
        if world_size > 1:
            mesh = init_device_mesh(device_type, (world_size,))
            apply_ddp(model, mesh, enable_compile=False, enable_compiled_autograd=False)

        trainer = object.__new__(Trainer)
        trainer.model_parts = [model]
        trainer.parallel_dims = ParallelDims(
            dp_replicate=world_size, dp_shard=1, cp=1, tp=1, pp=1,
            ep=1, etp=1, world_size=world_size,
        )
        trainer.train_context = lambda _: contextlib.nullcontext()
        trainer.maybe_enable_amp = contextlib.nullcontext()
        trainer.loss_fn = cross_entropy_loss

        # Two accumulated microbatches with different numbers of valid tokens per rank.
        inputs = torch.randn(2, world_size, 1, 4, 5)
        labels = torch.randint(0, 7, (2, world_size, 1, 4))
        for worker in range(world_size):
            labels[0, worker, :, :worker] = IGNORE_INDEX
        labels[1, 0, :, :2] = IGNORE_INDEX
        inputs, labels = inputs.to(device), labels.to(device)
        local_count = (labels[:, rank] != IGNORE_INDEX).sum()
        global_count = local_count.clone()
        dist.all_reduce(global_count)

        reference_loss = cross_entropy_loss(
            reference(inputs.reshape(-1, 4, 5)), labels.reshape(-1, 4)
        ) / global_count
        reference_loss.backward()
        logged_loss = torch.zeros((), device=device)
        for microbatch in range(2):
            loss = trainer.forward_backward_step(
                {"input": inputs[microbatch, rank]}, labels[microbatch, rank],
                global_count,
            )
            logged_loss += loss.detach()
        dist.all_reduce(logged_loss)
        torch.testing.assert_close(logged_loss, reference_loss.detach())

        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)

        torch.optim.SGD(model.parameters(), lr=0.1).step()
        torch.optim.SGD(reference.parameters(), lr=0.1).step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_accumulated_gradients_and_update_match_global_batch(
    tmp_path, world_size, device_type
):
    if device_type == "cuda" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"Requires NCCL and {world_size} CUDA devices")
    mp.spawn(
        _check_gradients,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), device_type),
        nprocs=world_size,
        join=True,
    )
