import time
from types import SimpleNamespace

import pytest

from torchfeather.components.metrics import DeviceMemStats, MetricsProcessor
from torchfeather.distributed import ParallelDims


class RecordingLogger:
    def __init__(self) -> None:
        self.metrics = None
        self.step = None

    def log(self, metrics, step):
        self.metrics = metrics
        self.step = step


class FixedMemoryMonitor:
    def get_peak_stats(self):
        return DeviceMemStats(1.0, 2.0, 3.0, 4.0, 0, 0)

    def reset_peak_stats(self):
        pass


@pytest.mark.parametrize(
    "replicate,shard,world_size",
    [(1, 1, 1), (4, 1, 4), (1, 4, 4), (2, 2, 4), (2, -1, 4)],
    ids=["single", "ddp4", "fsdp4", "hsdp4", "inferred-sharding"],
)
def test_metrics_include_per_device_and_global_throughput(
    replicate, shard, world_size,
) -> None:
    processor = object.__new__(MetricsProcessor)
    processor.parallel_dims = ParallelDims(
        dp_replicate=replicate, dp_shard=shard, world_size=world_size,
        cp=1, tp=1, pp=1, ep=1, etp=1,
    )
    processor.job_config = SimpleNamespace(
        metrics=SimpleNamespace(log_freq=1),
    )
    processor.device_memory_monitor = FixedMemoryMonitor()
    processor.gpu_peak_flops = 1.0
    processor.num_flops_per_token = 1.0
    processor.ntokens_since_last_log = 100
    processor.data_loading_times = [0.01]
    processor.time_last_log = time.perf_counter() - 1.0
    processor.logger = RecordingLogger()

    processor.log(
        step=1,
        global_avg_loss=2.0,
        global_max_loss=2.5,
        grad_norm=0.1,
    )

    metrics = processor.logger.metrics
    assert metrics is not None
    assert processor.logger.step == 1
    assert metrics["throughput/per_device_tokens_per_sec"] > 0
    assert metrics["throughput/global_tokens_per_sec"] == (
        metrics["throughput/per_device_tokens_per_sec"] * world_size
    )
