import time
from types import SimpleNamespace

from torchfeather.components.metrics import DeviceMemStats, MetricsProcessor


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


def test_metrics_include_per_device_and_global_throughput() -> None:
    processor = object.__new__(MetricsProcessor)
    processor.parallel_dims = SimpleNamespace(
        dp=4,
        non_data_parallel_size=1,
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
        metrics["throughput/per_device_tokens_per_sec"] * 4
    )
