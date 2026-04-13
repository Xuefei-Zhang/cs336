"""Wrapper-side Prometheus metrics for serving workflows."""

from __future__ import annotations

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)


class ServeMetrics:
    """Small wrapper around Prometheus collectors used by the serve wrapper."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry: CollectorRegistry = registry or REGISTRY
        self.adapter_load_total = Counter(
            "adapter_load_total",
            "Total number of LoRA adapter loads requested through the wrapper.",
            registry=self.registry,
        )
        self.adapter_load_latency_seconds = Histogram(
            "adapter_load_latency_seconds",
            "Latency for LoRA adapter load operations.",
            registry=self.registry,
        )
        self.request_fail_total = Counter(
            "request_fail_total",
            "Total number of failed wrapper HTTP requests.",
            registry=self.registry,
        )
        self.gpu_memory_bytes = Gauge(
            "wrapper_gpu_memory_bytes",
            "Best-effort GPU memory allocated by torch.",
            registry=self.registry,
        )

    def start_server(self, host: str, port: int) -> None:
        _ = start_http_server(port, addr=host, registry=self.registry)

    def observe_adapter_load(self, latency_seconds: float) -> None:
        self.adapter_load_total.inc()
        self.adapter_load_latency_seconds.observe(latency_seconds)

    def record_request_failure(self) -> None:
        self.request_fail_total.inc()

    def update_gpu_memory(self) -> None:
        try:
            import torch
        except ImportError:
            return

        if not torch.cuda.is_available():
            return

        try:
            self.gpu_memory_bytes.set(float(torch.cuda.memory_allocated()))
        except Exception:
            return
