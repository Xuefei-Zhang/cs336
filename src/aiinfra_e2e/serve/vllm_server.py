"""Subprocess wrapper for the vLLM OpenAI-compatible API server."""

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast
from urllib import request
from urllib.error import HTTPError, URLError

from aiinfra_e2e.config import ServeConfig
from aiinfra_e2e.serve.metrics import ServeMetrics


def build_vllm_environment(config: ServeConfig) -> dict[str, str]:
    del config
    env = dict(os.environ)
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    return env


def build_vllm_command(config: ServeConfig) -> list[str]:
    if not config.model_id:
        raise ValueError("Serve config must set model_id to start vLLM.")

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--model",
        config.model_id,
        "--enable-lora",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-loras",
        str(config.max_loras),
        "--max-lora-rank",
        str(config.max_lora_rank),
    ]
    if config.served_model_name:
        command.extend(["--served-model-name", config.served_model_name])
    if config.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    if config.dtype:
        command.extend(["--dtype", config.dtype])
    if config.extra_args:
        command.extend(config.extra_args)
    return command


def _json_request(
    url: str,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
) -> request.Request:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    return request.Request(url, data=body, headers=headers, method=method)


def openai_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 5.0,
) -> dict[str, Any]:
    http_request = _json_request(f"{base_url.rstrip('/')}{path}", payload=payload, method=method)
    with request.urlopen(http_request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Expected JSON object from {path}, got {type(decoded).__name__}")
    return cast(dict[str, Any], dict(decoded))


def get_vllm_smoke_skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return "vLLM smoke test requires torch to be importable."

    if not torch.cuda.is_available():
        return "vLLM smoke test requires CUDA/GPU availability."
    if importlib.util.find_spec("vllm") is None:
        return "vLLM smoke test requires the vLLM package to be installed."
    if shutil.which("nvidia-smi") is None:
        return "vLLM smoke test requires an NVIDIA runtime."
    if not os.environ.get("AIINFRA_E2E_VLLM_MODEL"):
        return "vLLM smoke test requires AIINFRA_E2E_VLLM_MODEL to avoid heavyweight default downloads."
    return None


@dataclass
class ManagedVLLMServer:
    config: ServeConfig
    metrics: ServeMetrics | None = None
    process: subprocess.Popen[str] | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = ServeMetrics()

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("vLLM server is already running.")
        assert self.metrics is not None
        self.metrics.start_server(self.config.metrics_host, self.config.metrics_port)
        self.metrics.update_gpu_memory()
        self.process = subprocess.Popen(
            build_vllm_command(self.config),
            env=build_vllm_environment(self.config),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.wait_until_ready()

    def wait_until_ready(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                stderr = ""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise RuntimeError(f"vLLM server exited before becoming ready: {stderr}")
            try:
                _ = openai_request(self.config.base_url, "/v1/models", timeout=2.0)
                return
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(f"Timed out waiting for vLLM server readiness: {last_error}")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        with suppress(subprocess.TimeoutExpired):
            _ = self.process.wait(timeout=10)
        if self.process.poll() is None:
            self.process.kill()
            _ = self.process.wait(timeout=5)
        self.process = None


def run_vllm_server_from_config(config: ServeConfig) -> None:
    server = ManagedVLLMServer(config)
    try:
        server.start()
        while True:
            time.sleep(1.0)
            assert server.metrics is not None
            server.metrics.update_gpu_memory()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
