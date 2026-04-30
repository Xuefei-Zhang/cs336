"""Helpers for lifecycle trace runs around the repo-native serve wrapper."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError

import yaml

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.vllm_server import openai_request


def _open_url(url: str, timeout: float):
    http_request = request.Request(url, method="GET")
    host = parse.urlparse(url).hostname
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        opener = request.build_opener(request.ProxyHandler({}))
        return opener.open(http_request, timeout=timeout)
    return request.urlopen(http_request, timeout=timeout)


def _port_is_in_use(host: str, port: int) -> bool:
    probe_host = host
    if probe_host in {"0.0.0.0", "::", ""}:
        probe_host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((probe_host, port))
        except OSError:
            return True
    return False


def _pick_free_local_port(excluded: set[int]) -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_artifact(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def write_env_snapshot(path: str | Path, env: dict[str, str] | None = None) -> None:
    environment = dict(os.environ if env is None else env)
    lines = [f"{key}={environment[key]}" for key in sorted(environment)]
    write_text_artifact(path, "\n".join(lines) + "\n")


def render_effective_serve_config(
    source_path: str | Path,
    output_path: str | Path,
    *,
    model_id: str | None = None,
    served_model_name: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    startup_timeout_seconds: float | None = None,
    dtype: str | None = None,
    extra_args: list[str] | None = None,
) -> ServeConfig:
    config = load_yaml(source_path, ServeConfig)
    payload = config.model_dump(mode="python")

    selected_port = config.port
    if _port_is_in_use(config.host, config.port):
        selected_port = _pick_free_local_port(set())

    selected_metrics_port = config.metrics_port
    if selected_metrics_port == selected_port or _port_is_in_use(config.metrics_host, config.metrics_port):
        selected_metrics_port = _pick_free_local_port({selected_port})

    payload["port"] = selected_port
    payload["metrics_port"] = selected_metrics_port

    if model_id is not None:
        payload["model_id"] = model_id
    if served_model_name is not None:
        payload["served_model_name"] = served_model_name
    if tensor_parallel_size is not None:
        payload["tensor_parallel_size"] = tensor_parallel_size
    if gpu_memory_utilization is not None:
        payload["gpu_memory_utilization"] = gpu_memory_utilization
    if startup_timeout_seconds is not None:
        payload["startup_timeout_seconds"] = startup_timeout_seconds
    if dtype is not None:
        payload["dtype"] = dtype
    if extra_args is not None:
        payload["extra_args"] = extra_args

    output = Path(output_path)
    _write_yaml(output, payload)
    return load_yaml(output, ServeConfig)


def probe_models(base_url: str, output_path: str | Path, timeout: float = 5.0) -> dict[str, Any]:
    payload = openai_request(base_url, "/v1/models", timeout=timeout)
    write_json_artifact(output_path, payload)
    return payload


def send_smoke_chat_completion(
    base_url: str,
    *,
    model: str,
    prompt: str,
    output_path: str | Path,
    max_tokens: int = 32,
    timeout: float = 30.0,
) -> dict[str, Any]:
    payload = openai_request(
        base_url,
        "/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        method="POST",
        timeout=timeout,
    )
    write_json_artifact(output_path, payload)
    return payload


def probe_health(url: str, output_path: str | Path, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with _open_url(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = int(response.status)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except URLError as exc:
        body = str(exc.reason)
        status = 0

    payload = {"url": url, "status": status, "body": body}
    write_json_artifact(output_path, payload)
    return payload


def scrape_metrics(url: str, output_path: str | Path, timeout: float = 5.0) -> str:
    with _open_url(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    write_text_artifact(output_path, body)
    return body
