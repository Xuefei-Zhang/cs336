import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from typer.testing import CliRunner

from aiinfra_e2e.cli import app
from aiinfra_e2e.config import LoRAAdapterConfig, ServeConfig, load_yaml
from aiinfra_e2e.serve.adapters import (
    build_chat_completion_payload,
    build_load_lora_request,
    build_unload_lora_request,
    call_vllm_adapter_endpoint,
)
from aiinfra_e2e.serve.metrics import ServeMetrics
from aiinfra_e2e.serve.vllm_server import (
    build_vllm_command,
    build_vllm_environment,
    get_vllm_smoke_skip_reason,
)

runner = CliRunner()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_load_yaml_parses_vllm_lora_serve_config(tmp_path: Path) -> None:
    config_path = tmp_path / "serve.yaml"
    _ = config_path.write_text(
        (
            "host: 127.0.0.1\n"
            "port: 8012\n"
            "model_id: meta-llama/Meta-Llama-3-8B-Instruct\n"
            "served_model_name: llama-serve\n"
            "metrics_host: 127.0.0.1\n"
            "metrics_port: 9100\n"
            "tensor_parallel_size: 2\n"
            "max_loras: 4\n"
            "max_lora_rank: 32\n"
            "adapters:\n"
            "  - name: support-bot\n"
            "    path: /models/lora/support-bot\n"
        ),
        encoding="utf-8",
    )

    config = load_yaml(config_path, ServeConfig)

    assert config.host == "127.0.0.1"
    assert config.port == 8012
    assert config.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert config.served_model_name == "llama-serve"
    assert config.metrics_port == 9100
    assert config.tensor_parallel_size == 2
    assert config.max_loras == 4
    assert config.max_lora_rank == 32
    assert config.adapters == [
        LoRAAdapterConfig(name="support-bot", path="/models/lora/support-bot")
    ]


def test_build_vllm_command_enables_openai_server_and_runtime_lora() -> None:
    config = ServeConfig(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        host="127.0.0.1",
        port=8012,
        served_model_name="llama-serve",
        tensor_parallel_size=2,
        max_loras=4,
        max_lora_rank=32,
        gpu_memory_utilization=0.9,
    )

    command = build_vllm_command(config)
    env = build_vllm_environment(config)

    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in command
    assert "meta-llama/Meta-Llama-3-8B-Instruct" in command
    assert "--served-model-name" in command
    assert "llama-serve" in command
    assert "--enable-lora" in command
    assert "--max-loras" in command
    assert "4" in command
    assert "--max-lora-rank" in command
    assert "32" in command
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_adapter_request_helpers_build_expected_payloads() -> None:
    adapter = LoRAAdapterConfig(name="support-bot", path="/models/lora/support-bot")

    load_request = build_load_lora_request("http://127.0.0.1:8012", adapter)
    unload_request = build_unload_lora_request("http://127.0.0.1:8012", adapter.name)
    payload = build_chat_completion_payload(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        adapter_name=adapter.name,
        max_tokens=32,
    )

    assert load_request.full_url == "http://127.0.0.1:8012/v1/load_lora_adapter"
    assert load_request.get_method() == "POST"
    load_body = load_request.data
    assert isinstance(load_body, bytes)
    assert json.loads(load_body.decode("utf-8")) == {
        "lora_name": "support-bot",
        "lora_path": "/models/lora/support-bot",
    }
    assert unload_request.full_url == "http://127.0.0.1:8012/v1/unload_lora_adapter"
    unload_body = unload_request.data
    assert isinstance(unload_body, bytes)
    assert json.loads(unload_body.decode("utf-8")) == {"lora_name": "support-bot"}
    assert payload["model"] == "support-bot"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 32


def test_call_vllm_adapter_endpoint_posts_request(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LoRAAdapterConfig(name="support-bot", path="/models/lora/support-bot")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"ok"}'

    def _fake_urlopen(req: Request, timeout: float = 0.0) -> _Response:
        request_body = req.data
        assert isinstance(request_body, bytes)
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(request_body.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("aiinfra_e2e.serve.adapters.request.urlopen", _fake_urlopen)

    response = call_vllm_adapter_endpoint(
        build_load_lora_request("http://127.0.0.1:8012", adapter),
        timeout=3.5,
    )

    assert captured == {
        "url": "http://127.0.0.1:8012/v1/load_lora_adapter",
        "method": "POST",
        "body": {
            "lora_name": "support-bot",
            "lora_path": "/models/lora/support-bot",
        },
        "timeout": 3.5,
    }
    assert response == {"status": "ok"}


def test_metrics_record_adapter_loads_and_failures() -> None:
    registry = CollectorRegistry()
    metrics = ServeMetrics(registry=registry)

    metrics.observe_adapter_load(0.25)
    metrics.record_request_failure()

    exposition = generate_latest(registry).decode("utf-8")

    assert "adapter_load_total 1.0" in exposition
    assert "request_fail_total 1.0" in exposition
    assert "adapter_load_latency_seconds" in exposition


def test_serve_command_runs_wrapper_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "serve.yaml"
    _ = config_path.write_text(
        "model_id: meta-llama/Meta-Llama-3-8B-Instruct\n",
        encoding="utf-8",
    )
    calls: list[ServeConfig] = []

    def _fake_run(config: ServeConfig) -> None:
        calls.append(config)

    monkeypatch.setattr("aiinfra_e2e.cli.run_vllm_server_from_config", _fake_run)

    result = runner.invoke(app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0
    assert calls[0].model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert "Started serve wrapper" in result.stdout


def test_get_vllm_smoke_skip_reason_requires_local_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIINFRA_E2E_VLLM_MODEL", raising=False)
    monkeypatch.setattr("aiinfra_e2e.serve.vllm_server.shutil.which", lambda name: None)

    reason = get_vllm_smoke_skip_reason()

    assert reason is not None
    assert "vLLM" in reason


def test_managed_vllm_server_start_does_not_pipe_child_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiinfra_e2e.serve.vllm_server import ManagedVLLMServer

    class _Process:
        def poll(self) -> None:
            return None

    captured: dict[str, object] = {}
    config = ServeConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        metrics_host="127.0.0.1",
        metrics_port=_free_port(),
    )
    server = ManagedVLLMServer(config)
    monkeypatch.setattr(server.metrics, "start_server", lambda host, port: None)
    monkeypatch.setattr(server.metrics, "update_gpu_memory", lambda: None)
    monkeypatch.setattr(ManagedVLLMServer, "wait_until_ready", lambda self: None)

    def _fake_popen(*args: object, **kwargs: object) -> _Process:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr("aiinfra_e2e.serve.vllm_server.subprocess.Popen", _fake_popen)

    server.start()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("stdout") is not subprocess.PIPE
    assert kwargs.get("stderr") is not subprocess.PIPE


def test_openai_api_smoke() -> None:
    skip_reason = get_vllm_smoke_skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    from aiinfra_e2e.serve.vllm_server import ManagedVLLMServer, openai_request

    port = _free_port()
    metrics_port = _free_port()
    config = ServeConfig(
        host="127.0.0.1",
        port=port,
        model_id=os.environ["AIINFRA_E2E_VLLM_MODEL"],
        served_model_name="smoke-model",
        metrics_host="127.0.0.1",
        metrics_port=metrics_port,
    )
    server = ManagedVLLMServer(config)

    try:
        server.start()
        models = openai_request(config.base_url, "/v1/models")
        assert "data" in models
        served_model_name = config.served_model_name
        assert served_model_name is not None

        response = openai_request(
            config.base_url,
            "/v1/chat/completions",
            payload=build_chat_completion_payload(
                model=served_model_name,
                messages=[{"role": "user", "content": "Say hello in one short sentence."}],
                max_tokens=16,
            ),
        )
        assert response["choices"]
    except HTTPError as exc:  # pragma: no cover - exercised only with optional GPU smoke
        pytest.fail(exc.read().decode("utf-8"))
    finally:
        server.stop()
