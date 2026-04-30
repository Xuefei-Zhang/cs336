from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.trace_run import probe_models, render_effective_serve_config, scrape_metrics


def _bind_listener() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, int(sock.getsockname()[1])


def test_render_effective_serve_config_picks_free_ports_and_writes_yaml(tmp_path: Path) -> None:
    port_sock, busy_port = _bind_listener()
    metrics_sock, busy_metrics_port = _bind_listener()
    source_config = tmp_path / "serve.yaml"
    output_config = tmp_path / "serve.effective.yaml"
    _ = source_config.write_text(
        (
            f"host: 0.0.0.0\n"
            f"port: {busy_port}\n"
            f"model_id: /models/qwen\n"
            f"served_model_name: qwen-trace\n"
            f"metrics_host: 127.0.0.1\n"
            f"metrics_port: {busy_metrics_port}\n"
            f"tensor_parallel_size: 2\n"
            f"gpu_memory_utilization: 0.9\n"
            "extra_args:\n"
            "  - --max-model-len\n"
            '  - "131072"\n'
        ),
        encoding="utf-8",
    )

    try:
        config = render_effective_serve_config(source_config, output_config)
    finally:
        port_sock.close()
        metrics_sock.close()

    written = load_yaml(output_config, ServeConfig)

    assert output_config.exists()
    assert config.port != busy_port
    assert config.metrics_port != busy_metrics_port
    assert config.metrics_port != config.port
    assert config.model_id == "/models/qwen"
    assert config.tensor_parallel_size == 2
    assert written == config


def test_probe_models_writes_models_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "models.json"
    expected_payload = {"data": [{"id": "qwen-trace"}]}

    def _fake_openai_request(base_url: str, path: str, *, timeout: float = 5.0):
        assert base_url == "http://127.0.0.1:8001"
        assert path == "/v1/models"
        assert timeout == 7.5
        return expected_payload

    monkeypatch.setattr("aiinfra_e2e.serve.trace_run.openai_request", _fake_openai_request)

    payload = probe_models("http://127.0.0.1:8001", output_path, timeout=7.5)

    assert payload == expected_payload
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected_payload


def test_scrape_metrics_writes_prometheus_text(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.prom"
    expected_text = "# HELP wrapper_gpu_memory_bytes demo\nwrapper_gpu_memory_bytes 0.0\n"

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            encoded = expected_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = scrape_metrics(
            f"http://127.0.0.1:{server.server_address[1]}/metrics",
            output_path,
            timeout=5.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert body == expected_text
    assert output_path.read_text(encoding="utf-8") == expected_text
