from pathlib import Path

from aiinfra_e2e.config import LoadTestConfig, load_yaml
from aiinfra_e2e.loadtest import (
    build_chat_completions_payload,
    log_loadtest_reports,
    resolve_loadtest_artifacts,
)


def test_load_yaml_parses_loadtest_config(tmp_path: Path) -> None:
    config_path = tmp_path / "loadtest.yaml"
    _ = config_path.write_text(
        (
            "run_name: smoke-loadtest\n"
            "output_dir: artifacts/runs\n"
            "users: 4\n"
            "spawn_rate: 2\n"
            "run_time: 30s\n"
            "prompt: Say hello in one short sentence.\n"
            "max_tokens: 48\n"
            "serve:\n"
            "  host: 127.0.0.1\n"
            "  port: 8012\n"
            "  served_model_name: llama-serve\n"
            "obs:\n"
            "  tracking_uri: mlruns\n"
            "  experiment_name: loadtest\n"
        ),
        encoding="utf-8",
    )

    config = load_yaml(config_path, LoadTestConfig)

    assert config.run_name == "smoke-loadtest"
    assert config.output_dir == "artifacts/runs"
    assert config.users == 4
    assert config.spawn_rate == 2.0
    assert config.run_time == "30s"
    assert config.serve.base_url == "http://127.0.0.1:8012"
    assert config.serve.served_model_name == "llama-serve"
    assert config.obs.tracking_uri == "mlruns"
    assert config.obs.experiment_name == "loadtest"
    assert build_chat_completions_payload(config) == {
        "model": "llama-serve",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 48,
    }


def test_serve_config_base_url_normalizes_wildcard_bind_hosts() -> None:
    from aiinfra_e2e.config import ServeConfig

    assert ServeConfig(host="0.0.0.0", port=1234, model_id="x").base_url == "http://127.0.0.1:1234"
    assert ServeConfig(host="::", port=1234, model_id="x").base_url == "http://127.0.0.1:1234"
    assert (
        ServeConfig(host="localhost", port=1234, model_id="x").base_url == "http://localhost:1234"
    )


def test_report_paths_live_under_run_dir_and_are_logged_to_mlflow(
    tmp_path: Path, monkeypatch
) -> None:
    config = LoadTestConfig.model_validate(
        {
            "run_name": "loadtest-demo",
            "output_dir": str(tmp_path / "artifacts" / "runs"),
            "prompt": "Ping",
            "serve": {"host": "127.0.0.1", "port": 8000, "served_model_name": "demo-model"},
            "obs": {"tracking_uri": str(tmp_path / "mlruns"), "experiment_name": "loadtest"},
        }
    )
    artifacts = resolve_loadtest_artifacts(config)
    calls: list[tuple[str, object]] = []

    def _set_tracking_uri(uri: str) -> None:
        calls.append(("tracking_uri", uri))

    def _set_experiment(name: str) -> str:
        calls.append(("experiment", name))
        return name

    class _Run:
        def __enter__(self):
            calls.append(("start_run", artifacts.run_id))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _start_run(*, run_name: str):
        calls.append(("run_name", run_name))
        return _Run()

    def _log_params(params: dict[str, str]) -> None:
        calls.append(("params", params))

    def _log_artifact(path: str) -> None:
        calls.append(("artifact", path))

    monkeypatch.setattr("aiinfra_e2e.loadtest.mlflow.set_tracking_uri", _set_tracking_uri)
    monkeypatch.setattr("aiinfra_e2e.loadtest.mlflow.set_experiment", _set_experiment)
    monkeypatch.setattr("aiinfra_e2e.loadtest.mlflow.start_run", _start_run)
    monkeypatch.setattr("aiinfra_e2e.loadtest.mlflow.log_params", _log_params)
    monkeypatch.setattr("aiinfra_e2e.loadtest.mlflow.log_artifact", _log_artifact)

    _ = artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    _ = artifacts.html_report_path.write_text("<html></html>\n", encoding="utf-8")
    summary = {
        "run_id": artifacts.run_id,
        "target_host": config.serve.base_url,
        "endpoint": "/v1/chat/completions",
        "html_report": str(artifacts.html_report_path),
    }
    _ = artifacts.json_report_path.write_text(
        __import__("json").dumps(summary) + "\n", encoding="utf-8"
    )

    log_loadtest_reports(config=config, artifacts=artifacts)

    assert artifacts.run_id == "loadtest-demo"
    assert artifacts.run_dir == tmp_path / "artifacts" / "runs" / "loadtest-demo"
    assert artifacts.html_report_path == artifacts.run_dir / "locust_report.html"
    assert artifacts.json_report_path == artifacts.run_dir / "locust_report.json"
    assert artifacts.html_report_path.exists()
    assert artifacts.json_report_path.exists()
    assert calls == [
        ("tracking_uri", str(tmp_path / "mlruns")),
        ("experiment", "loadtest"),
        ("run_name", "loadtest-demo"),
        ("start_run", "loadtest-demo"),
        (
            "params",
            {
                "endpoint": "/v1/chat/completions",
                "html_report": str(artifacts.html_report_path),
                "json_report": str(artifacts.json_report_path),
                "target_host": config.serve.base_url,
                "users": str(config.users),
                "spawn_rate": str(config.spawn_rate),
                "run_time": config.run_time,
            },
        ),
        ("artifact", str(artifacts.html_report_path)),
        ("artifact", str(artifacts.json_report_path)),
    ]
