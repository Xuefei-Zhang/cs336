"""Helpers for OpenAI-compatible Locust load tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow

from aiinfra_e2e.config import LoadTestConfig

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True)
class LoadTestArtifacts:
    run_id: str
    run_dir: Path
    html_report_path: Path
    json_report_path: Path


def resolve_loadtest_run_id(config: LoadTestConfig) -> str:
    if config.run_name:
        return config.run_name
    return f"loadtest-{config.users}u"


def resolve_loadtest_artifacts(config: LoadTestConfig) -> LoadTestArtifacts:
    run_id = resolve_loadtest_run_id(config)
    run_dir = Path(config.output_dir) / run_id
    return LoadTestArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        html_report_path=run_dir / "locust_report.html",
        json_report_path=run_dir / "locust_report.json",
    )


def build_chat_completions_payload(config: LoadTestConfig) -> dict[str, object]:
    model = config.serve.served_model_name or config.serve.model_id
    if model is None:
        raise ValueError("Load test config must define serve.served_model_name or serve.model_id")
    return {
        "model": model,
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
    }


def log_loadtest_reports(*, config: LoadTestConfig, artifacts: LoadTestArtifacts) -> None:
    mlflow.set_tracking_uri(config.obs.tracking_uri)
    _ = mlflow.set_experiment(config.obs.experiment_name)
    with mlflow.start_run(run_name=artifacts.run_id):
        _ = mlflow.log_params(
            {
                "endpoint": CHAT_COMPLETIONS_ENDPOINT,
                "html_report": str(artifacts.html_report_path),
                "json_report": str(artifacts.json_report_path),
                "target_host": config.serve.base_url,
                "users": str(config.users),
                "spawn_rate": str(config.spawn_rate),
                "run_time": config.run_time,
            }
        )
        if artifacts.html_report_path.exists():
            mlflow.log_artifact(str(artifacts.html_report_path))
        if artifacts.json_report_path.exists():
            mlflow.log_artifact(str(artifacts.json_report_path))
