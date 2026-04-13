"""Locust user that targets an OpenAI-compatible chat completions endpoint."""

# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

Environment = Any


class _EventHook:
    def add_listener(self, func):
        return func


class _Events:
    init_command_line_parser = _EventHook()
    quitting = _EventHook()


class _HttpUser:
    pass


def _between(*_: float):
    return None


def _task(func):
    return func


try:
    _locust = import_module("locust")
    HttpUser = _locust.HttpUser
    between = _locust.between
    events = _locust.events
    task = _locust.task
except ImportError:  # pragma: no cover - exercised only when Locust is unavailable
    HttpUser = _HttpUser
    between = _between
    events = _Events()
    task = _task


def _build_payload(environment: Environment) -> dict[str, object]:
    options = environment.parsed_options
    return {
        "model": options.model,
        "messages": [{"role": "user", "content": options.prompt}],
        "max_tokens": options.max_tokens,
    }


@events.init_command_line_parser.add_listener
def _add_custom_arguments(parser) -> None:
    parser.add_argument("--model", type=str, required=True, help="OpenAI model name to request.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Say hello in one short sentence.",
        help="User prompt sent to /v1/chat/completions.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="max_tokens value for the OpenAI chat completion payload.",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=os.environ.get("LOADTEST_JSON_REPORT"),
        help="Optional path where a JSON Locust summary report will be written.",
    )


class OpenAIChatUser(HttpUser):
    wait_time = between(0.5, 1.5)
    host = os.environ.get("LOADTEST_HOST", "http://127.0.0.1:8000")

    @task
    def chat_completion(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json=_build_payload(self.environment),
            name="POST /v1/chat/completions",
        )
        if response.status_code >= 400:
            response.failure(f"HTTP {response.status_code}: {response.text}")
            return
        try:
            payload = response.json()
        except json.JSONDecodeError:
            response.failure("Response was not valid JSON")
            return
        if not payload.get("choices"):
            response.failure("Response did not include choices")


@events.quitting.add_listener
def _write_json_report(environment: Environment, **_: object) -> None:
    options = environment.parsed_options
    json_report = getattr(options, "json_report", None)
    if not json_report:
        return

    total = environment.stats.total
    payload = {
        "target_host": environment.host,
        "endpoint": "/v1/chat/completions",
        "requests": total.num_requests,
        "failures": total.num_failures,
        "median_response_time": total.median_response_time,
        "average_response_time": total.avg_response_time,
        "min_response_time": total.min_response_time,
        "max_response_time": total.max_response_time,
        "requests_per_second": total.current_rps,
        "fail_ratio": total.fail_ratio,
    }
    report_path = Path(json_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
