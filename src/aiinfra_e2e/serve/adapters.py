"""Helpers for dynamic LoRA adapter lifecycle requests."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib import request

from aiinfra_e2e.config import LoRAAdapterConfig


def _post_request(base_url: str, path: str, payload: dict[str, Any]) -> request.Request:
    body = json.dumps(payload).encode("utf-8")
    return request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def build_load_lora_request(base_url: str, adapter: LoRAAdapterConfig) -> request.Request:
    payload: dict[str, Any] = {
        "lora_name": adapter.name,
        "lora_path": adapter.path,
    }
    if adapter.load_inplace:
        payload["load_inplace"] = True
    return _post_request(base_url, "/v1/load_lora_adapter", payload)


def build_unload_lora_request(base_url: str, adapter_name: str) -> request.Request:
    return _post_request(base_url, "/v1/unload_lora_adapter", {"lora_name": adapter_name})


def build_chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, object]],
    adapter_name: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": adapter_name or model,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def call_vllm_adapter_endpoint(
    http_request: request.Request,
    *,
    timeout: float = 10.0,
) -> dict[str, Any] | str:
    with request.urlopen(http_request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            return cast(dict[str, Any], decoded)
        return {"value": decoded}
    except json.JSONDecodeError:
        return raw
