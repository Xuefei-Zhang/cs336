import importlib
import json
from pathlib import Path
from typing import Any

import datasets
import pytest
from requests import exceptions as requests_exceptions

from aiinfra_e2e.config import DataConfig


def _data_api() -> tuple[Any, Any, Any]:
    split_module = importlib.import_module("aiinfra_e2e.data.split")
    hf_sync_module = importlib.import_module("aiinfra_e2e.data.hf_sync")
    return (
        getattr(split_module, "deterministic_split"),
        getattr(hf_sync_module, "load_hf_dataset"),
        getattr(split_module, "write_data_manifest"),
    )


@pytest.fixture
def sample_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "instruction": ["translate hello", "summarize story", "classify sentiment"],
            "output": ["你好", "short summary", "positive"],
        }
    )


def test_deterministic_split_is_stable_for_identical_input(sample_dataset: datasets.Dataset) -> None:
    deterministic_split, _, _ = _data_api()

    first_split = deterministic_split(
        sample_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        key_fields=("instruction", "output"),
    )
    second_split = deterministic_split(
        sample_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        key_fields=("instruction", "output"),
    )

    assert first_split["train"]["indices"] == second_split["train"]["indices"]
    assert first_split["val"]["indices"] == second_split["val"]["indices"]
    assert first_split["test"]["indices"] == second_split["test"]["indices"]


def test_write_data_manifest_hash_changes_when_split_membership_changes(tmp_path: Path) -> None:
    original_dataset = datasets.Dataset.from_dict(
        {
            "instruction": ["translate hello", "summarize story", "classify sentiment"],
            "output": ["你好", "short summary", "positive"],
        }
    )
    changed_dataset = datasets.Dataset.from_dict(
        {
            "instruction": ["translate hello", "summarize story updated", "classify sentiment"],
            "output": ["你好", "short summary", "positive"],
        }
    )

    deterministic_split, _, write_data_manifest = _data_api()

    original_split = deterministic_split(
        original_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        key_fields=("instruction", "output"),
    )
    changed_split = deterministic_split(
        changed_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        key_fields=("instruction", "output"),
    )

    original_manifest = write_data_manifest(
        tmp_path / "original",
        dataset_id="local/test-dataset",
        split_result=original_split,
        dataset_fingerprint=getattr(original_dataset, "_fingerprint", None),
    )
    changed_manifest = write_data_manifest(
        tmp_path / "changed",
        dataset_id="local/test-dataset",
        split_result=changed_split,
        dataset_fingerprint=getattr(changed_dataset, "_fingerprint", None),
    )

    original_payload = json.loads(original_manifest.read_text(encoding="utf-8"))
    changed_payload = json.loads(changed_manifest.read_text(encoding="utf-8"))

    assert original_payload["split_hashes"] != changed_payload["split_hashes"]


def test_load_hf_dataset_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_dataset = datasets.Dataset.from_dict({"instruction": ["hello"], "output": ["world"]})
    calls = {"count": 0}

    def fake_load_dataset(*args: object, **kwargs: object) -> datasets.Dataset:
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests_exceptions.ConnectionError("temporary network failure")
        return expected_dataset

    sleep_calls: list[float] = []

    monkeypatch.setattr("aiinfra_e2e.data.hf_sync.load_dataset", fake_load_dataset)
    monkeypatch.setattr("aiinfra_e2e.data.hf_sync.time.sleep", sleep_calls.append)

    _, load_hf_dataset, _ = _data_api()

    config = DataConfig(dataset_id="local/test-dataset", split="train", cache_dir="/tmp/hf-cache")

    dataset = load_hf_dataset(
        config,
        max_retries=5,
        base_delay_seconds=0.01,
        jitter_seconds=0.0,
    )

    assert dataset == expected_dataset
    assert calls["count"] == 3
    assert sleep_calls == [0.01, 0.02]
