"""Deterministic dataset splitting and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from datasets import Dataset


class SplitEntry(TypedDict):
    dataset: Dataset
    indices: list[int]
    keys: list[str]


SplitResult = dict[str, SplitEntry]


def _stable_serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _record_key(record: dict[str, Any], index: int, key_fields: Sequence[str] | None) -> str:
    payload: dict[str, Any]
    if key_fields:
        payload = {field: record.get(field) for field in key_fields}
    else:
        payload = record
    payload_with_index = {"index": index, "record": payload}
    return hashlib.sha256(_stable_serialize(payload_with_index).encode("utf-8")).hexdigest()


def _hash_values(values: Iterable[int] | Iterable[str]) -> str:
    joined = "\n".join(str(value) for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def deterministic_split(
    dataset: Dataset,
    *,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    key_fields: Sequence[str] | None = None,
) -> SplitResult:
    """Split a dataset deterministically using stable record hashes instead of RNG shuffling."""

    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to at most 1")

    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    keys_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    train_threshold = train_ratio
    val_threshold = train_ratio + val_ratio

    for index in range(len(dataset)):
        record = cast(dict[str, Any], dataset[index])
        key = _record_key(record, index, key_fields)
        fraction = int(key[:16], 16) / float(16**16)
        split_name = "train"
        if fraction >= val_threshold:
            split_name = "test"
        elif fraction >= train_threshold:
            split_name = "val"
        buckets[split_name].append(index)
        keys_by_split[split_name].append(key)

    return {
        split_name: {
            "dataset": dataset.select(indices),
            "indices": indices,
            "keys": keys_by_split[split_name],
        }
        for split_name, indices in buckets.items()
    }


def write_data_manifest(
    output_dir: str | Path,
    *,
    dataset_id: str,
    split_result: SplitResult,
    dataset_fingerprint: str | None = None,
) -> Path:
    """Write a deterministic manifest describing split membership and dataset fingerprint."""

    manifest_dir = Path(output_dir)
    _ = manifest_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {name: len(entry["indices"]) for name, entry in split_result.items()}
    split_hashes = {
        name: _hash_values(sorted(entry["keys"]))
        for name, entry in split_result.items()
    }

    manifest = {
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_id": dataset_id,
        "split_hashes": split_hashes,
        "split_sizes": split_sizes,
    }

    manifest_path = manifest_dir / "data_manifest.json"
    _ = manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
