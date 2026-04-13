"""Manifest writing helpers for reproducible runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_manifest(
    run_dir: str | Path,
    config_paths: list[str | Path],
    env_info: Mapping[str, JsonValue],
    git_info: Mapping[str, JsonValue],
    *,
    seed: int | None = None,
    dataset_id: str,
    model_id: str | None = None,
) -> Path:
    """Write a stable JSON manifest for a run directory."""

    run_path = Path(run_dir)
    _ = run_path.mkdir(parents=True, exist_ok=True)

    normalized_paths = [Path(path) for path in config_paths]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "config_paths": [str(path) for path in normalized_paths],
        "config_hashes": {str(path): _sha256_file(path) for path in normalized_paths},
        "env": env_info,
        "git": git_info,
    }

    manifest_path = run_path / "manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
