import json
from pathlib import Path
from typing import cast

from aiinfra_e2e.manifest import write_run_manifest


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def test_write_run_manifest_includes_required_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "run-001"
    config_path = tmp_path / "data.yaml"
    _ = config_path.write_text("dataset_id: hfl/alpaca_zh_51k\nsplit: train\n", encoding="utf-8")

    manifest_path = write_run_manifest(
        run_dir=run_dir,
        config_paths=[config_path],
        env_info={"python": "3.11.8"},
        git_info={"sha": "abc123"},
        seed=7,
        dataset_id="hfl/alpaca_zh_51k",
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )

    payload = _read_manifest(manifest_path)

    assert manifest_path == run_dir / "manifest.json"
    assert payload["created_at"]
    assert payload["seed"] == 7
    assert payload["dataset_id"] == "hfl/alpaca_zh_51k"
    assert payload["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["config_paths"] == [str(config_path)]
    config_hashes = cast(dict[str, str], payload["config_hashes"])
    assert config_hashes[str(config_path)]
    assert payload["env"] == {"python": "3.11.8"}
    assert payload["git"] == {"sha": "abc123"}


def test_write_run_manifest_hash_changes_when_config_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "run-001"
    config_path = tmp_path / "data.yaml"
    _ = config_path.write_text("dataset_id: dataset-a\nsplit: train\n", encoding="utf-8")

    first_manifest_path = write_run_manifest(
        run_dir=run_dir,
        config_paths=[config_path],
        env_info={},
        git_info={},
        dataset_id="dataset-a",
    )
    first_payload = _read_manifest(first_manifest_path)

    _ = config_path.write_text("dataset_id: dataset-b\nsplit: train\n", encoding="utf-8")

    second_manifest_path = write_run_manifest(
        run_dir=run_dir,
        config_paths=[config_path],
        env_info={},
        git_info={},
        dataset_id="dataset-b",
    )
    second_payload = _read_manifest(second_manifest_path)

    first_hashes = cast(dict[str, str], first_payload["config_hashes"])
    second_hashes = cast(dict[str, str], second_payload["config_hashes"])

    assert first_manifest_path == second_manifest_path
    assert first_hashes[str(config_path)] != second_hashes[str(config_path)]
