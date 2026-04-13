import json
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from aiinfra_e2e.cli import app

runner = CliRunner()


def test_root_help_lists_required_subcommands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("env", "data", "train", "eval", "serve", "obs", "loadtest"):
        assert command in result.stdout


def test_env_report_help_renders() -> None:
    result = runner.invoke(app, ["env", "report", "--help"])

    assert result.exit_code == 0
    assert "--out" in result.stdout


def test_data_command_rejects_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    result = runner.invoke(app, ["data", "--config", str(missing_path)])

    assert result.exit_code != 0
    assert f"Config file not found: {missing_path}" in result.stdout


def test_data_command_accepts_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    _ = config_path.write_text("dataset_id: demo-dataset\n", encoding="utf-8")

    result = runner.invoke(app, ["data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Loaded config" in result.stdout


def test_env_report_writes_env_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"

    result = runner.invoke(app, ["env", "report", "--out", str(run_dir)])

    assert result.exit_code == 0
    env_path = run_dir / "env.json"
    assert env_path.exists()
    payload = cast(dict[str, object], json.loads(env_path.read_text(encoding="utf-8")))
    assert payload["python_version"]
    assert payload["platform"]
    assert payload["timestamp"]
    assert "torch_version" in payload
    assert "cuda_available" in payload
