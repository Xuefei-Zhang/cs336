import json
from pathlib import Path

from datasets import Dataset
import mlflow
from mlflow.tracking import MlflowClient
import torch
from typer.testing import CliRunner

from aiinfra_e2e.cli import app
from aiinfra_e2e.train import sft as sft_module

runner = CliRunner()


def _write_yaml(path: Path, content: str) -> None:
    _ = path.write_text(content, encoding="utf-8")


def test_cpu_smoke_mode_ignores_pytest_env_var(monkeypatch) -> None:
    monkeypatch.delenv("AIINFRA_E2E_CPU_SMOKE", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_train_smoke_cpu.py::unit")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert sft_module._is_cpu_smoke_mode() is False


def test_qlora_is_disabled_by_explicit_cpu_smoke_flag(monkeypatch) -> None:
    monkeypatch.setenv("AIINFRA_E2E_CPU_SMOKE", "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert sft_module._should_enable_qlora() is False


def test_smoke_cpu_script_uses_project_owned_cpu_smoke_flag() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_cpu.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "AIINFRA_E2E_CPU_SMOKE" in script_text
    assert "PYTEST_CURRENT_TEST" not in script_text
    assert "RUN_ROOT=${CPU_SMOKE_RUN_ROOT:-$REPO_ROOT/artifacts/runs}" in script_text
    assert "TRACKING_URI=${CPU_SMOKE_TRACKING_URI:-$REPO_ROOT/mlruns}" in script_text
    assert "ENV_OUT=${CPU_SMOKE_ENV_OUT:-$REPO_ROOT/artifacts/env}" in script_text


def test_build_training_args_honors_configured_dataset_text_field(tmp_path: Path) -> None:
    training_args = sft_module._build_training_args(
        sft_module.TrainConfig(model_id="unit-test-model", output_dir=str(tmp_path)),
        tmp_path / "run",
        dataset_text_field="formatted_prompt",
    )

    assert training_args.dataset_text_field == "formatted_prompt"


def test_train_sft_cpu_smoke_writes_run_outputs_and_mlflow_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AIINFRA_E2E_CPU_SMOKE", "1")

    data_config_path = tmp_path / "data.yaml"
    train_config_path = tmp_path / "train.yaml"
    obs_config_path = tmp_path / "obs.yaml"
    run_root = tmp_path / "artifacts" / "runs"
    tracking_dir = tmp_path / "mlruns"

    _write_yaml(
        data_config_path,
        "\n".join(
            [
                "dataset_id: unit-test-dataset",
                "split: train",
                "train_ratio: 0.8",
                "val_ratio: 0.1",
                "split_key_fields:",
                "  - instruction",
                "  - output",
                "",
            ]
        ),
    )
    _write_yaml(
        train_config_path,
        "\n".join(
            [
                "model_id: Qwen/Qwen2.5-7B-Instruct",
                f"output_dir: {run_root}",
                "run_name: cpu-smoke",
                "seed: 13",
                "max_steps: 1",
                "per_device_train_batch_size: 1",
                "gradient_accumulation_steps: 1",
                "learning_rate: 0.0005",
                "logging_steps: 1",
                "lora_r: 4",
                "lora_alpha: 8",
                "lora_dropout: 0.0",
                "",
            ]
        ),
    )
    _write_yaml(
        obs_config_path,
        "\n".join(
            [
                f"tracking_uri: {tracking_dir}",
                "experiment_name: cpu-smoke-tests",
                "",
            ]
        ),
    )

    dataset = Dataset.from_list(
        [
            {"instruction": "Say hi", "input": "", "output": "hi"},
            {"instruction": "Say bye", "input": "", "output": "bye"},
            {"instruction": "Count", "input": "1 2", "output": "3"},
        ]
    )
    monkeypatch.setattr(
        "aiinfra_e2e.train.sft.load_hf_dataset",
        lambda config: dataset,
    )

    result = runner.invoke(
        app,
        [
            "train",
            "sft",
            "--data-config",
            str(data_config_path),
            "--train-config",
            str(train_config_path),
            "--obs-config",
            str(obs_config_path),
        ],
    )

    assert result.exit_code == 0, result.stdout

    run_dir = run_root / "cpu-smoke"
    manifest_path = run_dir / "manifest.json"
    adapter_dir = run_dir / "lora_adapter"
    log_path = run_dir / "train.log"

    assert run_dir.exists()
    assert manifest_path.exists()
    assert adapter_dir.exists()
    assert log_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "unit-test-dataset"
    assert manifest["seed"] == 13
    assert manifest["model_id"] == "sshleifer/tiny-gpt2"

    mlflow.set_tracking_uri(str(tracking_dir))
    client = MlflowClient(tracking_uri=str(tracking_dir))
    experiment = client.get_experiment_by_name("cpu-smoke-tests")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert "final_loss" in run.data.metrics

    artifact_paths = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert "manifest.json" in artifact_paths
    assert "train.yaml" in artifact_paths
