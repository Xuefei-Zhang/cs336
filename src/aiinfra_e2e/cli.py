"""CLI entrypoint for aiinfra_e2e."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, ClassVar

import typer
from pydantic import BaseModel, ConfigDict, ValidationError

from aiinfra_e2e.config import (
    DataConfig,
    EvalConfig,
    ObsConfig,
    ServeConfig,
    TrainConfig,
    load_yaml,
)
from aiinfra_e2e.train.sft import run_sft_from_paths

app = typer.Typer(help="AIInfra E2E command line interface.")
env_app = typer.Typer(help="Environment commands.")
train_app = typer.Typer(help="Training commands.")


class _StubConfig(BaseModel):
    """Permissive placeholder config for commands without a concrete schema yet."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Path to a YAML config file."),
]
OutOption = Annotated[
    Path,
    typer.Option("--out", help="Directory where the environment report will be written."),
]


def _load_config(config_path: Path, model_cls: type[BaseModel]) -> BaseModel:
    """Validate a YAML config path and load it into the requested model."""

    if not config_path.exists() or not config_path.is_file():
        typer.echo(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    try:
        return load_yaml(config_path, model_cls)
    except (ValidationError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc


def _handle_stub_command(
    command_name: str, config_path: Path | None, model_cls: type[BaseModel]
) -> None:
    """Load config when provided and otherwise keep command behavior stubbed."""

    if config_path is None:
        typer.echo(f"{command_name} command stub. Provide --config to validate a YAML config.")
        return

    _ = _load_config(config_path, model_cls)
    typer.echo(f"Loaded config from {config_path}")


def _collect_env_info() -> dict[str, object]:
    """Collect a minimal runtime environment snapshot."""

    env_info: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": None,
    }

    try:
        import torch
    except ImportError:
        return env_info

    env_info["torch_version"] = torch.__version__
    env_info["cuda_available"] = torch.cuda.is_available()
    return env_info


@app.callback()
def main_callback() -> None:
    """Run the root CLI callback."""


@env_app.callback()
def env_callback(
    ctx: typer.Context,
    config: ConfigOption = None,
) -> None:
    """Validate optional environment config for env subcommands."""

    ctx.obj = {"config": None, "config_path": config}
    if config is not None:
        ctx.obj["config"] = _load_config(config, _StubConfig)


@env_app.command("report")
def env_report(out: OutOption) -> None:
    """Write a minimal environment report into the requested run directory."""

    out.mkdir(parents=True, exist_ok=True)
    env_path = out / "env.json"
    payload = _collect_env_info()
    _ = env_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"Wrote environment report to {env_path}")


@app.command("data")
def data_command(config: ConfigOption = None) -> None:
    """Validate data config input for future data workflows."""

    _handle_stub_command("Data", config, DataConfig)


@train_app.callback(invoke_without_command=True)
def train_callback(
    ctx: typer.Context,
    config: ConfigOption = None,
) -> None:
    """Keep top-level train config validation available for backward compatibility."""

    if ctx.invoked_subcommand is not None:
        return
    _handle_stub_command("Train", config, TrainConfig)


@train_app.command("sft")
def train_sft_command(
    data_config: Annotated[Path, typer.Option("--data-config", help="Path to data YAML config.")],
    train_config: Annotated[
        Path, typer.Option("--train-config", help="Path to train YAML config.")
    ],
    obs_config: Annotated[
        Path, typer.Option("--obs-config", help="Path to observability YAML config.")
    ],
) -> None:
    """Run TRL SFT fine-tuning with PEFT LoRA and optional QLoRA."""

    for config_path in (data_config, train_config, obs_config):
        if not config_path.exists() or not config_path.is_file():
            typer.echo(f"Config file not found: {config_path}")
            raise typer.Exit(code=1)

    run_dir = run_sft_from_paths(
        data_config_path=data_config,
        train_config_path=train_config,
        obs_config_path=obs_config,
    )
    typer.echo(f"Finished SFT run in {run_dir}")


@app.command("eval")
def eval_command(config: ConfigOption = None) -> None:
    """Validate eval config input for future evaluation workflows."""

    _handle_stub_command("Eval", config, EvalConfig)


@app.command("serve")
def serve_command(config: ConfigOption = None) -> None:
    """Validate serve config input for future serving workflows."""

    _handle_stub_command("Serve", config, ServeConfig)


@app.command("obs")
def obs_command(config: ConfigOption = None) -> None:
    """Validate observability config input for future obs workflows."""

    _handle_stub_command("Obs", config, ObsConfig)


@app.command("loadtest")
def loadtest_command(config: ConfigOption = None) -> None:
    """Validate loadtest config input for future load testing workflows."""

    _handle_stub_command("Loadtest", config, _StubConfig)


app.add_typer(env_app, name="env")
app.add_typer(train_app, name="train")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
