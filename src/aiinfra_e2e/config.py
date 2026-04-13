"""YAML-backed configuration models and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import InitErrorDetails

ModelT = TypeVar("ModelT", bound=BaseModel)


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset_id: str
    split: str = "train"
    text_field: str = "text"
    prompt_field: str | None = None
    response_field: str | None = None
    cache_dir: str | None = None
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    split_key_fields: list[str] | None = None


class TrainConfig(StrictModel):
    model_id: str
    output_dir: str = "artifacts/runs"
    run_name: str | None = None
    seed: int = 42
    max_steps: int = 100
    save_steps: int = 50
    resume_from_checkpoint: bool | str = False
    oom_retries: int = 0
    max_seq_len: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 1
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] | None = "all-linear"


class GoldenPromptConfig(StrictModel):
    name: str
    prompt: str
    constraints: dict[str, int | float | str | list[str]]


class EvalConfig(StrictModel):
    metric: str = "loss"
    batch_size: int = 1
    sample_count: int = 1
    output_dir: str = "artifacts/runs"
    run_name: str | None = None
    golden_prompts: list[GoldenPromptConfig] = []


class LoRAAdapterConfig(StrictModel):
    name: str
    path: str
    load_inplace: bool = False


class ServeConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_id: str | None = None
    served_model_name: str | None = None
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9100
    tensor_parallel_size: int = 1
    max_loras: int = 1
    max_lora_rank: int = 64
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    adapters: list[LoRAAdapterConfig] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ObsConfig(StrictModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "default"


class RunConfig(StrictModel):
    data: DataConfig
    train: TrainConfig | None = None
    eval: EvalConfig | None = None
    serve: ServeConfig | None = None
    obs: ObsConfig | None = None


def load_yaml(path: str | Path, model_cls: type[ModelT]) -> ModelT:
    """Load a UTF-8 YAML file and validate it into a Pydantic model."""

    config_path = Path(path)

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read config file {config_path}: {exc}") from exc

    try:
        loaded = yaml.safe_load(raw_text)
        payload: object = loaded
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config {config_path}: {exc}") from exc

    if payload is None:
        payload = {}

    if not isinstance(payload, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping at the top level, got {type(payload).__name__}."
        )

    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise ValidationError.from_exception_data(
            title=f"Validation failed for config file {config_path}",
            line_errors=cast(list[InitErrorDetails], exc.errors()),
            input_type="python",
            hide_input=False,
        ) from exc
