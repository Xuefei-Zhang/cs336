"""TRL SFT training entrypoint with PEFT LoRA and optional QLoRA."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
import huggingface_hub.constants as hf_constants
import mlflow
from peft import prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
import transformers.utils.hub as transformers_hub

from aiinfra_e2e.config import DataConfig, ObsConfig, TrainConfig, load_yaml
from aiinfra_e2e.data.hf_sync import load_hf_dataset
from aiinfra_e2e.data.preprocess import preprocess_record
from aiinfra_e2e.data.split import deterministic_split
from aiinfra_e2e.logging import configure_logging
from aiinfra_e2e.manifest import JsonValue, write_run_manifest
from aiinfra_e2e.obs.mlflow import start_run as start_mlflow_run
from aiinfra_e2e.train.accelerate import get_accelerate_runtime, wait_for_everyone
from aiinfra_e2e.train.checkpointing import (
    next_oom_retry_config,
    resolve_resume_checkpoint,
)
from aiinfra_e2e.train.qlora import build_lora_config, build_quantization_config
from aiinfra_e2e.train.trl_compat import SFTConfig, SFTTrainer

CPU_SMOKE_MODEL_ID = "sshleifer/tiny-gpt2"
CPU_SMOKE_ENV_VAR = "AIINFRA_E2E_CPU_SMOKE"


class SupervisedDataCollator:
    """Pad pretokenized SFT records while preserving assistant-only labels."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        raw_pad_token_id = tokenizer.pad_token_id
        raw_eos_token_id = tokenizer.eos_token_id
        if isinstance(raw_pad_token_id, int):
            self.pad_token_id: int = raw_pad_token_id
        elif isinstance(raw_eos_token_id, int):
            self.pad_token_id = raw_eos_token_id
        else:
            self.pad_token_id = 0

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            feature_input_ids = feature["input_ids"]
            feature_labels = feature["labels"]
            pad_width = max_length - len(feature_input_ids)
            input_ids.append(feature_input_ids + [self.pad_token_id] * pad_width)
            attention_mask.append([1] * len(feature_input_ids) + [0] * pad_width)
            labels.append(feature_labels + [-100] * pad_width)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _is_cpu_smoke_mode() -> bool:
    return (not torch.cuda.is_available()) or (os.environ.get(CPU_SMOKE_ENV_VAR) == "1")


def _resolve_model_id(train_config: TrainConfig) -> str:
    if _is_cpu_smoke_mode():
        return CPU_SMOKE_MODEL_ID
    return train_config.model_id


def _resolve_run_id(train_config: TrainConfig) -> str:
    if train_config.run_name:
        return train_config.run_name
    return f"seed-{train_config.seed}"


def _should_enable_qlora() -> bool:
    return torch.cuda.is_available() and (not _is_cpu_smoke_mode())


def _resolve_model_cache_dir(*, data_config: DataConfig, train_config: TrainConfig) -> Path:
    if data_config.cache_dir is not None:
        cache_dir = Path(data_config.cache_dir)
    else:
        cache_dir = Path(train_config.output_dir).parent / "hf-cache"
    _ = cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _set_module_attr(module: object, name: str, value: object) -> None:
    if hasattr(module, name):
        setattr(module, name, value)


@contextmanager
def _override_hf_cache_env(cache_dir: Path):
    """Force all Hugging Face download paths into a writable project-local cache."""

    env_keys = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
    previous = {key: os.environ.get(key) for key in env_keys}
    previous_constants = {
        "hf_home": hf_constants.HF_HOME,
        "hf_hub_cache": hf_constants.HF_HUB_CACHE,
        "transformers_hf_home": transformers_hub.constants.HF_HOME,
        "transformers_hf_hub_cache": transformers_hub.constants.HF_HUB_CACHE,
        "transformers_cache": getattr(transformers_hub, "TRANSFORMERS_CACHE", None),
        "hf_modules_cache": getattr(transformers_hub, "HF_MODULES_CACHE", None),
    }
    hub_cache = cache_dir / "hub"
    transformers_cache = cache_dir / "transformers"
    modules_cache = cache_dir / "modules"
    _ = hub_cache.mkdir(parents=True, exist_ok=True)
    _ = transformers_cache.mkdir(parents=True, exist_ok=True)
    _ = modules_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    hf_constants.HF_HOME = str(cache_dir)
    hf_constants.HF_HUB_CACHE = str(hub_cache)
    transformers_hub.constants.HF_HOME = str(cache_dir)
    transformers_hub.constants.HF_HUB_CACHE = str(hub_cache)
    _set_module_attr(transformers_hub, "TRANSFORMERS_CACHE", str(transformers_cache))
    _set_module_attr(transformers_hub, "HF_MODULES_CACHE", str(modules_cache))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        hf_constants.HF_HOME = previous_constants["hf_home"]
        hf_constants.HF_HUB_CACHE = previous_constants["hf_hub_cache"]
        transformers_hub.constants.HF_HOME = previous_constants["transformers_hf_home"]
        transformers_hub.constants.HF_HUB_CACHE = previous_constants["transformers_hf_hub_cache"]
        _set_module_attr(
            transformers_hub, "TRANSFORMERS_CACHE", previous_constants["transformers_cache"]
        )
        _set_module_attr(
            transformers_hub, "HF_MODULES_CACHE", previous_constants["hf_modules_cache"]
        )


def _collect_env_info() -> dict[str, JsonValue]:
    return {
        "python_version": cast(JsonValue, sys.version.split()[0]),
        "torch_version": cast(JsonValue, torch.__version__),
        "cuda_available": cast(JsonValue, torch.cuda.is_available()),
    }


def _git_output(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _collect_git_info() -> dict[str, JsonValue]:
    sha = _git_output(["rev-parse", "HEAD"])
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git_output(["status", "--short"])
    return {
        "sha": cast(JsonValue, sha),
        "branch": cast(JsonValue, branch),
        "dirty": cast(JsonValue, bool(status)),
    }


def _load_training_dataset(data_config: DataConfig) -> Dataset:
    dataset_like = load_hf_dataset(data_config)
    if not isinstance(dataset_like, Dataset):
        raise TypeError(
            "SFT training currently expects a Hugging Face Dataset split, not a dataset dict."
        )
    return dataset_like


def _prepare_train_split(
    dataset: Dataset, *, data_config: DataConfig, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    split_result = deterministic_split(
        dataset,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        key_fields=data_config.split_key_fields,
    )
    train_split = split_result["train"]["dataset"]
    original_columns = list(train_split.column_names)

    processed = train_split.map(
        lambda record: preprocess_record(
            record,
            tokenizer=tokenizer,
            instruction_field=data_config.prompt_field or "instruction",
            output_field=data_config.response_field or "output",
            text_field=data_config.text_field,
        ),
        remove_columns=original_columns,
    )
    return processed.select_columns(["input_ids", "labels", data_config.text_field])


def _load_tokenizer(model_id: str, *, cache_dir: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_id: str,
    *,
    cache_dir: Path,
    qlora_enabled: bool,
    gradient_checkpointing: bool,
) -> PreTrainedModel:
    runtime = get_accelerate_runtime()
    quantization_config = build_quantization_config(enabled=qlora_enabled)
    model_kwargs: dict[str, object] = {}
    model_kwargs["cache_dir"] = str(cache_dir)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": runtime.local_process_index}
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs))
    model.config.use_cache = False

    if quantization_config is not None:
        model = cast(
            PreTrainedModel,
            prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            ),
        )

    return model


def _build_training_args(
    train_config: TrainConfig, run_dir: Path, *, dataset_text_field: str
) -> Any:
    use_cpu = not torch.cuda.is_available()
    return SFTConfig(
        output_dir=str(run_dir),
        run_name=_resolve_run_id(train_config),
        seed=train_config.seed,
        max_steps=train_config.max_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        logging_steps=train_config.logging_steps,
        save_strategy="steps",
        save_steps=train_config.save_steps,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataset_text_field=dataset_text_field,
        max_length=train_config.max_seq_len,
        packing=False,
        use_cpu=use_cpu,
        fp16=False,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=train_config.gradient_checkpointing,
        ddp_find_unused_parameters=False if get_accelerate_runtime().num_processes > 1 else None,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    return torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError)


def _train_once(
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    cache_dir: Path,
    resolved_model_id: str,
    run_dir: Path,
) -> tuple[float, bool, str | None, Dataset]:
    tokenizer = _load_tokenizer(resolved_model_id, cache_dir=cache_dir)
    dataset = _load_training_dataset(data_config)
    train_dataset = _prepare_train_split(dataset, data_config=data_config, tokenizer=tokenizer)

    qlora_enabled = _should_enable_qlora()
    model = _load_model(
        resolved_model_id,
        cache_dir=cache_dir,
        qlora_enabled=qlora_enabled,
        gradient_checkpointing=train_config.gradient_checkpointing,
    )
    training_args = _build_training_args(
        train_config,
        run_dir,
        dataset_text_field=data_config.text_field,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=SupervisedDataCollator(tokenizer),
        peft_config=build_lora_config(train_config),
    )
    resume_checkpoint = resolve_resume_checkpoint(run_dir, train_config)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_loss = float(cast(float, train_result.training_loss))
    trainer.save_model(str(run_dir / "lora_adapter"))
    return final_loss, qlora_enabled, resume_checkpoint, train_dataset


def _log_mlflow_run(
    *,
    obs_config: ObsConfig,
    run_id: str,
    manifest_path: Path,
    train_config_path: Path,
    params: dict[str, str],
    final_loss: float,
) -> None:
    with start_mlflow_run(
        tracking_uri=obs_config.tracking_uri,
        experiment_name=obs_config.experiment_name,
        run_name=run_id,
    ):
        _ = mlflow.log_params(params)
        _ = mlflow.log_metric("final_loss", final_loss)
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_artifact(str(train_config_path))


def run_sft(
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    obs_config: ObsConfig,
    data_config_path: Path,
    train_config_path: Path,
    obs_config_path: Path,
) -> Path:
    """Execute supervised fine-tuning and return the created run directory."""

    runtime = get_accelerate_runtime()
    set_seed(train_config.seed)
    logger = configure_logging()

    resolved_model_id = _resolve_model_id(train_config)
    run_id = _resolve_run_id(train_config)
    run_dir = Path(train_config.output_dir) / run_id
    log_path = run_dir / "train.log"
    _ = run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve_model_cache_dir(data_config=data_config, train_config=train_config)
    retry_decisions: list[str] = []

    with _override_hf_cache_env(cache_dir):
        active_train_config = train_config
        retry_attempt = 0
        while True:
            try:
                final_loss, qlora_enabled, resumed_from, train_dataset = _train_once(
                    data_config=data_config,
                    train_config=active_train_config,
                    cache_dir=cache_dir,
                    resolved_model_id=resolved_model_id,
                    run_dir=run_dir,
                )
                break
            except BaseException as exc:
                if not _is_cuda_oom(exc):
                    raise
                if retry_attempt >= train_config.oom_retries:
                    raise
                next_config, decision = next_oom_retry_config(active_train_config)
                if next_config is None or decision is None:
                    raise
                retry_attempt += 1
                active_train_config = next_config
                retry_decisions.append(decision)
                logger.warning(
                    "CUDA OOM while training run %s; retry %s/%s. %s",
                    run_id,
                    retry_attempt,
                    train_config.oom_retries,
                    decision,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    wait_for_everyone()

    manifest_path = write_run_manifest(
        run_dir=run_dir,
        config_paths=[data_config_path, train_config_path, obs_config_path],
        env_info=_collect_env_info(),
        git_info=_collect_git_info(),
        seed=train_config.seed,
        dataset_id=data_config.dataset_id,
        model_id=resolved_model_id,
    )

    if runtime.is_main_process:
        summary = {
            "run_id": run_id,
            "model_id": resolved_model_id,
            "qlora_enabled": qlora_enabled,
            "num_train_rows": len(train_dataset),
            "final_loss": final_loss,
            "resumed_from_checkpoint": resumed_from,
            "oom_fallbacks": retry_decisions,
        }
        _ = log_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _log_mlflow_run(
            obs_config=obs_config,
            run_id=run_id,
            manifest_path=manifest_path,
            train_config_path=train_config_path,
            params={
                "dataset_id": data_config.dataset_id,
                "requested_model_id": train_config.model_id,
                "resolved_model_id": resolved_model_id,
                "seed": str(train_config.seed),
                "max_steps": str(train_config.max_steps),
                "per_device_train_batch_size": str(train_config.per_device_train_batch_size),
                "gradient_accumulation_steps": str(train_config.gradient_accumulation_steps),
                "learning_rate": str(train_config.learning_rate),
                "qlora_enabled": str(qlora_enabled),
                "world_size": str(runtime.num_processes),
            },
            final_loss=final_loss,
        )
        logger.info("Finished SFT run %s with loss %.6f", run_id, final_loss)

    return run_dir


def run_sft_from_paths(
    *,
    data_config_path: str | Path,
    train_config_path: str | Path,
    obs_config_path: str | Path,
) -> Path:
    """Load YAML configs and execute SFT training."""

    resolved_data_path = Path(data_config_path)
    resolved_train_path = Path(train_config_path)
    resolved_obs_path = Path(obs_config_path)
    return run_sft(
        data_config=load_yaml(resolved_data_path, DataConfig),
        train_config=load_yaml(resolved_train_path, TrainConfig),
        obs_config=load_yaml(resolved_obs_path, ObsConfig),
        data_config_path=resolved_data_path,
        train_config_path=resolved_train_path,
        obs_config_path=resolved_obs_path,
    )
