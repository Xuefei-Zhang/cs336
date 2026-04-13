"""Checkpoint discovery and bounded retry helpers for training."""

from __future__ import annotations

import re
from pathlib import Path

from aiinfra_e2e.config import TrainConfig

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the newest numeric checkpoint directory inside a run directory."""

    base_dir = Path(run_dir)
    if not base_dir.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.match(child.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), child))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def should_resume(train_config: TrainConfig) -> bool:
    """Return whether resume behavior is enabled for this config."""

    resume_value = train_config.resume_from_checkpoint
    return resume_value is True or (isinstance(resume_value, str) and bool(resume_value.strip()))


def resolve_resume_checkpoint(run_dir: str | Path, train_config: TrainConfig) -> str | None:
    """Resolve the checkpoint path to hand to Trainer.train(...)."""

    resume_value = train_config.resume_from_checkpoint
    if resume_value is False:
        return None
    if isinstance(resume_value, str):
        return resume_value
    checkpoint_path = latest_checkpoint(run_dir)
    if checkpoint_path is None:
        return None
    return str(checkpoint_path)


def next_oom_retry_config(train_config: TrainConfig) -> tuple[TrainConfig | None, str | None]:
    """Return a smaller config for the next OOM retry, if one exists."""

    if train_config.per_device_train_batch_size > 1:
        next_batch_size = max(1, train_config.per_device_train_batch_size // 2)
        updated = train_config.model_copy(update={"per_device_train_batch_size": next_batch_size})
        return updated, (
            "Reduced per_device_train_batch_size from "
            f"{train_config.per_device_train_batch_size} to {next_batch_size}"
        )

    if train_config.max_seq_len is not None and train_config.max_seq_len > 1:
        next_max_seq_len = max(1, train_config.max_seq_len // 2)
        updated = train_config.model_copy(update={"max_seq_len": next_max_seq_len})
        return updated, f"Reduced max_seq_len from {train_config.max_seq_len} to {next_max_seq_len}"

    return None, None
