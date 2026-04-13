"""Hugging Face dataset loading helpers with bounded retries."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from requests import exceptions as requests_exceptions

from aiinfra_e2e.config import DataConfig

DatasetLike = Dataset | DatasetDict | IterableDataset | IterableDatasetDict
_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests_exceptions.ConnectionError,
    requests_exceptions.Timeout,
)


def _is_transient_error(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT_EXCEPTIONS)


def _compute_backoff_delay(attempt: int, base_delay_seconds: float, jitter_seconds: float) -> float:
    jitter = random.uniform(0.0, jitter_seconds) if jitter_seconds > 0 else 0.0
    return base_delay_seconds * (2 ** (attempt - 1)) + jitter


def load_hf_dataset(
    config: DataConfig,
    *,
    max_retries: int = 5,
    base_delay_seconds: float = 1.0,
    jitter_seconds: float = 0.25,
    dataset_loader: Callable[..., Any] | None = None,
) -> DatasetLike:
    """Load a dataset from Hugging Face using the configured cache dir and bounded retries."""

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    loader = dataset_loader or load_dataset

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return loader(config.dataset_id, split=config.split, cache_dir=config.cache_dir)
        except Exception as exc:
            if not _is_transient_error(exc) or attempt == max_retries:
                raise
            last_error = exc
            time.sleep(_compute_backoff_delay(attempt, base_delay_seconds, jitter_seconds))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Dataset load failed without raising an exception")
