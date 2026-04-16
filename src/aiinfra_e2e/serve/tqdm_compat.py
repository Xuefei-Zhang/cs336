"""Compatibility helpers for known vLLM/tqdm constructor mismatches."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import wraps
from typing import Any, cast


def ensure_vllm_tqdm_compat() -> None:
    """Patch vLLM's DisabledTqdm wrapper to ignore caller-provided disable kwargs."""
    try:
        weight_utils = importlib.import_module("vllm.model_executor.model_loader.weight_utils")
    except ImportError:
        return

    DisabledTqdm = weight_utils.DisabledTqdm

    current_init = DisabledTqdm.__init__
    if getattr(current_init, "_aiinfra_e2e_vllm_tqdm_compat", False):
        return

    @wraps(current_init)
    def _compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("disable", None)
        current_init(self, *args, **kwargs)

    setattr(_compat_init, "_aiinfra_e2e_vllm_tqdm_compat", True)
    DisabledTqdm.__init__ = cast(Callable[..., None], _compat_init)
