"""Compatibility shim for TRL SFT public imports."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_cached_objects: tuple[type[Any], type[Any]] | None = None


def _clear_cached_callable(func: Callable[..., object] | object) -> None:
    cache_clear = getattr(func, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


@lru_cache(maxsize=1)
def disable_torchvision_if_broken() -> bool:
    """Disable torchvision-dependent transformers paths when torchvision import is broken."""

    try:
        importlib.import_module("torchvision")
    except Exception as exc:
        for module_name in tuple(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)

        import transformers.utils
        import transformers.utils.import_utils as transformers_import_utils

        _clear_cached_callable(transformers.utils.is_torchvision_available)
        _clear_cached_callable(transformers_import_utils.is_torchvision_available)

        transformers.utils.is_torchvision_available = lambda: False
        transformers_import_utils.is_torchvision_available = lambda: False
        logger.warning(
            "Disabled torchvision for TRL import because importing torchvision failed: %s", exc
        )
        return True

    return False


def load_trl_sft_objects(*, force_reload: bool = False) -> tuple[type[Any], type[Any]]:
    """Load TRL SFT objects via public API only."""

    global _cached_objects
    if _cached_objects is not None and not force_reload:
        return _cached_objects

    disable_torchvision_if_broken()

    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "TRL public exports are unavailable; install a TRL version that exposes "
            "trl.SFTTrainer and trl.SFTConfig. Private trainer-internal imports are unsupported."
        ) from exc

    _cached_objects = (SFTTrainer, SFTConfig)
    return _cached_objects
