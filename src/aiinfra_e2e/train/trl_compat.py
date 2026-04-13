"""Compatibility shim for TRL SFT public imports."""

from __future__ import annotations

from typing import Any

_cached_objects: tuple[type[Any], type[Any]] | None = None


def load_trl_sft_objects(*, force_reload: bool = False) -> tuple[type[Any], type[Any]]:
    """Load TRL SFT objects via public API only."""

    global _cached_objects
    if _cached_objects is not None and not force_reload:
        return _cached_objects

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "TRL public exports are unavailable; install a TRL version that exposes "
            "trl.SFTTrainer and trl.SFTConfig. Private trainer-internal imports are unsupported."
        ) from exc

    _cached_objects = (SFTTrainer, SFTConfig)
    return _cached_objects


SFTTrainer, SFTConfig = load_trl_sft_objects()
