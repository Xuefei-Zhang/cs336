"""Training pipeline helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["run_sft", "run_sft_from_paths"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from aiinfra_e2e.train.sft import run_sft, run_sft_from_paths

    exports = {
        "run_sft": run_sft,
        "run_sft_from_paths": run_sft_from_paths,
    }
    return exports[name]
