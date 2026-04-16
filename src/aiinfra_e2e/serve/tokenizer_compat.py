"""Compatibility helpers for tokenizer API mismatches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast


def ensure_all_special_tokens_extended() -> None:
    """Add a compatibility property expected by vLLM when missing."""
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    def _all_special_tokens_extended(self: Any) -> list[str]:
        return list(cast(Any, self).all_special_tokens)

    setattr(
        PreTrainedTokenizerBase,
        "all_special_tokens_extended",
        property(cast(Callable[[Any], list[str]], _all_special_tokens_extended)),
    )
