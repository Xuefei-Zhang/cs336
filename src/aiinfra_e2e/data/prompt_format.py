"""Prompt formatting helpers for chat-style SFT datasets."""

from __future__ import annotations

from typing import Any, Protocol, cast

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

Message = dict[str, str]


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[Message],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str: ...


def has_configured_chat_template(tokenizer: Any | None) -> bool:
    """Return whether the tokenizer can safely render chat templates."""

    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return False
    if not hasattr(tokenizer, "chat_template"):
        return True
    return bool(getattr(tokenizer, "chat_template"))


def _build_user_content(instruction: str, input_text: str | None = None) -> str:
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def build_messages(
    *,
    instruction: str,
    output: str | None = None,
    input_text: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[Message]:
    """Build canonical chat messages from Alpaca-style fields."""

    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(instruction, input_text)},
    ]

    if output is not None:
        messages.append({"role": "assistant", "content": output})

    return messages


def render_prompt(messages: list[Message], *, tokenizer: Any | None = None) -> str:
    """Render chat messages to text, preferring tokenizer chat templates when available."""

    if has_configured_chat_template(tokenizer):
        template_tokenizer = cast(ChatTemplateTokenizer, tokenizer)
        return template_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
