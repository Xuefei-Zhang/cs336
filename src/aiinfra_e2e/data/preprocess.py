"""SFT preprocessing helpers for Alpaca-style datasets."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from aiinfra_e2e.data.prompt_format import (
    Message,
    build_messages,
    has_configured_chat_template,
    render_prompt,
)


class SFTRecord(TypedDict):
    messages: list[Message]
    text: str
    input_ids: list[int]
    labels: list[int]


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    return cast(list[int], input_ids)


def _build_user_prefix(messages: list[Message], *, tokenizer: Any) -> list[int]:
    prompt_messages = messages[:-1]
    if has_configured_chat_template(tokenizer):
        prefix_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        return cast(list[int], prefix_ids)

    prompt_text = render_prompt(prompt_messages, tokenizer=None)
    return _tokenize_text(tokenizer, prompt_text)


def preprocess_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    text_field: str = "text",
) -> SFTRecord:
    """Convert an Alpaca-style row into chat messages, rendered text, and assistant-only labels."""

    instruction = str(record[instruction_field])
    input_value = record.get(input_field)
    output = str(record[output_field])

    input_text = None if input_value in (None, "") else str(input_value)
    messages = build_messages(instruction=instruction, input_text=input_text, output=output)
    text = render_prompt(messages, tokenizer=tokenizer)
    input_ids = _tokenize_text(tokenizer, text)
    prefix_ids = _build_user_prefix(messages, tokenizer=tokenizer)

    labels = [-100] * len(input_ids)
    assistant_start = len(prefix_ids)
    labels[assistant_start:] = input_ids[assistant_start:]

    processed_record: SFTRecord = {
        "messages": messages,
        "text": text,
        "input_ids": input_ids,
        "labels": labels,
    }
    if text_field != "text":
        processed_record[text_field] = text
    return processed_record
