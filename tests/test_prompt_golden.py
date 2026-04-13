from aiinfra_e2e.data.prompt_format import DEFAULT_SYSTEM_PROMPT, build_messages, render_prompt


class FakeChatTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = "\n".join(f"<{message['role']}>:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\n<assistant>:"
        return rendered


def test_build_messages_returns_canonical_qwen_chat_structure() -> None:
    messages = build_messages(
        instruction="Translate to Chinese",
        input_text="hello world",
        output="你好，世界",
    )

    assert messages == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Translate to Chinese\n\nInput:\nhello world"},
        {"role": "assistant", "content": "你好，世界"},
    ]


def test_render_prompt_uses_chat_template_when_available() -> None:
    messages = build_messages(
        instruction="Summarize the text",
        output="Short summary.",
    )

    rendered = render_prompt(messages, tokenizer=FakeChatTokenizer())

    assert rendered == (
        f"<system>:{DEFAULT_SYSTEM_PROMPT}\n<user>:Summarize the text\n<assistant>:Short summary."
    )
