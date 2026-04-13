from typing import cast

from aiinfra_e2e.data.preprocess import preprocess_record
from aiinfra_e2e.data.prompt_format import DEFAULT_SYSTEM_PROMPT


class FakeEncoding(dict[str, list[int]]):
    pass


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.next_id = 100

    def _token_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.next_id += 1
        return self.vocab[token]

    def _encode_text(self, text: str) -> list[int]:
        return [self._token_id(token) for token in text.split()]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}> {message['content']}")
        if add_generation_prompt:
            rendered.append("<assistant>")
        text = "\n".join(rendered)
        if tokenize:
            return self._encode_text(text)
        return text

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> FakeEncoding:
        return FakeEncoding({"input_ids": self._encode_text(text)})


def test_preprocess_record_returns_sft_messages_and_masked_labels() -> None:
    tokenizer = FakeTokenizer()

    record = preprocess_record(
        {
            "instruction": "Translate to Chinese",
            "input": "good morning",
            "output": "早上好",
        },
        tokenizer=tokenizer,
    )

    assert record["messages"] == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Translate to Chinese\n\nInput:\ngood morning"},
        {"role": "assistant", "content": "早上好"},
    ]
    assert record["text"] == (
        f"<system> {DEFAULT_SYSTEM_PROMPT}\n"
        "<user> Translate to Chinese\n\nInput:\ngood morning\n"
        "<assistant> 早上好"
    )
    assert len(record["input_ids"]) == len(record["labels"])


def test_preprocess_record_masks_prefix_and_keeps_assistant_tokens_trainable() -> None:
    tokenizer = FakeTokenizer()

    record = preprocess_record(
        {
            "instruction": "Summarize",
            "output": "brief answer",
        },
        tokenizer=tokenizer,
    )

    prefix_text = f"<system> {DEFAULT_SYSTEM_PROMPT}\n<user> Summarize"
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    assistant_ids = tokenizer("<assistant> brief answer", add_special_tokens=False)["input_ids"]

    assert record["labels"][: len(prefix_ids)] == [-100] * len(prefix_ids)
    assert record["labels"][-len(assistant_ids) :] == assistant_ids


def test_preprocess_record_honors_custom_prompt_response_and_text_fields() -> None:
    tokenizer = FakeTokenizer()

    record = preprocess_record(
        {
            "prompt": "Translate to English",
            "context": "早上好",
            "answer": "good morning",
        },
        tokenizer=tokenizer,
        instruction_field="prompt",
        input_field="context",
        output_field="answer",
        text_field="formatted_prompt",
    )

    record_dict = cast(dict[str, object], cast(object, record))

    assert record_dict["formatted_prompt"] == (
        f"<system> {DEFAULT_SYSTEM_PROMPT}\n"
        "<user> Translate to English\n\nInput:\n早上好\n"
        "<assistant> good morning"
    )
    assert record["messages"][1]["content"] == "Translate to English\n\nInput:\n早上好"
    assert record["labels"][-2:] == tokenizer("good morning", add_special_tokens=False)["input_ids"]
