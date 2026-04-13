"""Golden prompt regression validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GoldenValidationResult:
    name: str | None
    prompt: str
    output: str
    passed: bool
    failed_constraints: list[str]
    metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ConstraintValue = int | float | str | list[str]
Constraints = Mapping[str, ConstraintValue]


def repetition_ratio(text: str) -> float:
    """Return the share of repeated adjacent tokens in a text."""

    tokens = [token for token in text.lower().split() if token]
    if len(tokens) < 2:
        return 0.0
    repeated = sum(1 for left, right in zip(tokens, tokens[1:], strict=False) if left == right)
    return repeated / (len(tokens) - 1)


def _contains_all(output: str, required_substrings: Sequence[str]) -> bool:
    return all(substring in output for substring in required_substrings)


def _contains_none(output: str, banned_substrings: Sequence[str]) -> bool:
    return all(substring not in output for substring in banned_substrings)


def validate_golden_case(
    *,
    prompt: str,
    output: str,
    constraints: Constraints,
    name: str | None = None,
) -> GoldenValidationResult:
    """Validate one generated output against simple regression constraints."""

    failed_constraints: list[str] = []
    char_length = len(output)
    repeated_ratio = repetition_ratio(output)

    min_length = constraints.get("min_length")
    if isinstance(min_length, int) and char_length < min_length:
        failed_constraints.append("min_length")

    max_length = constraints.get("max_length")
    if isinstance(max_length, int) and char_length > max_length:
        failed_constraints.append("max_length")

    must_contain = constraints.get("must_contain")
    if isinstance(must_contain, list) and not _contains_all(output, must_contain):
        failed_constraints.append("must_contain")

    must_not_contain = constraints.get("must_not_contain")
    if isinstance(must_not_contain, list) and not _contains_none(output, must_not_contain):
        failed_constraints.append("must_not_contain")

    max_repetition_ratio = constraints.get("max_repetition_ratio")
    if isinstance(max_repetition_ratio, int | float) and repeated_ratio > float(max_repetition_ratio):
        failed_constraints.append("max_repetition_ratio")

    return GoldenValidationResult(
        name=name,
        prompt=prompt,
        output=output,
        passed=not failed_constraints,
        failed_constraints=failed_constraints,
        metrics={
            "char_length": char_length,
            "repetition_ratio": repeated_ratio,
        },
    )
