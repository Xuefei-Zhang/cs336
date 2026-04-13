"""Accelerate runtime helpers for distributed training."""

from __future__ import annotations

from dataclasses import dataclass

from accelerate import PartialState


@dataclass(frozen=True)
class AccelerateRuntime:
    process_index: int
    local_process_index: int
    num_processes: int
    is_main_process: bool


def get_accelerate_runtime() -> AccelerateRuntime:
    """Read the current Accelerate process topology."""

    state = PartialState()
    return AccelerateRuntime(
        process_index=state.process_index,
        local_process_index=state.local_process_index,
        num_processes=state.num_processes,
        is_main_process=state.is_main_process,
    )


def wait_for_everyone() -> None:
    """Synchronize all ranks when running with Accelerate."""

    PartialState().wait_for_everyone()
