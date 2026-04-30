import importlib
import subprocess
from typing import Protocol, cast

import pytest


class _GpuSelector(Protocol):
    def __call__(self, current_value: str | None = None) -> str | None: ...


def _load_gpu_selector() -> _GpuSelector:
    module = importlib.import_module("aiinfra_e2e.gpu")
    return cast(_GpuSelector, getattr(module, "select_cuda_visible_devices"))


def test_select_cuda_visible_devices_keeps_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("nvidia-smi should not run when CUDA_VISIBLE_DEVICES is already set")

    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert _load_gpu_selector()(current_value="3") == "3"


def test_select_cuda_visible_devices_prefers_gpu_with_most_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, 813\n1, 813\n2, 81140\n3, 81140\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert _load_gpu_selector()(None) == "2"


def test_select_cuda_visible_devices_keeps_legacy_single_gpu_wrapper_contract() -> None:
    from aiinfra_e2e.serve.resource_plan import GpuDevice, select_gpus

    inventory = [
        GpuDevice(index=0, memory_used_mb=10, memory_total_mb=81920, utilization_gpu=0, process_count=0),
        GpuDevice(index=1, memory_used_mb=20, memory_total_mb=81920, utilization_gpu=0, process_count=0),
    ]

    allocation = select_gpus(inventory, count=1, policy="single_free")

    assert allocation.cuda_visible_devices == "0"
    assert allocation.selected_gpu_indices == [0]
