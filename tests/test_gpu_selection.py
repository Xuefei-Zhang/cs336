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
