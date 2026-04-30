from __future__ import annotations

import subprocess

import pytest


def test_select_gpus_prefers_low_residency_devices_for_multi_gpu_request() -> None:
    from aiinfra_e2e.serve.resource_plan import GpuDevice, select_gpus

    inventory = [
        GpuDevice(index=0, memory_used_mb=74000, memory_total_mb=81920, utilization_gpu=0, process_count=1),
        GpuDevice(index=2, memory_used_mb=20000, memory_total_mb=81920, utilization_gpu=0, process_count=1),
        GpuDevice(index=3, memory_used_mb=17, memory_total_mb=81920, utilization_gpu=0, process_count=0),
    ]

    allocation = select_gpus(inventory, count=2, policy="multi_free")

    assert allocation.cuda_visible_devices == "3,2"
    assert allocation.selected_gpu_indices == [3, 2]


def test_select_gpus_respect_env_returns_existing_value() -> None:
    from aiinfra_e2e.serve.resource_plan import GpuDevice, select_gpus

    inventory = [GpuDevice(index=1, memory_used_mb=0, memory_total_mb=81920, utilization_gpu=0, process_count=0)]

    allocation = select_gpus(inventory, count=1, policy="respect_env", current_value="5,6")

    assert allocation.cuda_visible_devices == "5,6"
    assert allocation.selected_gpu_indices == [5, 6]


def test_select_gpus_preferred_then_auto_honors_preferences_and_falls_back() -> None:
    from aiinfra_e2e.serve.resource_plan import GpuDevice, select_gpus

    inventory = [
        GpuDevice(index=1, memory_used_mb=2, memory_total_mb=81920, utilization_gpu=0, process_count=0),
        GpuDevice(index=4, memory_used_mb=1, memory_total_mb=81920, utilization_gpu=0, process_count=0),
        GpuDevice(index=7, memory_used_mb=0, memory_total_mb=81920, utilization_gpu=0, process_count=0),
    ]

    preferred = select_gpus(
        inventory,
        count=2,
        policy="preferred_then_auto",
        preferred_indices=[4, 1],
    )
    fallback = select_gpus(
        inventory,
        count=2,
        policy="preferred_then_auto",
        preferred_indices=[9, 4],
    )

    assert preferred.selected_gpu_indices == [4, 1]
    assert preferred.cuda_visible_devices == "4,1"
    assert fallback.selected_gpu_indices == [4, 7]
    assert fallback.cuda_visible_devices == "4,7"


def test_allocate_free_port_avoids_excluded_ports() -> None:
    from aiinfra_e2e.serve.resource_plan import allocate_free_port

    excluded = {0, 1, 2}

    port = allocate_free_port(excluded_ports=excluded)

    assert port not in excluded
    assert 0 < port < 65536


def test_collect_gpu_inventory_parses_nvidia_smi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from aiinfra_e2e.serve.resource_plan import collect_gpu_inventory

    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "0, 74000, 81920, 0, 1\n"
                "3, 17, 81920, 0, 0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    inventory = collect_gpu_inventory()

    assert [device.index for device in inventory] == [0, 3]
    assert inventory[1].memory_used_mb == 17
