from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import socket
import subprocess
from typing import cast


@dataclass(frozen=True)
class GpuDevice:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu: int
    process_count: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


@dataclass(frozen=True)
class GpuAllocation:
    selected_gpu_indices: list[int]
    cuda_visible_devices: str


def collect_gpu_inventory() -> list[GpuDevice]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    inventory: list[GpuDevice] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index = int(parts[0])
        memory_used_mb = int(parts[1])
        memory_total_mb = int(parts[2])
        utilization_gpu = int(parts[3])
        process_count = int(parts[4]) if len(parts) > 4 else 0
        inventory.append(
            GpuDevice(
                index=index,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                utilization_gpu=utilization_gpu,
                process_count=process_count,
            )
        )
    return inventory


def allocate_free_port(*, excluded_ports: Iterable[int] | None = None) -> int:
    excluded = set(excluded_ports or ())
    for _ in range(1024):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            address = cast(tuple[str, int], sock.getsockname())
            port = int(address[1])
        if port not in excluded:
            return port
    raise RuntimeError("Unable to allocate a free port outside the excluded set.")


def select_gpus(
    inventory: list[GpuDevice],
    *,
    count: int,
    policy: str,
    current_value: str | None = None,
    preferred_indices: list[int] | None = None,
) -> GpuAllocation:
    if policy == "respect_env":
        if current_value is None:
            raise ValueError("current_value is required for respect_env policy")
        return _allocation_from_cuda_visible_devices(current_value)

    if not inventory:
        raise ValueError("GPU inventory is empty")

    if policy == "single_free":
        selected = _select_most_free_devices(inventory, 1)
    elif policy == "multi_free":
        selected = _select_low_residency_devices(inventory, count)
    elif policy == "preferred_then_auto":
        selected = _select_preferred_then_auto(inventory, count, preferred_indices or [])
    else:
        raise ValueError(f"Unknown GPU selection policy: {policy}")

    return _allocation_from_indices(selected)


def _allocation_from_indices(indices: list[int]) -> GpuAllocation:
    return GpuAllocation(selected_gpu_indices=indices, cuda_visible_devices=",".join(str(index) for index in indices))


def _allocation_from_cuda_visible_devices(current_value: str) -> GpuAllocation:
    indices = [int(part) for part in current_value.split(",") if part.strip()]
    return _allocation_from_indices(indices)


def _device_sort_key(device: GpuDevice) -> tuple[int, int]:
    return (device.memory_used_mb, device.index)


def _select_low_residency_devices(inventory: list[GpuDevice], count: int) -> list[int]:
    selected = [device.index for device in sorted(inventory, key=_device_sort_key)[:count]]
    if len(selected) != count:
        raise ValueError("Not enough free GPUs available")
    return selected


def _select_most_free_devices(inventory: list[GpuDevice], count: int) -> list[int]:
    selected = [device.index for device in sorted(inventory, key=lambda device: (-device.memory_free_mb, device.index))[:count]]
    if len(selected) != count:
        raise ValueError("Not enough free GPUs available")
    return selected


def _select_preferred_then_auto(
    inventory: list[GpuDevice],
    count: int,
    preferred_indices: list[int],
) -> list[int]:
    selected: list[int] = []
    remaining = {device.index: device for device in inventory}

    for index in preferred_indices:
        if index in remaining and index not in selected:
            selected.append(index)
            _ = remaining.pop(index)
            if len(selected) == count:
                return selected

    if len(selected) < count:
        auto_selected = [device.index for device in sorted(remaining.values(), key=_device_sort_key)]
        selected.extend(auto_selected[: count - len(selected)])

    if len(selected) != count:
        raise ValueError("Not enough GPUs available for preferred_then_auto policy")
    return selected
