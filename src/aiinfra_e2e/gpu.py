from __future__ import annotations

import subprocess


def select_cuda_visible_devices(current_value: str | None = None) -> str | None:
    if current_value:
        return current_value

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    best_gpu: tuple[int, int] | None = None
    for line in completed.stdout.splitlines():
        raw_index, raw_memory_free = [part.strip() for part in line.split(",", maxsplit=1)]
        candidate = (int(raw_memory_free), int(raw_index))
        if (
            best_gpu is None
            or candidate[0] > best_gpu[0]
            or (candidate[0] == best_gpu[0] and candidate[1] < best_gpu[1])
        ):
            best_gpu = candidate

    if best_gpu is None:
        return None

    return str(best_gpu[1])
