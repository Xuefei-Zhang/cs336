from pathlib import Path


def test_smoke_cpu_script_prefers_python_bin_for_cli() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_cpu.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert '"$PYTHON_BIN" -m aiinfra_e2e.cli "$@"' in script_text
    assert 'command -v "$CLI_BIN"' not in script_text


def test_e2e_gpu_script_prefers_python_bin_for_cli() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "e2e_gpu.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert '"$PYTHON_BIN" -m aiinfra_e2e.cli "$@"' in script_text
    assert 'command -v "$CLI_BIN"' not in script_text
