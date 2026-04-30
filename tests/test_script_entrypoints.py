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


def test_serve_trace_gpu23_script_prefers_python_bin_for_cli() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "serve_trace_gpu23.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert '"$PYTHON_BIN" -m aiinfra_e2e.cli serve --config "$EFFECTIVE_SERVE_CONFIG"' in script_text


def test_serve_trace_gpu23_script_is_strict_bash_and_has_trap_cleanup() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "serve_trace_gpu23.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script_text
    assert "trap cleanup EXIT" in script_text
    assert 'TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aiinfra-e2e-serve-trace.XXXXXX")' in script_text


def test_serve_trace_gpu23_script_has_enable_nsys_toggle() -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "serve_trace_gpu23.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert 'if [[ "${ENABLE_NSYS:-0}" == "1" ]]; then' in script_text
    assert "nsys profile" in script_text
