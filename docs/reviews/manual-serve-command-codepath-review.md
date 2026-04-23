# manual serve command codepath review

## What is being traced

This review traces the exact runtime path for the manual serve command you ran:

```bash
cd /home/xuefeiz2/self/cs336 && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
HF_HOME=$PWD/artifacts/hf-cache \
HF_HUB_CACHE=$PWD/artifacts/hf-cache/hub \
HF_DATASETS_CACHE=$PWD/artifacts/hf-cache/datasets \
TRANSFORMERS_CACHE=$PWD/artifacts/hf-cache/transformers \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost && \
.venv/bin/python -m aiinfra_e2e.cli serve --config artifacts/tmp/interactive-serve.yaml
```

The goal is to show how execution moves through four layers:

1. shell/runtime assumptions
2. repo CLI dispatch
3. repo-owned serve wrapper
4. spawned vLLM child process

This document is intentionally narrower than `docs/reviews/serve-vllm-deep-dive.md`. That earlier document explains the serve architecture in general; this one follows **your exact manual command** as a concrete runtime path.

## Ordered codepath

1. The shell sets CUDA visibility, Hugging Face cache paths, and localhost proxy-bypass variables before Python starts.
2. `.venv/bin/python -m aiinfra_e2e.cli` runs the repo's CLI module, so control enters `src/aiinfra_e2e/cli.py`.
3. Typer dispatches the `serve` subcommand to `serve_command(config=...)`.
4. `serve_command` validates that `artifacts/tmp/interactive-serve.yaml` exists, then loads it into a typed `ServeConfig` via `_load_config(..., ServeConfig)` and `load_yaml(...)`.
5. The CLI hands the validated config to `run_vllm_server_from_config(serve_config)`.
6. `run_vllm_server_from_config` constructs a `ManagedVLLMServer`, starts it, and then stays alive in a wrapper-side keepalive loop that periodically updates metrics.
7. `ManagedVLLMServer.start()` first starts the wrapper metrics HTTP server, then builds the child command and child environment.
8. The child command is `sys.executable -m vllm.entrypoints.openai.api_server ...`, so the repo is still running the standard vLLM OpenAI-compatible API server, just through a wrapper.
9. The child environment prepends the repo root to `PYTHONPATH` and sets compat flags so that `sitecustomize.py` can patch vLLM startup behavior inside the child process.
10. After spawning the child with `subprocess.Popen(...)`, the wrapper polls `GET /v1/models` against `ServeConfig.base_url` until the server is ready or the startup timeout expires.
11. Once the child is ready, the wrapper stays alive in a loop, updating wrapper-side GPU metrics every second.
12. On Ctrl+C or other normal Python shutdown paths, the wrapper calls `stop()`, which tries `terminate()`, waits, and then escalates to `kill()` if the child is still alive.

## Snippets

### 1. The module entrypoint lands in `cli.py`
**File:** `src/aiinfra_e2e/cli.py:206-211`
```python
def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

Because the command uses `-m aiinfra_e2e.cli`, Python executes this module and reaches `main()`, which hands control to the Typer app.

### 2. `serve` is a real CLI route, not a shell alias
**File:** `src/aiinfra_e2e/cli.py:175-185`
```python
@app.command("serve")
def serve_command(config: ConfigOption = None) -> None:
    """Validate serve config and start the vLLM wrapper."""

    if config is None:
        typer.echo("Serve command stub. Provide --config to start a YAML-configured server.")
        return

    serve_config = cast(ServeConfig, _load_config(config, ServeConfig))
    typer.echo(f"Started serve wrapper from {config}")
    run_vllm_server_from_config(serve_config)
```

This is the first important hop. The manual shell command is not starting vLLM directly; it is entering the repo's `serve` CLI route.

### 3. The config path is validated before YAML loading
**File:** `src/aiinfra_e2e/cli.py:46-57`
```python
def _load_config(config_path: Path, model_cls: type[BaseModel]) -> BaseModel:
    """Validate a YAML config path and load it into the requested model."""

    if not config_path.exists() or not config_path.is_file():
        typer.echo(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    try:
        return load_yaml(config_path, model_cls)
    except (ValidationError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
```

The relative path `artifacts/tmp/interactive-serve.yaml` is interpreted relative to the current working directory, which is why the leading `cd /home/xuefeiz2/self/cs336` matters.

### 4. YAML is loaded into a strict `ServeConfig`
**File:** `src/aiinfra_e2e/config.py:75-95`
```python
class ServeConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = 8000
    startup_timeout_seconds: float = 180.0
    model_id: str | None = None
    served_model_name: str | None = None
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9100
    tensor_parallel_size: int = 1
    max_loras: int = 1
    max_lora_rank: int = 64
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    adapters: list[LoRAAdapterConfig] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @property
    def base_url(self) -> str:
        client_host = "127.0.0.1" if self.host in {"0.0.0.0", "::", ""} else self.host
        return f"http://{client_host}:{self.port}"
```

This defines the serve-side contract that the YAML must satisfy. The most subtle field here is `base_url`: even if the server binds to `0.0.0.0`, the wrapper still probes readiness through `127.0.0.1:<port>`.

### 5. YAML parsing and validation are centralized in `load_yaml`
**File:** `src/aiinfra_e2e/config.py:122-154`
```python
def load_yaml(path: str | Path, model_cls: type[ModelT]) -> ModelT:
    """Load a UTF-8 YAML file and validate it into a Pydantic model."""

    config_path = Path(path)
    ...
    loaded = yaml.safe_load(raw_text)
    ...
    return model_cls.model_validate(payload)
```

This is the point where the raw YAML file stops being "just config text" and becomes a checked `ServeConfig` object with defaults, required fields, and forbidden unknown keys.

### 6. The CLI-to-wrapper import hop is explicit
**File:** `src/aiinfra_e2e/serve/__init__.py:1-5`
```python
"""Serving helpers for vLLM OpenAI-compatible endpoints."""

from aiinfra_e2e.serve.vllm_server import ManagedVLLMServer, run_vllm_server_from_config

__all__ = ["ManagedVLLMServer", "run_vllm_server_from_config"]
```

This is a small hop, but it explains how `cli.py` reaches the actual wrapper implementation without importing `vllm_server.py` by a long dotted path each time.

### 7. The wrapper owns the long-running serve lifecycle
**File:** `src/aiinfra_e2e/serve/vllm_server.py:179-190`
```python
def run_vllm_server_from_config(config: ServeConfig) -> None:
    server = ManagedVLLMServer(config)
    try:
        server.start()
        while True:
            time.sleep(1.0)
            assert server.metrics is not None
            server.metrics.update_gpu_memory()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
```

This is the line where the architecture becomes clear: the repo does not just launch vLLM and exit. It stays alive as a wrapper process that owns metrics updates and shutdown.

### 8. `start()` begins with wrapper-side metrics, not child startup
**File:** `src/aiinfra_e2e/serve/vllm_server.py:137-149`
```python
def start(self) -> None:
    if self.process is not None:
        raise RuntimeError("vLLM server is already running.")
    assert self.metrics is not None
    self.metrics.start_server(self.config.metrics_host, self.config.metrics_port)
    self.metrics.update_gpu_memory()
    self.process = subprocess.Popen(
        build_vllm_command(self.config),
        env=build_vllm_environment(self.config),
        text=True,
    )
    self.wait_until_ready(timeout=self.config.startup_timeout_seconds)
```

The wrapper starts its own Prometheus endpoint before the vLLM child is declared ready. That is why your metrics port and your chat-completions port are separate concerns.

### 9. The vLLM child command is built from the validated config
**File:** `src/aiinfra_e2e/serve/vllm_server.py:38-68`
```python
def build_vllm_command(config: ServeConfig) -> list[str]:
    if not config.model_id:
        raise ValueError("Serve config must set model_id to start vLLM.")

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--model",
        config.model_id,
        "--enable-lora",
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-loras",
        str(config.max_loras),
        "--max-lora-rank",
        str(config.max_lora_rank),
    ]
    if config.served_model_name:
        command.extend(["--served-model-name", config.served_model_name])
    if config.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    if config.dtype:
        command.extend(["--dtype", config.dtype])
    if config.extra_args:
        command.extend(config.extra_args)
    return command
```

This is the direct bridge from repo wrapper into the actual vLLM OpenAI server. The repo is not replacing vLLM; it is shaping the child process that launches vLLM.

One subtle point matters here: the child uses `sys.executable`, so if you started with `.venv/bin/python`, the child also uses the repo virtualenv's Python.

### 10. The child environment is intentionally shaped before launch
**File:** `src/aiinfra_e2e/serve/vllm_server.py:24-35`
```python
def build_vllm_environment(config: ServeConfig) -> dict[str, str]:
    del config
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[3]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    )
    env["AIINFRA_E2E_ENABLE_VLLM_TOKENIZER_COMPAT"] = "1"
    env["AIINFRA_E2E_ENABLE_VLLM_TQDM_COMPAT"] = "1"
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    return env
```

This is where your shell-level exports and the repo's own child-level exports meet. Your command line already sets CUDA visibility, Hugging Face cache paths, and proxy bypass in the parent environment; this function preserves those and adds repo-specific child settings on top.

### 11. `sitecustomize.py` is the child startup hook
**File:** `sitecustomize.py:1-12`
```python
import os
from aiinfra_e2e.serve.tokenizer_compat import ensure_all_special_tokens_extended
from aiinfra_e2e.serve.tqdm_compat import ensure_vllm_tqdm_compat

if os.environ.get("AIINFRA_E2E_ENABLE_VLLM_TOKENIZER_COMPAT") == "1":
    ensure_all_special_tokens_extended()

if os.environ.get("AIINFRA_E2E_ENABLE_VLLM_TQDM_COMPAT") == "1":
    ensure_vllm_tqdm_compat()
```

Because the wrapper prepends the repo root to `PYTHONPATH`, Python can import `sitecustomize.py` automatically when the child interpreter starts. This gives the repo one controlled place to patch startup behavior inside the vLLM child.

### 12. Readiness is defined as `/v1/models` responding successfully
**File:** `src/aiinfra_e2e/serve/vllm_server.py:150-166`
```python
def wait_until_ready(self, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if self.process is not None and self.process.poll() is not None:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"vLLM server exited before becoming ready: {stderr}")
        try:
            _ = openai_request(self.config.base_url, "/v1/models", timeout=2.0)
            return
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM server readiness: {last_error}")
```

The wrapper does not treat "subprocess started" as success. It treats **a successful `/v1/models` response** as success.

### 13. Localhost probing deliberately bypasses proxy settings
**File:** `src/aiinfra_e2e/serve/vllm_server.py:81-100`
```python
def _is_localhost_url(base_url: str) -> bool:
    host = parse.urlparse(base_url).hostname
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


def openai_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 5.0,
) -> dict[str, Any]:
    http_request = _json_request(f"{base_url.rstrip('/')}{path}", payload=payload, method=method)
    if _is_localhost_url(base_url):
        opener = request.build_opener(request.ProxyHandler({}))
        response_context = opener.open(http_request, timeout=timeout)
    else:
        response_context = request.urlopen(http_request, timeout=timeout)
```

Your shell command already exported `NO_PROXY` and `no_proxy`, which is good operational hygiene. But the wrapper still adds an extra safety layer: localhost readiness checks explicitly bypass proxy handlers even if the broader environment is messy.

### 14. Wrapper-side metrics are a first-class feature
**File:** `src/aiinfra_e2e/serve/metrics.py:41-43, 51-63`
```python
def start_server(self, host: str, port: int) -> None:
    _ = start_http_server(port, addr=host, registry=self.registry)

def update_gpu_memory(self) -> None:
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    try:
        self.gpu_memory_bytes.set(float(torch.cuda.memory_allocated()))
    except Exception:
        return
```

This is why your config can expose a metrics endpoint like `127.0.0.1:52103` alongside the chat endpoint on `46335`. The metrics endpoint belongs to the wrapper, not to vLLM itself.

### 15. Shutdown is two-stage: terminate first, then kill if needed
**File:** `src/aiinfra_e2e/serve/vllm_server.py:167-177`
```python
def stop(self) -> None:
    if self.process is None:
        return
    self.process.terminate()
    with suppress(subprocess.TimeoutExpired):
        _ = self.process.wait(timeout=10)
    if self.process.poll() is None:
        self.process.kill()
        _ = self.process.wait(timeout=5)
    self.process = None
```

This explains what happens when you stop the wrapper cleanly. The repo tries a graceful child shutdown first, then escalates only if the child refuses to exit.

### 16. Your concrete YAML turns the generic serve path into one specific runtime
**File:** `artifacts/tmp/interactive-serve.yaml:1-13`
```yaml
host: 0.0.0.0
port: 46335
model_id: Qwen/Qwen2.5-7B-Instruct
served_model_name: qwen2.5-7b-instruct
metrics_host: 127.0.0.1
metrics_port: 52103
tensor_parallel_size: 1
max_loras: 4
max_lora_rank: 64
gpu_memory_utilization: 0.9
adapters:
  - name: support-bot
    path: artifacts/adapters/support-bot
```

This file is what makes the abstract serve wrapper become *your* serve instance: Qwen 2.5 7B, port `46335`, metrics `52103`, tensor parallel size `1`, and a named adapter entry.

## Short summary

- Your manual command does **not** run vLLM directly. It enters the repo's `serve` CLI, which loads a strict `ServeConfig` and hands it to a wrapper.
- The wrapper starts its own metrics server, shapes the child process command and environment, and then launches `vllm.entrypoints.openai.api_server` as a subprocess.
- The repo uses `PYTHONPATH` + `sitecustomize.py` to inject child-startup compatibility patches without modifying the installed vLLM package.
- Readiness is defined by a successful `/v1/models` response, not by "the subprocess exists".
- The wrapper stays alive after startup because it owns metrics updates and shutdown; this is why stopping the wrapper process is the clean way to stop the served model.
