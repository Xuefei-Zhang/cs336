# inference metrics codepath review

## What is being traced

This review traces two concrete outputs observed on the live system:

1. `cat artifacts/runs/serve-loadtest/locust_report.json | jq`
2. `curl http://localhost:52103/metrics`

The goal is to map those observed values back to the exact codepaths that generated them.

## Ordered codepath

### A. `locust_report.json`
1. `make loadtest` loads `LoadTestConfig` and resolves canonical report paths.
2. `make loadtest` launches `python -m locust -f scripts/loadtest_locust.py ...`.
3. `scripts/loadtest_locust.py` registers extra arguments such as `--model`, `--prompt`, and `--json-report`.
4. `OpenAIChatUser.chat_completion()` repeatedly POSTs to `/v1/chat/completions`.
5. On Locust shutdown, `_write_json_report()` reads aggregated stats from `environment.stats.total`.
6. `_write_json_report()` writes those stats to `artifacts/runs/serve-loadtest/locust_report.json`.
7. Separately, `src/aiinfra_e2e/loadtest.py::log_loadtest_reports()` logs the HTML/JSON report files and metadata to MLflow.

### B. `/metrics`
1. `aiinfra_e2e.cli serve --config ...` calls `run_vllm_server_from_config()`.
2. `run_vllm_server_from_config()` constructs `ManagedVLLMServer`.
3. `ManagedVLLMServer.__post_init__()` creates a `ServeMetrics` object.
4. `ManagedVLLMServer.start()` calls `self.metrics.start_server(host, port)`.
5. `ServeMetrics.start_server()` starts a Prometheus HTTP metrics endpoint.
6. `ManagedVLLMServer.start()` and the outer serve loop call `self.metrics.update_gpu_memory()`.
7. The counters and gauges exposed by `ServeMetrics` become the text returned by `curl http://localhost:52103/metrics`.

## Snippets

### 1. `make loadtest` builds the Locust run
**File:** `Makefile:49-59`
```make
loadtest: ensure-venv
	@if ! $(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('locust') else 1)"; then \
		printf 'Skipping loadtest: locust is not installed in the active Python environment.\n'; \
		exit 0; \
	fi
	@if [ -z "$(LOADTEST_HOST)" ]; then \
		printf 'Skipping loadtest: set LOADTEST_HOST to the target base URL ...\n'; \
		exit 0; \
	fi
	@LOADTEST_HOST="$(LOADTEST_HOST)" ... $(PYTHON) -c '... config = load_yaml("$(LOADTEST_CONFIG)", LoadTestConfig); artifacts = resolve_loadtest_artifacts(config); ... command = [sys.executable, "-m", "locust", "-f", "scripts/loadtest_locust.py", ... "--html", ... "--json-report", ...]; subprocess.run(command, check=True)'
```

This is the outer entrypoint for the JSON report you read. It decides the target host, invokes Locust, and points Locust at the canonical HTML/JSON output paths.

### 2. Artifact file naming is centralized in `loadtest.py`
**File:** `src/aiinfra_e2e/loadtest.py:24-38`
```python
def resolve_loadtest_run_id(config: LoadTestConfig) -> str:
    if config.run_name:
        return config.run_name
    return f"loadtest-{config.users}u"


def resolve_loadtest_artifacts(config: LoadTestConfig) -> LoadTestArtifacts:
    run_id = resolve_loadtest_run_id(config)
    run_dir = Path(config.output_dir) / run_id
    return LoadTestArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        html_report_path=run_dir / "locust_report.html",
        json_report_path=run_dir / "locust_report.json",
    )
```

This explains why the file lives at `artifacts/runs/serve-loadtest/locust_report.json`.

### 3. Payload fields come from repo-owned loadtest policy
**File:** `src/aiinfra_e2e/loadtest.py:41-49`
```python
def build_chat_completions_payload(config: LoadTestConfig) -> dict[str, object]:
    model = config.serve.served_model_name or config.serve.model_id
    if model is None:
        raise ValueError("Load test config must define serve.served_model_name or serve.model_id")
    return {
        "model": model,
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
    }
```

This is why the benchmark traffic uses the same OpenAI-style chat payload shape as your manual `curl` requests.

### 4. Locust defines the actual benchmark traffic
**File:** `scripts/loadtest_locust.py:51-57, 83-103`
```python
def _build_payload(environment: Environment) -> dict[str, object]:
    options = environment.parsed_options
    return {
        "model": options.model,
        "messages": [{"role": "user", "content": options.prompt}],
        "max_tokens": options.max_tokens,
    }
```

```python
class OpenAIChatUser(HttpUser):
    wait_time = between(0.5, 1.5)
    host = os.environ.get("LOADTEST_HOST", "http://127.0.0.1:8000")

    @task
    def chat_completion(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json=_build_payload(self.environment),
            name="POST /v1/chat/completions",
        )
        if response.status_code >= 400:
            response.failure(f"HTTP {response.status_code}: {response.text}")
            return
        ...
        if not payload.get("choices"):
            response.failure("Response did not include choices")
```

This proves that `endpoint: "/v1/chat/completions"` in your JSON report is not a post-hoc label; it is the real request path being benchmarked.

### 5. The JSON report fields are written explicitly on quitting
**File:** `scripts/loadtest_locust.py:106-128`
```python
@events.quitting.add_listener
def _write_json_report(environment: Environment, **_: object) -> None:
    ...
    total = environment.stats.total
    payload = {
        "target_host": environment.host,
        "endpoint": "/v1/chat/completions",
        "requests": total.num_requests,
        "failures": total.num_failures,
        "median_response_time": total.median_response_time,
        "average_response_time": total.avg_response_time,
        "min_response_time": total.min_response_time,
        "max_response_time": total.max_response_time,
        "requests_per_second": total.current_rps,
        "fail_ratio": total.fail_ratio,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

This is the direct source of the exact fields you saw:
- `average_response_time`
- `endpoint`
- `fail_ratio`
- `failures`
- `max_response_time`
- `median_response_time`
- `min_response_time`
- `requests`
- `requests_per_second`
- `target_host`

### 6. The observed JSON report matches the code-defined shape
**File:** `artifacts/runs/serve-loadtest/locust_report.json:1-12`
```json
{
  "average_response_time": 48.719783159048035,
  "endpoint": "/v1/chat/completions",
  "fail_ratio": 0.0,
  "failures": 0,
  "max_response_time": 64.2473567277193,
  "median_response_time": 48,
  "min_response_time": 43.79459749907255,
  "requests": 223,
  "requests_per_second": 3.9,
  "target_host": "http://127.0.0.1:46335"
}
```

This is a clean example of repo-generated benchmark output, not an incidental Locust side effect.

### 7. MLflow logging is a second-stage consumer of the same artifacts
**File:** `src/aiinfra_e2e/loadtest.py:52-72`
```python
def log_loadtest_reports(*, config: LoadTestConfig, artifacts: LoadTestArtifacts) -> None:
    with start_mlflow_run(
        tracking_uri=config.obs.tracking_uri,
        experiment_name=config.obs.experiment_name,
        run_name=artifacts.run_id,
    ):
        _ = mlflow.log_params(
            {
                "endpoint": CHAT_COMPLETIONS_ENDPOINT,
                "html_report": str(artifacts.html_report_path),
                "json_report": str(artifacts.json_report_path),
                "target_host": config.serve.base_url,
                "users": str(config.users),
                "spawn_rate": str(config.spawn_rate),
                "run_time": config.run_time,
            }
        )
        if artifacts.html_report_path.exists():
            mlflow.log_artifact(str(artifacts.html_report_path))
        if artifacts.json_report_path.exists():
            mlflow.log_artifact(str(artifacts.json_report_path))
```

This is why the loadtest stage is more than a transient benchmark: the repo turns the benchmark outputs into persistent experiment artifacts.

### 8. The `/metrics` endpoint is created by the serve wrapper, not by Locust
**File:** `src/aiinfra_e2e/serve/vllm_server.py:127-148`
```python
@dataclass
class ManagedVLLMServer:
    config: ServeConfig
    metrics: ServeMetrics | None = None
    process: subprocess.Popen[str] | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = ServeMetrics()

    def start(self) -> None:
        ...
        self.metrics.start_server(self.config.metrics_host, self.config.metrics_port)
        self.metrics.update_gpu_memory()
        self.process = subprocess.Popen(...)
        self.wait_until_ready(timeout=self.config.startup_timeout_seconds)
```

This explains why `curl http://localhost:52103/metrics` works even though vLLM itself is serving on another port. The metrics endpoint belongs to the wrapper process.

### 9. The metrics names come from `ServeMetrics`
**File:** `src/aiinfra_e2e/serve/metrics.py:15-39`
```python
class ServeMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.adapter_load_total = Counter(
            "adapter_load_total",
            "Total number of LoRA adapter loads requested through the wrapper.",
            registry=self.registry,
        )
        ...
        self.request_fail_total = Counter(
            "request_fail_total",
            "Total number of failed wrapper HTTP requests.",
            registry=self.registry,
        )
        self.gpu_memory_bytes = Gauge(
            "wrapper_gpu_memory_bytes",
            "Best-effort GPU memory allocated by torch.",
            registry=self.registry,
        )
```

This is the direct source of the metrics names you observed.

### 10. The observed metrics lines match the declared collectors
**Observed output:** `curl http://localhost:52103/metrics`
```text
# HELP adapter_load_total Total number of LoRA adapter loads requested through the wrapper.
adapter_load_total 0.0
# HELP request_fail_total Total number of failed wrapper HTTP requests.
request_fail_total 0.0
# HELP wrapper_gpu_memory_bytes Best-effort GPU memory allocated by torch.
wrapper_gpu_memory_bytes 0.0
```

These values mean:
- `adapter_load_total=0.0`: no dynamic adapter load requests have been issued through the wrapper yet.
- `request_fail_total=0.0`: the wrapper has not recorded any failed wrapper-side HTTP requests.
- `wrapper_gpu_memory_bytes=0.0`: the wrapper process itself is not observing active torch-allocated GPU memory, which does **not** mean the vLLM engine core is not using GPU memory.

### 11. Why `wrapper_gpu_memory_bytes` can be zero while vLLM is still serving
**File:** `src/aiinfra_e2e/serve/metrics.py:51-63`
```python
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

This gauge measures `torch.cuda.memory_allocated()` in the wrapper process, not the full memory footprint of every vLLM child process. That is why `nvidia-smi` can show heavy vLLM GPU usage while this wrapper-side gauge still reads `0.0`.

## Short summary

- `locust_report.json` is generated by the Locust quitting hook in `scripts/loadtest_locust.py`, after `make loadtest` launches a headless benchmark run.
- The exact fields in your JSON file are explicitly assembled from `environment.stats.total` and written by repo code.
- The `/metrics` endpoint is created by `ManagedVLLMServer.start()` through `ServeMetrics.start_server()`; it is wrapper-side telemetry, not a generic dump of full vLLM internals.
- Your observed outputs cleanly map to two separate but complementary instrumentation layers: benchmark results (`locust_report.json`) and live wrapper metrics (`/metrics`).
