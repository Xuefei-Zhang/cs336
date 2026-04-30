#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

SOURCE_SERVE_CONFIG=${SOURCE_SERVE_CONFIG:-$REPO_ROOT/configs/serve/qwen35_122b_tp2_trace.yaml}
TRACE_ROOT_BASE=${TRACE_ROOT_BASE:-$REPO_ROOT/artifacts/trace}
TRACE_RUN_NAME=${TRACE_RUN_NAME:-$(date +%Y%m%d-%H%M%S)-qwen35-tp2}
TRACE_RUN_DIR=${TRACE_RUN_DIR:-$TRACE_ROOT_BASE/$TRACE_RUN_NAME}
KEEP_SERVE_RUNNING=${KEEP_SERVE_RUNNING:-0}
ENABLE_NSYS=${ENABLE_NSYS:-0}
NSYS_ARGS=${NSYS_ARGS:--t cuda,nvtx,osrt,cudnn,cublas --sample=none --force-overwrite true --trace-fork-before-exec=true}
SERVE_PROMPT=${SERVE_PROMPT:-Say hello in one short sentence.}
SERVE_MAX_TOKENS=${SERVE_MAX_TOKENS:-32}

: "${CUDA_DEVICE_ORDER:=PCI_BUS_ID}"
: "${CUDA_VISIBLE_DEVICES:=2,3}"

HF_DEFAULT_ROOT=$REPO_ROOT/artifacts/hf-home
HF_DEFAULT_CACHE_ROOT=$REPO_ROOT/artifacts/hf-cache

: "${HF_HOME:=$HF_DEFAULT_ROOT}"
: "${HF_HUB_CACHE:=$HF_DEFAULT_CACHE_ROOT/hub}"
: "${HF_DATASETS_CACHE:=$HF_DEFAULT_CACHE_ROOT/datasets}"
: "${TRANSFORMERS_CACHE:=$HF_DEFAULT_CACHE_ROOT/transformers}"
: "${NO_PROXY:=127.0.0.1,localhost}"
: "${no_proxy:=$NO_PROXY}"

mkdir -p "$TRACE_ROOT_BASE" "$TRACE_RUN_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aiinfra-e2e-serve-trace.XXXXXX")
EFFECTIVE_SERVE_CONFIG=$TRACE_RUN_DIR/serve.effective.yaml
SERVE_LOG_PATH=$TRACE_RUN_DIR/serve.log
GPU_LOG_PATH=$TRACE_RUN_DIR/gpu.csv
ENV_LOG_PATH=$TRACE_RUN_DIR/env.txt
MODELS_PATH=$TRACE_RUN_DIR/models.json
HEALTH_PATH=$TRACE_RUN_DIR/health.json
METRICS_PATH=$TRACE_RUN_DIR/metrics.prom
SMOKE_CHAT_PATH=$TRACE_RUN_DIR/smoke_chat.json
TRACE_SUMMARY_PATH=$TRACE_RUN_DIR/summary.txt
NSYS_OUTPUT_DIR=$TRACE_RUN_DIR/nsys

SERVE_PID=''
GPU_POLL_PID=''

cleanup() {
  if [[ -n "$GPU_POLL_PID" ]] && kill -0 "$GPU_POLL_PID" >/dev/null 2>&1; then
    kill "$GPU_POLL_PID" || true
    wait "$GPU_POLL_PID" || true
  fi
  if [[ "$KEEP_SERVE_RUNNING" != "1" ]] && [[ -n "$SERVE_PID" ]] && kill -0 "$SERVE_PID" >/dev/null 2>&1; then
    kill "$SERVE_PID" || true
    wait "$SERVE_PID" || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

run_cli() {
  "$PYTHON_BIN" -m aiinfra_e2e.cli "$@"
}

printf '==> Writing trace environment snapshot\n'
export CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE NO_PROXY no_proxy
export SOURCE_SERVE_CONFIG EFFECTIVE_SERVE_CONFIG TRACE_RUN_DIR SERVE_PROMPT SERVE_MAX_TOKENS
env | sort > "$ENV_LOG_PATH"
run_cli env report --out "$TRACE_RUN_DIR/env"

printf '==> Rendering effective serve config\n'
"$PYTHON_BIN" - <<'PY'
import os

from aiinfra_e2e.serve.trace_run import render_effective_serve_config

config = render_effective_serve_config(
    os.environ["SOURCE_SERVE_CONFIG"],
    os.environ["EFFECTIVE_SERVE_CONFIG"],
)
print(f"Effective serve config: {os.environ['EFFECTIVE_SERVE_CONFIG']}")
print(f"Serve base URL: {config.base_url}")
print(f"Metrics endpoint: http://{config.metrics_host}:{config.metrics_port}/metrics")
PY

printf '==> Starting GPU polling\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    printf 'timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw\n'
    while true; do
      timestamp=$(date '+%F %T')
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits \
        | while IFS= read -r line; do
            printf '%s,%s\n' "$timestamp" "$line"
          done
      sleep 1
    done
  ) > "$GPU_LOG_PATH" 2>&1 &
  GPU_POLL_PID=$!
else
  printf 'nvidia-smi not found; skipping GPU polling.\n' | tee -a "$TRACE_SUMMARY_PATH"
fi

printf '==> Starting serve wrapper\n'
mkdir -p "$NSYS_OUTPUT_DIR"

if [[ "${ENABLE_NSYS:-0}" == "1" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    printf 'ENABLE_NSYS=1 but nsys is not installed.\n' >&2
    exit 1
  fi
  nsys profile $NSYS_ARGS --output "$NSYS_OUTPUT_DIR/vllm" \
    "$PYTHON_BIN" -m aiinfra_e2e.cli serve --config "$EFFECTIVE_SERVE_CONFIG" \
    2>&1 | tee "$SERVE_LOG_PATH" &
else
  "$PYTHON_BIN" -m aiinfra_e2e.cli serve --config "$EFFECTIVE_SERVE_CONFIG" \
    2>&1 | tee "$SERVE_LOG_PATH" &
fi
SERVE_PID=$!

printf '==> Probing health and readiness\n'
"$PYTHON_BIN" - <<'PY'
import os
import time

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.trace_run import probe_health, probe_models

config = load_yaml(os.environ["EFFECTIVE_SERVE_CONFIG"], ServeConfig)
deadline = time.time() + config.startup_timeout_seconds
last_error = None
while time.time() < deadline:
    health = probe_health(f"{config.base_url}/health", os.environ["HEALTH_PATH"], timeout=2.0)
    try:
        probe_models(config.base_url, os.environ["MODELS_PATH"], timeout=5.0)
        print(f"Ready via /v1/models at {config.base_url}; health_status={health['status']}")
        break
    except Exception as exc:  # pragma: no cover - script runtime path
        last_error = exc
        time.sleep(2.0)
else:
    raise TimeoutError(f"Timed out waiting for vLLM readiness: {last_error}")
PY

printf '==> Capturing smoke request and metrics\n'
"$PYTHON_BIN" - <<'PY'
import os

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.trace_run import scrape_metrics, send_smoke_chat_completion

config = load_yaml(os.environ["EFFECTIVE_SERVE_CONFIG"], ServeConfig)
model_name = config.served_model_name or config.model_id
if model_name is None:
    raise ValueError("Serve config must define served_model_name or model_id")

send_smoke_chat_completion(
    config.base_url,
    model=model_name,
    prompt=os.environ["SERVE_PROMPT"],
    max_tokens=int(os.environ["SERVE_MAX_TOKENS"]),
    output_path=os.environ["SMOKE_CHAT_PATH"],
)
scrape_metrics(
    f"http://{config.metrics_host}:{config.metrics_port}/metrics",
    os.environ["METRICS_PATH"],
)
print(f"Smoke request completed for model={model_name}")
PY

cat > "$TRACE_SUMMARY_PATH" <<EOF
trace_run_dir=$TRACE_RUN_DIR
effective_serve_config=$EFFECTIVE_SERVE_CONFIG
serve_log=$SERVE_LOG_PATH
gpu_log=$GPU_LOG_PATH
health_json=$HEALTH_PATH
models_json=$MODELS_PATH
metrics_prom=$METRICS_PATH
smoke_chat_json=$SMOKE_CHAT_PATH
enable_nsys=$ENABLE_NSYS
keep_serve_running=$KEEP_SERVE_RUNNING
EOF

printf 'Trace bundle written to %s\n' "$TRACE_RUN_DIR"
