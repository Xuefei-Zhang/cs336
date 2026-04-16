#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

DATA_CONFIG=${DATA_CONFIG:-$REPO_ROOT/configs/data/alpaca_zh_51k.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-$REPO_ROOT/configs/train/qwen2p5_7b_qlora_ddp4.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-$REPO_ROOT/configs/eval/offline.yaml}
SERVE_CONFIG=${SERVE_CONFIG:-$REPO_ROOT/configs/serve/vllm_openai_lora.yaml}
LOADTEST_CONFIG=${LOADTEST_CONFIG:-$REPO_ROOT/configs/serve/loadtest.yaml}
ORIGINAL_SERVE_CONFIG=$SERVE_CONFIG
ORIGINAL_LOADTEST_CONFIG=$LOADTEST_CONFIG
OBS_CONFIG=${OBS_CONFIG:-}
OBS_TRACKING_URI=${OBS_TRACKING_URI:-mlruns}
OBS_EXPERIMENT_NAME=${OBS_EXPERIMENT_NAME:-gpu-e2e-demo}
HF_DEFAULT_ROOT=$REPO_ROOT/artifacts/hf-home
HF_DEFAULT_CACHE_ROOT=$REPO_ROOT/artifacts/hf-cache

: "${HF_HOME:=$HF_DEFAULT_ROOT}"
: "${HF_HUB_CACHE:=$HF_DEFAULT_CACHE_ROOT/hub}"
: "${HF_DATASETS_CACHE:=$HF_DEFAULT_CACHE_ROOT/datasets}"
: "${TRANSFORMERS_CACHE:=$HF_DEFAULT_CACHE_ROOT/transformers}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

run_cli() {
  "$PYTHON_BIN" -m aiinfra_e2e.cli "$@"
}

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aiinfra-e2e-gpu.XXXXXX")
EFFECTIVE_SERVE_CONFIG=$TMP_DIR/serve.effective.yaml
EFFECTIVE_LOADTEST_CONFIG=$TMP_DIR/loadtest.effective.yaml
SERVE_PID=''
cleanup() {
  if [[ -n "$SERVE_PID" ]] && kill -0 "$SERVE_PID" >/dev/null 2>&1; then
    kill "$SERVE_PID"
    wait "$SERVE_PID" || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ -z "$OBS_CONFIG" ]]; then
  OBS_CONFIG=$TMP_DIR/obs.yaml
  cat > "$OBS_CONFIG" <<EOF
tracking_uri: $OBS_TRACKING_URI
experiment_name: $OBS_EXPERIMENT_NAME
EOF
fi

export DATA_CONFIG TRAIN_CONFIG EVAL_CONFIG SERVE_CONFIG LOADTEST_CONFIG OBS_CONFIG REPO_ROOT PYTHON_BIN
export ORIGINAL_SERVE_CONFIG ORIGINAL_LOADTEST_CONFIG EFFECTIVE_SERVE_CONFIG EFFECTIVE_LOADTEST_CONFIG
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE

printf '==> GPU selection\n'
CUDA_SELECTION=$(
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}" "$PYTHON_BIN" - <<'PY'
import os

from aiinfra_e2e.gpu import select_cuda_visible_devices

selected = select_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
if selected is not None:
    print(selected)
PY
)
if [[ -n "$CUDA_SELECTION" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_SELECTION"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
else
  printf 'CUDA_VISIBLE_DEVICES remains unset\n'
fi

printf '==> Preparing effective serve/loadtest configs\n'
"$PYTHON_BIN" - <<'PY'
import os
import socket
from pathlib import Path

import yaml

from aiinfra_e2e.config import LoadTestConfig, ServeConfig, load_yaml


def _port_is_in_use(host: str, port: int) -> bool:
    probe_host = host
    if probe_host in {"0.0.0.0", "::", ""}:
        probe_host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((probe_host, port))
        except OSError:
            return True
    return False


def _pick_free_local_port(excluded: set[int]) -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port


serve_config = load_yaml(os.environ["ORIGINAL_SERVE_CONFIG"], ServeConfig)
loadtest_config = load_yaml(os.environ["ORIGINAL_LOADTEST_CONFIG"], LoadTestConfig)

selected_serve_port = serve_config.port
if _port_is_in_use(serve_config.host, serve_config.port):
    selected_serve_port = _pick_free_local_port(set())
    print(
        f"Configured serve port {serve_config.port} is busy; using free localhost port {selected_serve_port}"
    )
else:
    print(f"Configured serve port {serve_config.port} is available; using it")

selected_metrics_port = serve_config.metrics_port
if selected_metrics_port == selected_serve_port or _port_is_in_use(
    serve_config.metrics_host, serve_config.metrics_port
):
    selected_metrics_port = _pick_free_local_port({selected_serve_port})
    print(
        f"Configured metrics port {serve_config.metrics_port} is unavailable; using free localhost port {selected_metrics_port}"
    )
else:
    print(f"Configured metrics port {serve_config.metrics_port} is available; using it")

serve_payload = serve_config.model_dump(mode="python")
serve_payload["port"] = selected_serve_port
serve_payload["metrics_port"] = selected_metrics_port

loadtest_payload = loadtest_config.model_dump(mode="python")
loadtest_payload["serve"]["port"] = selected_serve_port

serve_output_path = Path(os.environ["EFFECTIVE_SERVE_CONFIG"])
loadtest_output_path = Path(os.environ["EFFECTIVE_LOADTEST_CONFIG"])
serve_output_path.write_text(yaml.safe_dump(serve_payload, sort_keys=False), encoding="utf-8")
loadtest_output_path.write_text(yaml.safe_dump(loadtest_payload, sort_keys=False), encoding="utf-8")

print(f"Effective serve config: {serve_output_path}")
print(f"Effective loadtest config: {loadtest_output_path}")
print(f"Selected serve endpoint: http://127.0.0.1:{selected_serve_port}")
print(f"Selected wrapper metrics endpoint: http://127.0.0.1:{selected_metrics_port}")
PY

printf '==> Hugging Face cache paths\n'
printf 'HF_HOME=%s\nHF_HUB_CACHE=%s\nHF_DATASETS_CACHE=%s\nTRANSFORMERS_CACHE=%s\n' \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

printf '==> Validating configs\n'
run_cli data --config "$DATA_CONFIG"
run_cli train --config "$TRAIN_CONFIG"
run_cli eval --config "$EVAL_CONFIG"
run_cli loadtest --config "$EFFECTIVE_LOADTEST_CONFIG"
"$PYTHON_BIN" - <<'PY'
import os
from aiinfra_e2e.config import ObsConfig, ServeConfig, load_yaml

load_yaml(os.environ["EFFECTIVE_SERVE_CONFIG"], ServeConfig)
load_yaml(os.environ["OBS_CONFIG"], ObsConfig)
print(f"Loaded config from {os.environ['EFFECTIVE_SERVE_CONFIG']}")
print(f"Loaded config from {os.environ['OBS_CONFIG']}")
PY

printf '==> Data sync\n'
"$PYTHON_BIN" - <<'PY'
import os
from aiinfra_e2e.config import DataConfig, load_yaml
from aiinfra_e2e.data.hf_sync import load_hf_dataset

config = load_yaml(os.environ["DATA_CONFIG"], DataConfig)
dataset = load_hf_dataset(config)
print(f"Synced dataset {config.dataset_id} ({type(dataset).__name__})")
PY

printf '==> Preprocess sample\n'
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from transformers import AutoTokenizer

from aiinfra_e2e.config import DataConfig, TrainConfig, load_yaml
from aiinfra_e2e.data.hf_sync import load_hf_dataset
from aiinfra_e2e.data.preprocess import preprocess_record

data_config = load_yaml(os.environ["DATA_CONFIG"], DataConfig)
train_config = load_yaml(os.environ["TRAIN_CONFIG"], TrainConfig)
dataset = load_hf_dataset(data_config)
record = dataset[0]
cache_dir = Path(data_config.cache_dir or (Path(train_config.output_dir).parent / "hf-cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(train_config.model_id, cache_dir=str(cache_dir))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
processed = preprocess_record(
    record,
    tokenizer=tokenizer,
    instruction_field=data_config.prompt_field or "instruction",
    output_field=data_config.response_field or "output",
)
print(
    f"Preprocessed sample: text_chars={len(processed['text'])}, tokens={len(processed['input_ids'])}"
)
PY

printf '==> Train\n'
run_cli train sft --data-config "$DATA_CONFIG" --train-config "$TRAIN_CONFIG" --obs-config "$OBS_CONFIG"

printf '==> Offline eval\n'
"$PYTHON_BIN" - <<'PY'
import os

from aiinfra_e2e.config import EvalConfig, ObsConfig, load_yaml
from aiinfra_e2e.eval.offline import run_offline_eval

eval_config = load_yaml(os.environ["EVAL_CONFIG"], EvalConfig)
obs_config = load_yaml(os.environ["OBS_CONFIG"], ObsConfig)
result = run_offline_eval(
    eval_config=eval_config,
    obs_tracking_uri=obs_config.tracking_uri,
    obs_experiment_name=obs_config.experiment_name,
    generator=lambda prompt: f"offline-eval::{prompt}",
)
print(f"Offline eval report: {result.report_path}")
PY

printf '==> Serve\n'
run_cli serve --config "$EFFECTIVE_SERVE_CONFIG" &
SERVE_PID=$!
"$PYTHON_BIN" - <<'PY'
import os
import time

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.vllm_server import openai_request

config = load_yaml(os.environ["EFFECTIVE_SERVE_CONFIG"], ServeConfig)
deadline = time.time() + 120
last_error = None
while time.time() < deadline:
    try:
        openai_request(config.base_url, "/v1/models", timeout=5.0)
        print(f"Serve endpoint ready: {config.base_url}")
        break
    except Exception as exc:  # pragma: no cover - script-only polling path
        last_error = exc
        time.sleep(2.0)
else:
    raise TimeoutError(f"Timed out waiting for serve endpoint readiness: {last_error}")
PY

printf '==> Loadtest\n'
"$PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys

from aiinfra_e2e.config import LoadTestConfig, load_yaml
from aiinfra_e2e.loadtest import log_loadtest_reports, resolve_loadtest_artifacts

config = load_yaml(os.environ["EFFECTIVE_LOADTEST_CONFIG"], LoadTestConfig)
artifacts = resolve_loadtest_artifacts(config)
artifacts.run_dir.mkdir(parents=True, exist_ok=True)
model_name = config.serve.served_model_name or config.serve.model_id
if model_name is None:
    raise ValueError("Load test config must define serve.served_model_name or serve.model_id")

command = [
    sys.executable,
    "-m",
    "locust",
    "-f",
    os.path.join(os.environ["REPO_ROOT"], "scripts", "loadtest_locust.py"),
    "--headless",
    "--host",
    config.serve.base_url,
    "-u",
    str(config.users),
    "-r",
    str(config.spawn_rate),
    "-t",
    config.run_time,
    "--model",
    model_name,
    "--prompt",
    config.prompt,
    "--max-tokens",
    str(config.max_tokens),
    "--html",
    str(artifacts.html_report_path),
    "--json-report",
    str(artifacts.json_report_path),
]
subprocess.run(command, check=True)
log_loadtest_reports(config=config, artifacts=artifacts)
print(f"Loadtest reports: {artifacts.html_report_path}, {artifacts.json_report_path}")
PY

printf 'GPU E2E demo completed successfully.\n'
