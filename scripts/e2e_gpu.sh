#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CLI_BIN=${CLI_BIN:-aiinfra-e2e}

DATA_CONFIG=${DATA_CONFIG:-$REPO_ROOT/configs/data/alpaca_zh_51k.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-$REPO_ROOT/configs/train/qwen2p5_7b_qlora_ddp4.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-$REPO_ROOT/configs/eval/offline.yaml}
SERVE_CONFIG=${SERVE_CONFIG:-$REPO_ROOT/configs/serve/vllm_openai_lora.yaml}
LOADTEST_CONFIG=${LOADTEST_CONFIG:-$REPO_ROOT/configs/serve/loadtest.yaml}
OBS_CONFIG=${OBS_CONFIG:-}
OBS_TRACKING_URI=${OBS_TRACKING_URI:-mlruns}
OBS_EXPERIMENT_NAME=${OBS_EXPERIMENT_NAME:-gpu-e2e-demo}

run_cli() {
  if command -v "$CLI_BIN" >/dev/null 2>&1; then
    "$CLI_BIN" "$@"
  else
    "$PYTHON_BIN" -m aiinfra_e2e.cli "$@"
  fi
}

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aiinfra-e2e-gpu.XXXXXX")
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

printf '==> Validating configs\n'
run_cli data --config "$DATA_CONFIG"
run_cli train --config "$TRAIN_CONFIG"
run_cli eval --config "$EVAL_CONFIG"
run_cli loadtest --config "$LOADTEST_CONFIG"
"$PYTHON_BIN" - <<'PY'
import os
from aiinfra_e2e.config import ObsConfig, ServeConfig, load_yaml

load_yaml(os.environ["SERVE_CONFIG"], ServeConfig)
load_yaml(os.environ["OBS_CONFIG"], ObsConfig)
print(f"Loaded config from {os.environ['SERVE_CONFIG']}")
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
run_cli serve --config "$SERVE_CONFIG" &
SERVE_PID=$!
"$PYTHON_BIN" - <<'PY'
import os
import time

from aiinfra_e2e.config import ServeConfig, load_yaml
from aiinfra_e2e.serve.vllm_server import openai_request

config = load_yaml(os.environ["SERVE_CONFIG"], ServeConfig)
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

config = load_yaml(os.environ["LOADTEST_CONFIG"], LoadTestConfig)
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
