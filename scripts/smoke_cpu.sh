#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CLI_BIN=${CLI_BIN:-aiinfra-e2e}

run_cli() {
  if command -v "$CLI_BIN" >/dev/null 2>&1; then
    "$CLI_BIN" "$@"
  else
    "$PYTHON_BIN" -m aiinfra_e2e.cli "$@"
  fi
}

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aiinfra-e2e-smoke.XXXXXX")
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

RUN_ROOT=${CPU_SMOKE_RUN_ROOT:-$TMP_DIR/artifacts/runs}
TRACKING_URI=${CPU_SMOKE_TRACKING_URI:-$TMP_DIR/mlruns}
TRAIN_RUN_NAME=${CPU_SMOKE_RUN_NAME:-cpu-smoke-script}
EVAL_RUN_NAME=${CPU_SMOKE_EVAL_RUN_NAME:-offline-eval-smoke-script}
EVAL_EXPERIMENT=${CPU_SMOKE_EVAL_EXPERIMENT:-cpu-smoke-eval}
ENV_OUT=${CPU_SMOKE_ENV_OUT:-$TMP_DIR/env}

printf '==> Validating checked-in configs\n'
run_cli data --config "$REPO_ROOT/configs/data/alpaca_zh_51k.yaml"
run_cli train --config "$REPO_ROOT/configs/train/qwen2p5_7b_qlora_ddp4.yaml"
run_cli eval --config "$REPO_ROOT/configs/eval/offline.yaml"
run_cli loadtest --config "$REPO_ROOT/configs/serve/loadtest.yaml"
run_cli env report --out "$ENV_OUT"

printf '==> Writing temporary smoke configs\n'
DATA_CONFIG=$TMP_DIR/data.yaml
TRAIN_CONFIG=$TMP_DIR/train.yaml
OBS_CONFIG=$TMP_DIR/obs.yaml
EVAL_CONFIG=$TMP_DIR/eval.yaml

cat > "$DATA_CONFIG" <<EOF
dataset_id: cpu-smoke-inline-dataset
split: train
train_ratio: 0.8
val_ratio: 0.1
split_key_fields:
  - instruction
  - output
EOF

cat > "$TRAIN_CONFIG" <<EOF
model_id: Qwen/Qwen2.5-7B-Instruct
output_dir: $RUN_ROOT
run_name: $TRAIN_RUN_NAME
seed: 13
max_steps: 1
save_steps: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 0.0005
logging_steps: 1
lora_r: 4
lora_alpha: 8
lora_dropout: 0.0
gradient_checkpointing: false
EOF

cat > "$OBS_CONFIG" <<EOF
tracking_uri: $TRACKING_URI
experiment_name: cpu-smoke-train
EOF

cat > "$EVAL_CONFIG" <<EOF
metric: offline
sample_count: 2
output_dir: $RUN_ROOT
run_name: $EVAL_RUN_NAME
golden_prompts:
  - name: concise-answer
    prompt: Say hello politely
    constraints:
      min_length: 8
      must_contain:
        - smoke
  - name: avoid-repetition
    prompt: Explain caching without repeating words too much
    constraints:
      max_repetition_ratio: 0.6
EOF

printf '==> Running minimal CPU training + offline eval pipeline\n'
export CUDA_VISIBLE_DEVICES=
export PYTEST_CURRENT_TEST=${PYTEST_CURRENT_TEST:-scripts/smoke_cpu.sh::cpu_smoke}
export DATA_CONFIG TRAIN_CONFIG OBS_CONFIG EVAL_CONFIG TRACKING_URI TRAIN_RUN_NAME EVAL_EXPERIMENT
"$PYTHON_BIN" - <<'PY'
import os
from datasets import Dataset

import aiinfra_e2e.train.sft as sft_module
from aiinfra_e2e.config import EvalConfig, load_yaml
from aiinfra_e2e.eval.offline import run_offline_eval

dataset = Dataset.from_list(
    [
        {"instruction": "Say hi", "input": "", "output": "hi"},
        {"instruction": "Say bye", "input": "", "output": "bye"},
        {"instruction": "Count", "input": "1 2", "output": "3"},
    ]
)
sft_module.load_hf_dataset = lambda config: dataset
run_dir = sft_module.run_sft_from_paths(
    data_config_path=os.environ["DATA_CONFIG"],
    train_config_path=os.environ["TRAIN_CONFIG"],
    obs_config_path=os.environ["OBS_CONFIG"],
)
assert run_dir.name == os.environ["TRAIN_RUN_NAME"]

eval_config = load_yaml(os.environ["EVAL_CONFIG"], EvalConfig)
result = run_offline_eval(
    eval_config=eval_config,
    obs_tracking_uri=os.environ["TRACKING_URI"],
    obs_experiment_name=os.environ["EVAL_EXPERIMENT"],
    generator=lambda prompt: f"smoke::{prompt}",
)
print(f"CPU smoke train run: {run_dir}")
print(f"CPU smoke eval report: {result.report_path}")
PY

printf 'CPU smoke completed successfully.\n'
