# repo structure map

## Goal

This document answers a simple question:

> If I want to understand this repo quickly, which files matter, and what role does each one play?

## The high-level layout

- `Makefile` — outer developer entrypoints (`make test`, `make smoke`, `make e2e`, `make serve`, `make loadtest`)
- `scripts/` — shell orchestration and Locust runner files
- `configs/` — YAML configuration for data, train, eval, serve, and loadtest
- `src/aiinfra_e2e/` — actual Python implementation
- `tests/` — regression coverage for runtime behavior and host-specific fixes
- `docs/reviews/` — generated understanding docs for humans

## Most important files by role

### 1. Top-level orchestration

#### `Makefile`
Use this file when you want to know:
- what the intended entrypoints are
- which commands are considered official
- how the project pins itself to `.venv/bin/python`

#### `scripts/e2e_gpu.sh`
Use this file when you want to know:
- what `make e2e` actually does
- how host adaptation works
- where effective configs, cache env vars, readiness polling, and loadtest invocation live

#### `scripts/smoke_cpu.sh`
Use this file when you want to understand the CPU-only demo path.

### 2. Config schema and validation

#### `src/aiinfra_e2e/config.py`
This is the contract layer between YAML files and runtime code. It tells you:
- what fields each stage expects
- which defaults are checked in
- how wildcard serve hosts are normalized to a client-safe `base_url`

### 3. CLI dispatch layer

#### `src/aiinfra_e2e/cli.py`
This file answers:
- which commands are real
- which commands are mostly config validators
- which CLI command actually jumps into training or serving code

A useful mental model is:
- `data` / `eval` / `loadtest` commands are currently lightweight validators
- `train sft` and `serve` are the real runtime dispatchers

### 4. Data stage

#### `src/aiinfra_e2e/data/hf_sync.py`
Loads datasets from Hugging Face with bounded retry behavior.

#### `src/aiinfra_e2e/data/preprocess.py`
Turns Alpaca-style rows into chat-formatted SFT records with assistant-only labels.

### 5. Training stage

#### `src/aiinfra_e2e/train/sft.py`
This is the most important implementation file in the repo.

It owns:
- YAML-to-runtime training dispatch
- dataset loading and split preparation
- tokenizer/model setup
- TRL trainer invocation
- artifact writing
- MLflow train logging
- smoke-mode behavior
- checkpoint resume / OOM fallback integration

#### `src/aiinfra_e2e/train/trl_compat.py`
This file exists because the host environment is not perfectly clean. It protects the training import path from broken `torchvision` behavior when loading TRL.

### 6. Evaluation stage

#### `src/aiinfra_e2e/eval/offline.py`
This file runs a simple offline evaluation pass, writes `eval_report.json`, and logs it to MLflow.

### 7. Serving stage

#### `src/aiinfra_e2e/serve/vllm_server.py`
This is the serving control center.

It owns:
- vLLM subprocess command construction
- child environment setup
- subprocess compatibility flags
- readiness polling
- localhost proxy bypass behavior
- wrapper lifetime management

#### `sitecustomize.py`
#### `src/aiinfra_e2e/serve/tokenizer_compat.py`
#### `src/aiinfra_e2e/serve/tqdm_compat.py`
These files support the vLLM subprocess at import/runtime startup. They are not generic app logic; they are targeted compatibility shims.

### 8. Load testing stage

#### `scripts/loadtest_locust.py`
The actual Locust task definition used during load testing.

#### `src/aiinfra_e2e/loadtest.py`
Artifact naming, payload construction, and MLflow logging helpers for the loadtest stage.

### 9. Host adaptation helpers

#### `src/aiinfra_e2e/gpu.py`
Small but important file that chooses a free GPU for shared-host runs.

## Config files that matter most for `make e2e`

- `configs/data/alpaca_zh_51k.yaml`
- `configs/train/qwen2p5_7b_qlora_ddp4.yaml`
- `configs/eval/offline.yaml`
- `configs/serve/vllm_openai_lora.yaml`
- `configs/serve/loadtest.yaml`

These are the source configs. `scripts/e2e_gpu.sh` may generate **effective** runtime versions for serve and loadtest, but these checked-in YAML files are the starting point.

## Tests worth reading first

- `tests/test_openai_api_smoke.py` — serving wrapper behavior and compatibility fixes
- `tests/test_train_smoke_cpu.py` — smoke-mode and SFT runtime expectations
- `tests/test_resume_checkpoint.py` — checkpoint/resume training behavior
- `tests/test_trl_compat.py` — TRL import compatibility contract
- `tests/test_script_entrypoints.py` — shell scripts must honor `PYTHON_BIN`
- `tests/test_gpu_selection.py` — shared-host GPU selection behavior

## Suggested study order

If you want the fastest path to understanding, read in this order:

1. `Makefile`
2. `scripts/e2e_gpu.sh`
3. `src/aiinfra_e2e/config.py`
4. `src/aiinfra_e2e/cli.py`
5. `src/aiinfra_e2e/train/sft.py`
6. `src/aiinfra_e2e/serve/vllm_server.py`
7. `src/aiinfra_e2e/loadtest.py`
8. the tests listed above

## Short takeaway

This repo is best understood as a **demo-oriented orchestration repo** that wraps several Python subsystems with a practical shell driver designed for imperfect shared GPU hosts.
