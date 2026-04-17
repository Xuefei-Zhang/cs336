# aiinfra-e2e

Minimal scaffold for the AIInfra E2E LLM LoRA/QLoRA project.

## Setup

```bash
make setup
```

## Lint

```bash
make lint
```

## Tests

```bash
make test
```

## Interview Demo

### Prerequisites

- Run `make setup` first so `aiinfra-e2e` and the Python dependencies are available.
- `scripts/smoke_cpu.sh` is CPU-only and self-contained; it validates the checked-in configs, writes a temporary env report, runs a one-step CPU SFT smoke path, and writes an offline eval report.
- `scripts/e2e_gpu.sh` is for a GPU host with CUDA, an NVIDIA runtime, `vllm`, and `locust` available in the active Python environment.
- The default GPU configs point at real Hugging Face assets. If you are using gated weights, export the required Hugging Face credentials first; if you want different paths, override the config env vars shown below instead of editing the script.

### CPU smoke demo

```bash
bash scripts/smoke_cpu.sh
```

### GPU end-to-end demo

Optional: start the observability stack first so MLflow, Prometheus, and Grafana are live during the demo.

```bash
make obs-up
```

Run the full GPU playbook with env-overridable config paths:

```bash
DATA_CONFIG=configs/data/alpaca_zh_51k.yaml \
TRAIN_CONFIG=configs/train/qwen2p5_7b_qlora_ddp4.yaml \
EVAL_CONFIG=configs/eval/offline.yaml \
SERVE_CONFIG=configs/serve/vllm_openai_lora.yaml \
LOADTEST_CONFIG=configs/serve/loadtest.yaml \
OBS_TRACKING_URI=mlruns \
OBS_EXPERIMENT_NAME=interview-demo \
bash scripts/e2e_gpu.sh
```

`e2e_gpu.sh` runs the stages in order: data sync, preprocess sample, train, offline eval, `serve --config`, and a headless Locust load test that writes HTML/JSON reports under the configured `output_dir`.

### Hugging Face host configuration notes

On the current GPU host, the verified Hugging Face environment looked like this:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/mnt/vtg_dataset/LLM`
- `hf auth whoami` reported `Not logged in`

`make e2e` still completed successfully in that setup because the checked-in scripts do not rely on a pre-existing local login for these public assets:

- model: `Qwen/Qwen2.5-7B-Instruct`
- dataset: `hfl/alpaca_zh_51k`

At runtime, `scripts/e2e_gpu.sh` preserves `HF_HOME` if it is already set, but it pins the actual cache directories used by the run to project-owned paths unless you override them explicitly:

- `HF_HUB_CACHE=artifacts/hf-cache/hub`
- `HF_DATASETS_CACHE=artifacts/hf-cache/datasets`
- `TRANSFORMERS_CACHE=artifacts/hf-cache/transformers`

That split is intentional: it lets the host keep its existing Hugging Face home directory while making the repo's downloads and dataset cache locations predictable and writable for the demo run.

If you need to use gated assets, export a valid Hugging Face token before running `make e2e`. If you want a different cache layout, override `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE` in the shell before invoking the script.

## Observability

Start the wrapper metrics endpoint first so Prometheus has something to scrape, then bring up the observability stack:

```bash
make obs-up
```

Stop it with:

```bash
make obs-down
```

Ports:

- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login: `admin` / `admin`)

Prometheus scrapes the wrapper metrics target from `WRAPPER_METRICS_TARGET`, which defaults to `host.docker.internal:9100` to match the serve wrapper's default Prometheus port.

On Linux, Docker may need host-gateway support for `host.docker.internal`; this compose file includes `extra_hosts: host.docker.internal:host-gateway`, but if your Docker setup does not support that alias you can set `WRAPPER_METRICS_TARGET` to another reachable host:port before running `make obs-up`.
