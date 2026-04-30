# Repo-native serve + trace entrypoint

Use the checked-in config plus the new lifecycle trace wrapper:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/serve_trace_gpu23.sh
```

## What this wrapper does

- forces `CUDA_VISIBLE_DEVICES=2,3` by default
- uses the repo-native serve path: `python -m aiinfra_e2e.cli serve --config ...`
- renders an effective serve config under `artifacts/trace/<run-id>/serve.effective.yaml`
- captures:
  - `serve.log`
  - `gpu.csv`
  - `health.json`
  - `models.json`
  - `metrics.prom`
  - `smoke_chat.json`
  - `env.txt`
- optionally enables NVIDIA Nsight Systems with `ENABLE_NSYS=1`

## Default source config

The wrapper defaults to:

`configs/serve/qwen35_122b_tp2_trace.yaml`

That config maps the original external command into repo-native `ServeConfig` fields and `extra_args`.

## Optional nsys mode

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
ENABLE_NSYS=1 \
bash scripts/serve_trace_gpu23.sh
```

Artifacts land under:

`artifacts/trace/<timestamp>-qwen35-tp2/`
