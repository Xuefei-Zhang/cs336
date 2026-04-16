# Learnings


- Task 1 scaffold: setuptools build backend worked for editable installs in this environment, while hatchling was not importable during pip editable resolution.
- Task 3 CLI: Typer subcommands can stay forward-compatible by sharing a single `--config` loader helper and keeping non-implemented commands as validation-only stubs until later tasks land.
- Task 4 data pipeline: deterministic splits stay reproducible without RNG by hashing a stable per-record payload (selected fields plus row index) and deriving split membership from the hash fraction; manifest split hashes should be computed from sorted stable keys so any record change shows up in `data_manifest.json`.
- Task 4 HF retry: keeping the dataset loader injectable or resolved at call time avoids Python default-argument capture, which makes transient-failure retry tests honest when `datasets.load_dataset` is monkeypatched.
- Task 5 prompt/masking: rendering the full conversation and then deriving the assistant boundary from the prompt-only prefix keeps label masking deterministic across chat templates, so assistant loss stays aligned even when `apply_chat_template` controls formatting.
- Task 6 training: in this environment, `transformers` may probe adapter metadata through Hugging Face cache constants that were frozen at import time, so CPU smoke runs need a writable project-local HF cache override at both the environment and in-memory constant levels; also, GPT-style tokenizers can expose `apply_chat_template` without a configured `chat_template`, so preprocessing must check for actual template configuration before taking the chat-template path.
- Task 7 checkpoint/resume: using TRL's built-in `save_strategy="steps"` plus `trainer.train(resume_from_checkpoint=...)` was enough to support periodic checkpoints and CPU-tested resume without adding a custom training loop; the OOM fallback stays bounded by shrinking config knobs (batch size first, then max sequence length) and retrying from the latest saved checkpoint.
- Task 8 offline eval: keeping eval generation behind an injected `generator(prompt) -> str` interface makes offline regression tests fully CPU-safe while still matching training run-dir and MLflow artifact conventions.
- Task 9 serving: keeping the wrapper on `python -m vllm.entrypoints.openai.api_server` with lazy runtime checks avoids import-time `vllm` coupling in CPU tests, while loaded LoRA adapters can be selected on OpenAI requests by sending the adapter name as the `model` value after a stdlib `POST /v1/load_lora_adapter` call.
- Task 10 observability: keeping Prometheus target substitution in a tiny mounted shell script avoids Docker Compose variable-interpolation warnings while preserving a runtime-configurable `WRAPPER_METRICS_TARGET`; Grafana provisioning is simplest when the datasource uses a fixed UID and dashboards reference that UID directly.
- Task 11 load testing: keeping loadtest config as a small wrapper around existing `ServeConfig` and `ObsConfig` lets tests validate OpenAI payload construction, run-artifact paths, and MLflow artifact logging without requiring a live vLLM server or Locust execution.

- Final Verification Wave F3: manual QA on 2026-04-13 verified `python -m aiinfra_e2e.cli --help`, `bash scripts/smoke_cpu.sh`, and `docker compose -f docker/compose.yaml config` all completed successfully in the CPU environment; GPU-only execution remained unverified here, but the repo has explicit skip guards in `src/aiinfra_e2e/serve/vllm_server.py:get_vllm_smoke_skip_reason()` and `tests/test_openai_api_smoke.py` to avoid running vLLM smoke without CUDA, `vllm`, `nvidia-smi`, and `AIINFRA_E2E_VLLM_MODEL`.

- Final Verification Wave F4: when plan DoD treats `make` as the single orchestrator, command-level success (`pytest`, `ruff`, editable install, compose config) is necessary but not sufficient; fidelity requires Makefile targets to execute the same real workflows that README/CI claim.

## 2026-04-14 Task: e2e port conflict workaround
- E2E now uses effective temp serve/loadtest configs so hosts with ports 8000 or 9100 already occupied can still run without changing checked-in defaults.
- The workaround stays isolated to runtime config selection, preserving the committed serve/loadtest port settings for normal use.
- This avoids brittle failures on shared machines while keeping repository defaults stable.
