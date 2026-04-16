PYTHON := .venv/bin/python
SERVE_CONFIG ?= configs/serve/vllm_openai_lora.yaml
LOADTEST_CONFIG ?= configs/serve/loadtest.yaml
LOADTEST_HOST ?=
LOADTEST_USERS ?=
LOADTEST_SPAWN_RATE ?=
LOADTEST_RUN_TIME ?=
LOADTEST_PROMPT ?=
LOADTEST_MAX_TOKENS ?=
LOADTEST_MODEL ?=
LOADTEST_HTML_REPORT ?=
LOADTEST_JSON_REPORT ?=

.PHONY: ensure-venv setup lint test smoke e2e obs-up obs-down serve loadtest

ensure-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		printf 'Missing project virtualenv Python: %s\n' "$(PYTHON)"; \
		printf 'Create it first, for example:\n'; \
		printf '  python3 -m venv .venv\n'; \
		printf '  .venv/bin/python -m pip install -U pip\n'; \
		exit 1; \
	fi

setup: ensure-venv
	$(PYTHON) -m pip install -e .

lint: ensure-venv
	$(PYTHON) -m ruff check .

test: ensure-venv
	$(PYTHON) -m pytest -q

smoke: ensure-venv
	PYTHON_BIN="$(PYTHON)" bash scripts/smoke_cpu.sh

e2e: ensure-venv
	PYTHON_BIN="$(PYTHON)" bash scripts/e2e_gpu.sh

obs-up:
	docker compose -f docker/compose.yaml up -d

obs-down:
	docker compose -f docker/compose.yaml down

serve: ensure-venv
	$(PYTHON) -m aiinfra_e2e.cli serve --config $(SERVE_CONFIG)

loadtest: ensure-venv
	@if ! $(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('locust') else 1)"; then \
		printf 'Skipping loadtest: locust is not installed in the active Python environment.\n'; \
		printf 'Install locust and set LOADTEST_HOST to run scripts/loadtest_locust.py.\n'; \
		exit 0; \
	fi
	@if [ -z "$(LOADTEST_HOST)" ]; then \
		printf 'Skipping loadtest: set LOADTEST_HOST to the target base URL (for example http://127.0.0.1:8000).\n'; \
		exit 0; \
	fi
	@LOADTEST_HOST="$(LOADTEST_HOST)" LOADTEST_USERS="$(LOADTEST_USERS)" LOADTEST_SPAWN_RATE="$(LOADTEST_SPAWN_RATE)" LOADTEST_RUN_TIME="$(LOADTEST_RUN_TIME)" LOADTEST_PROMPT="$(LOADTEST_PROMPT)" LOADTEST_MAX_TOKENS="$(LOADTEST_MAX_TOKENS)" LOADTEST_MODEL="$(LOADTEST_MODEL)" LOADTEST_HTML_REPORT="$(LOADTEST_HTML_REPORT)" LOADTEST_JSON_REPORT="$(LOADTEST_JSON_REPORT)" $(PYTHON) -c 'import os, subprocess, sys; from aiinfra_e2e.config import LoadTestConfig, load_yaml; from aiinfra_e2e.loadtest import build_chat_completions_payload, resolve_loadtest_artifacts; config = load_yaml("$(LOADTEST_CONFIG)", LoadTestConfig); artifacts = resolve_loadtest_artifacts(config); artifacts.run_dir.mkdir(parents=True, exist_ok=True); payload = build_chat_completions_payload(config); env = os.environ; command = [sys.executable, "-m", "locust", "-f", "scripts/loadtest_locust.py", "--headless", "--host", env["LOADTEST_HOST"], "-u", env.get("LOADTEST_USERS") or str(config.users), "-r", env.get("LOADTEST_SPAWN_RATE") or str(config.spawn_rate), "-t", env.get("LOADTEST_RUN_TIME") or config.run_time, "--model", env.get("LOADTEST_MODEL") or str(payload["model"]), "--prompt", env.get("LOADTEST_PROMPT") or config.prompt, "--max-tokens", env.get("LOADTEST_MAX_TOKENS") or str(config.max_tokens), "--html", env.get("LOADTEST_HTML_REPORT") or str(artifacts.html_report_path), "--json-report", env.get("LOADTEST_JSON_REPORT") or str(artifacts.json_report_path)]; subprocess.run(command, check=True)'
