# manual serve command review design

## Goal

Create one review-style markdown document under `docs/reviews/` that traces the exact runtime path for this command:

```bash
.venv/bin/python -m aiinfra_e2e.cli serve --config artifacts/tmp/interactive-serve.yaml
```

The document should help the reader understand how control moves from the shell invocation into the repo CLI, into validated `ServeConfig` loading, into the wrapper process manager, and finally into the spawned vLLM child process.

## Chosen output

- **Output file:** `docs/reviews/manual-serve-command-codepath-review.md`
- **Document type:** codepath review, not a general serve architecture note
- **Primary audience:** someone learning this repo from a real manual command they actually ran

## Scope

Include only the path-defining hops for the manual serve command:

1. shell/module entrypoint assumptions
2. `aiinfra_e2e.cli` module execution and Typer dispatch
3. config path validation and YAML loading into `ServeConfig`
4. handoff into `run_vllm_server_from_config`
5. `ManagedVLLMServer.start()` lifecycle
6. child command shaping for `vllm.entrypoints.openai.api_server`
7. child environment shaping and `sitecustomize.py` startup hooks
8. readiness probe semantics through `/v1/models`
9. wrapper keepalive loop and shutdown semantics

Do not include unrelated branches like train, eval, `make e2e`, or loadtest except for short contrast notes if needed.

## Structure

The markdown document will use this structure:

1. `# manual serve command codepath review`
2. `## What is being traced`
3. `## Ordered codepath`
4. `## Snippets`
   - one subsection per hop
   - each snippet includes file path and line reference
5. `## Short summary`

## Content rules

- Keep snippets compact and path-defining.
- Every snippet must cite its file path and line span.
- Explanations should stay tied to execution order, not drift into broad architecture commentary.
- Include the operational meaning of the most important environment assumptions only when they directly affect the path:
  - `.venv/bin/python`
  - relative `--config` resolution from repo root
  - `sys.executable` reuse for the child process
  - wrapper-side localhost probing and proxy bypass

## Expected source files

- `src/aiinfra_e2e/cli.py`
- `src/aiinfra_e2e/config.py`
- `src/aiinfra_e2e/serve/__init__.py`
- `src/aiinfra_e2e/serve/vllm_server.py`
- `src/aiinfra_e2e/serve/metrics.py`
- `sitecustomize.py`
- `artifacts/tmp/interactive-serve.yaml`

## Verification

Before completion, verify:

- the markdown file exists under `docs/reviews/`
- the ordered path is continuous from command entrypoint to child process launch and shutdown
- each snippet has a file citation
- the saved document stays focused on this manual command rather than the broader e2e flow
