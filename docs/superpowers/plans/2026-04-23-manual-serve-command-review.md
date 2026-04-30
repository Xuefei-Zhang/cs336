# Manual Serve Command Review Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a review-style markdown document that traces the exact runtime path for the approved manual serve command.

**Architecture:** Add one focused doc under `docs/reviews/` that follows the execution chain from module invocation through config loading, wrapper startup, vLLM child launch, readiness probing, and shutdown. Reuse existing repo review-doc conventions, keep snippets compact, and cite each hop with exact file/line references.

**Tech Stack:** Markdown, Typer CLI, Pydantic config models, Python subprocess wrapper, vLLM serve path

---

## Chunk 1: Write and verify the review markdown

### Task 1: Create the manual serve command codepath review

**Files:**
- Create: `docs/reviews/manual-serve-command-codepath-review.md`
- Reference: `docs/superpowers/specs/2026-04-23-manual-serve-command-review-design.md`
- Reference: `src/aiinfra_e2e/cli.py`
- Reference: `src/aiinfra_e2e/config.py`
- Reference: `src/aiinfra_e2e/serve/__init__.py`
- Reference: `src/aiinfra_e2e/serve/vllm_server.py`
- Reference: `src/aiinfra_e2e/serve/metrics.py`
- Reference: `sitecustomize.py`
- Reference: `artifacts/tmp/interactive-serve.yaml`

- [ ] **Step 1: Gather the approved path-defining snippets**

Read the spec and the source files listed above. Confirm the exact execution chain:

1. `.venv/bin/python -m aiinfra_e2e.cli`
2. Typer `serve` command dispatch
3. `_load_config(..., ServeConfig)`
4. `load_yaml(...)`
5. `run_vllm_server_from_config(...)`
6. `ManagedVLLMServer.start()`
7. `build_vllm_command(...)`
8. `build_vllm_environment(...)`
9. child startup via `sitecustomize.py`
10. `/v1/models` readiness loop
11. `stop()` shutdown semantics

- [ ] **Step 2: Write the markdown document**

Use this structure:

```markdown
# manual serve command codepath review

## What is being traced

## Ordered codepath

## Snippets
### 1. ...
**File:** `...:line-line`
```python
...
```

## Short summary
```

Keep explanations tied to execution order. Do not drift into `make e2e`, training, or loadtest.

- [ ] **Step 3: Verify the document content**

Check all of the following:

- the file exists at `docs/reviews/manual-serve-command-codepath-review.md`
- the ordered path is continuous from command entrypoint to shutdown
- each snippet includes file/line citation
- the review is codepath-first rather than broad architecture prose

- [ ] **Step 4: Read the saved document back**

Read the saved file and confirm the rendered markdown still matches the approved scope and has no obvious path gaps.
