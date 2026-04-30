# Generic Hugging Face Serve Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the repo’s serve path so generic Hugging Face text/chat models can be served through the existing vLLM backend without repeated code edits, while adding multi-GPU auto-selection and a first-class lifecycle trace bundle.

**Architecture:** Keep the backend fixed to vLLM, but split the current single `ServeConfig -> vllm_server` path into stable layers: user intent, model inspection, serve planning, resource planning, command rendering, and execution/trace capture. Preserve the current CLI surface initially by translating the old serve config into the new pipeline, then emit a resolved trace bundle that records what the system observed, decided, and executed.

**Tech Stack:** Python, Typer, Pydantic, PyYAML, subprocess-managed vLLM OpenAI server, `nvidia-smi`, Prometheus text scrape, pytest.

---

## File Structure

### New files
- Create: `src/aiinfra_e2e/serve/models.py`
  - Stable typed models for `ServeUserConfig`, `ModelProfile`, `ServePlan`, `GpuAllocation`, `CommandSpec`, `TraceManifest`.
- Create: `src/aiinfra_e2e/serve/model_inspect.py`
  - HF model/profile inspection from local path or Hub metadata files; capability detection for text/chat-first serving.
- Create: `src/aiinfra_e2e/serve/resource_plan.py`
  - GPU inventory, single/multi-GPU selection policies, and reusable port allocation helpers.
- Create: `src/aiinfra_e2e/serve/serve_plan.py`
  - Planner that turns user config + model profile + overrides into a concrete `ServePlan`.
- Create: `src/aiinfra_e2e/serve/command_render.py`
  - Pure rendering from `ServePlan + GpuAllocation + port selections` to argv/env/cwd.
- Create: `src/aiinfra_e2e/serve/trace_bundle.py`
  - Writes versioned trace bundle directories, lifecycle events, resolved JSON artifacts, and summary output.
- Create: `tests/test_model_inspect.py`
  - Unit tests for profile extraction and override behavior.
- Create: `tests/test_serve_plan.py`
  - Unit tests for serve planning decisions.
- Create: `tests/test_resource_plan.py`
  - Unit tests for multi-GPU auto-selection and port policy.
- Create: `tests/test_command_render.py`
  - Deterministic argv/env rendering tests.
- Create: `tests/test_trace_bundle.py`
  - Trace bundle artifact and event-sequence tests.
- Create: `docs/superpowers/specs/2026-04-24-generic-hf-serve-design.md`
  - Approved design spec from brainstorming, if not already written by the implementation session.

### Existing files to modify
- Modify: `src/aiinfra_e2e/config.py`
  - Add compatibility-layer config types for the new user-facing serve schema while preserving the current `ServeConfig` entry path.
- Modify: `src/aiinfra_e2e/cli.py`
  - Route `serve` through the new pipeline while keeping current CLI behavior.
- Modify: `src/aiinfra_e2e/gpu.py`
  - Keep backward compatibility but migrate shared GPU inventory logic into the new resource-planning layer.
- Modify: `src/aiinfra_e2e/serve/vllm_server.py`
  - Reduce responsibility so it executes a rendered command/env rather than inferring model/runtime policy.
- Modify: `src/aiinfra_e2e/serve/trace_run.py`
  - Convert current ad hoc trace helpers into compatibility wrappers over the new trace bundle writer.
- Modify: `src/aiinfra_e2e/serve/__init__.py`
  - Export the new stable serve pipeline entrypoints.
- Modify: `scripts/e2e_gpu.sh`
  - Replace inline serve config rewriting/GPU selection with the new planner outputs; preserve current demo flow.
- Modify: `scripts/serve_trace_gpu23.sh`
  - Convert from hard-coded trace script to a wrapper around the new trace-enabled serve command while preserving gpu2,3 override behavior.
- Modify: `tests/test_gpu_selection.py`
  - Keep the old selector contract working while adding coverage for new policy functions in the new test file.
- Modify: `tests/test_serve_trace_run.py`
  - Point coverage at resolved artifacts / trace bundle behavior instead of only the old effective YAML helper.
- Modify: `tests/test_openai_api_smoke.py`
  - Update smoke tests to assert the new compatibility path and trace bundle generation.
- Modify: `README.md`
  - Document the new generic serve model and trace bundle entrypoints.
- Modify: `docs/reviews/serve-vllm-deep-dive.md`
  - Update architecture explanation once implementation stabilizes.
- Modify: `docs/reviews/manual-serve-command-codepath-review.md`
  - Update concrete command trace if the flow changes.

---

## Chunk 1: Stabilize the new domain model and inspection boundary

### Task 1: Add typed serve pipeline models

**Files:**
- Create: `src/aiinfra_e2e/serve/models.py`
- Test: `tests/test_model_inspect.py`

- [ ] **Step 1: Write failing tests for the core typed objects**

```python
def test_model_profile_round_trips_with_text_chat_defaults():
    profile = ModelProfile(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        source="hub",
        task="chat",
        supports_chat_template=True,
        supports_multimodal=False,
        architectures=["LlamaForCausalLM"],
    )
    assert profile.task == "chat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_inspect.py -v`
Expected: FAIL with import or symbol-not-found errors for the new serve models.

- [ ] **Step 3: Write minimal typed models**

Implement Pydantic/dataclass types for:
- `ServeUserConfig`
- `ModelProfile`
- `ServePlan`
- `GpuDevice`
- `GpuAllocation`
- `CommandSpec`
- `TraceBundlePaths`

Keep fields explicit and version-friendly. Avoid `dict[str, Any]` except for narrowly scoped metadata extension fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_inspect.py -v`
Expected: PASS for the new typed-object tests.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/models.py tests/test_model_inspect.py
git commit -m "refactor: add serve pipeline domain models"
```

### Task 2: Add model inspection for generic HF text/chat models

**Files:**
- Create: `src/aiinfra_e2e/serve/model_inspect.py`
- Modify: `src/aiinfra_e2e/config.py`
- Test: `tests/test_model_inspect.py`

- [ ] **Step 1: Write failing tests for HF model inspection**

```python
def test_inspect_model_profile_reads_local_hf_config(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}')
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "{{ bos_token }}"}')

    profile = inspect_model_profile(tmp_path)

    assert profile.source == "local"
    assert profile.supports_chat_template is True
    assert profile.supports_multimodal is False
```
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_inspect.py -v`
Expected: FAIL because `inspect_model_profile` does not exist yet.

- [ ] **Step 3: Write minimal inspection implementation**

Implement inspection that:
- reads local `config.json` and `tokenizer_config.json`
- identifies architectures/model_type
- detects whether a chat template exists
- flags obvious multimodal models as unsupported-for-v1 rather than silently treating them as text-only
- supports YAML/user overrides after inspection

Preserve the “text/chat models first” boundary from the approved design.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_inspect.py -v`
Expected: PASS for local-path and override cases.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/model_inspect.py src/aiinfra_e2e/config.py tests/test_model_inspect.py
git commit -m "feat: add generic hf model inspection"
```

---

## Chunk 2: Add serve planning and resource planning

### Task 3: Add serve planning from user intent + model profile

**Files:**
- Create: `src/aiinfra_e2e/serve/serve_plan.py`
- Test: `tests/test_serve_plan.py`

- [ ] **Step 1: Write failing tests for serve planning defaults**

```python
def test_build_serve_plan_prefers_profile_defaults_with_user_override():
    profile = ModelProfile(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        source="hub",
        task="chat",
        supports_chat_template=True,
        supports_multimodal=False,
        architectures=["LlamaForCausalLM"],
        suggested_dtype="bfloat16",
    )
    user = ServeUserConfig(model={"model_id": profile.model_id}, runtime={"startup_timeout_seconds": 600})

    plan = build_serve_plan(user, profile)

    assert plan.model_id == profile.model_id
    assert plan.startup_timeout_seconds == 600
    assert plan.task == "chat"
```
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_serve_plan.py -v`
Expected: FAIL because the planner does not exist.

- [ ] **Step 3: Write minimal serve planner**

The planner should:
- accept `ServeUserConfig + ModelProfile`
- resolve text/chat task mode
- choose structured fields for `dtype`, `startup_timeout_seconds`, `tensor_parallel_size`, `served_model_name`
- keep backend-specific flags in structured fields first, with `extra_args` only as a last-resort escape hatch
- produce a serializable `ServePlan`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_serve_plan.py -v`
Expected: PASS for defaulting and override tests.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/serve_plan.py tests/test_serve_plan.py
git commit -m "feat: add structured serve planning"
```

### Task 4: Add resource planning for ports and single/multi-GPU selection

**Files:**
- Create: `src/aiinfra_e2e/serve/resource_plan.py`
- Modify: `src/aiinfra_e2e/gpu.py`
- Test: `tests/test_resource_plan.py`
- Test: `tests/test_gpu_selection.py`

- [ ] **Step 1: Write failing tests for GPU inventory and allocation policies**

```python
def test_select_gpus_prefers_low_residency_devices_for_multi_gpu_request():
    inventory = [
        GpuDevice(index=0, memory_used_mb=74000, memory_total_mb=81920, utilization_gpu=0, process_count=1),
        GpuDevice(index=2, memory_used_mb=20000, memory_total_mb=81920, utilization_gpu=0, process_count=1),
        GpuDevice(index=3, memory_used_mb=17, memory_total_mb=81920, utilization_gpu=0, process_count=0),
    ]

    allocation = select_gpus(inventory, count=2, policy="multi_free")

    assert allocation.cuda_visible_devices == "3,2"
```
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resource_plan.py tests/test_gpu_selection.py -v`
Expected: FAIL because new inventory/allocation functions do not exist.

- [ ] **Step 3: Write minimal resource planner**

Implement:
- GPU inventory collection from `nvidia-smi`
- selection policies: `respect_env`, `single_free`, `multi_free`, `preferred_then_auto`
- shared port-allocation helpers consolidated from `trace_run.py` and `e2e_gpu.sh`
- compatibility wrapper so existing `select_cuda_visible_devices()` still works for the old single-GPU path

Do not make command-rendering decisions here. Output should be a `GpuAllocation` / `ResourcePlan`, not raw CLI flags.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resource_plan.py tests/test_gpu_selection.py -v`
Expected: PASS, including legacy selector compatibility.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/resource_plan.py src/aiinfra_e2e/gpu.py tests/test_resource_plan.py tests/test_gpu_selection.py
git commit -m "feat: add serve resource planning"
```

---

## Chunk 3: Add command rendering and trace bundle contracts

### Task 5: Add deterministic command rendering

**Files:**
- Create: `src/aiinfra_e2e/serve/command_render.py`
- Modify: `src/aiinfra_e2e/serve/vllm_server.py`
- Test: `tests/test_command_render.py`
- Test: `tests/test_openai_api_smoke.py`

- [ ] **Step 1: Write failing tests for command rendering**

```python
def test_render_vllm_command_uses_resolved_plan_and_gpu_allocation():
    plan = ServePlan(model_id="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2, task="chat")
    allocation = GpuAllocation(cuda_visible_devices="2,3", selected_gpu_indices=[2, 3])

    spec = render_command_spec(plan, allocation)

    assert spec.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "--tensor-parallel-size" in spec.argv
    assert "2" in spec.argv
```
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_command_render.py -v`
Expected: FAIL because renderer symbols do not exist.

- [ ] **Step 3: Write minimal command renderer and adapt executor boundary**

Implement a renderer that:
- receives `ServePlan + GpuAllocation`
- returns a `CommandSpec` with argv/env/cwd
- keeps current compat env shaping (`PYTHONPATH`, tokenizer/tqdm compat, runtime LoRA env)
- moves policy logic out of `build_vllm_command` and `build_vllm_environment`

Update `vllm_server.py` so process execution consumes `CommandSpec` instead of building policy inline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_command_render.py tests/test_openai_api_smoke.py -v`
Expected: PASS for deterministic rendering and backward-compatible smoke behavior.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/command_render.py src/aiinfra_e2e/serve/vllm_server.py tests/test_command_render.py tests/test_openai_api_smoke.py
git commit -m "refactor: render vllm commands from resolved serve plans"
```

### Task 6: Add a first-class lifecycle trace bundle

**Files:**
- Create: `src/aiinfra_e2e/serve/trace_bundle.py`
- Modify: `src/aiinfra_e2e/serve/trace_run.py`
- Modify: `src/aiinfra_e2e/manifest.py`
- Test: `tests/test_trace_bundle.py`
- Test: `tests/test_serve_trace_run.py`

- [ ] **Step 1: Write failing tests for trace bundle artifacts**

```python
def test_trace_bundle_writes_manifest_resolved_artifacts_and_lifecycle_events(tmp_path: Path):
    bundle = create_trace_bundle(tmp_path, run_name="serve-demo")
    bundle.record_event("inspect_start", status="ok")
    bundle.write_resolved_json("model_profile.json", {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"})
    bundle.finalize()

    assert (bundle.root / "manifest.json").exists()
    assert (bundle.root / "resolved" / "model_profile.json").exists()
    assert (bundle.root / "lifecycle" / "events.jsonl").exists()
```
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trace_bundle.py tests/test_serve_trace_run.py -v`
Expected: FAIL because the new trace bundle API does not exist.

- [ ] **Step 3: Write minimal trace bundle implementation**

Implement a versioned trace-bundle writer with these required outputs:
- `manifest.json`
- `inputs/`
- `resolved/model_profile.json`
- `resolved/serve_plan.json`
- `resolved/resources.json`
- `resolved/command.json`
- `lifecycle/events.jsonl`
- `runtime/serve.log`, `health.json`, `models.json`, `smoke_chat.json`, `metrics.prom`
- `system/env.txt`, `gpu_inventory.json`, optional polling CSV
- `summary.md` or equivalent short human-readable summary

Adapt `trace_run.py` so old helper entrypoints become compatibility wrappers over the new trace writer.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trace_bundle.py tests/test_serve_trace_run.py -v`
Expected: PASS with stable artifact structure and event ordering.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/serve/trace_bundle.py src/aiinfra_e2e/serve/trace_run.py src/aiinfra_e2e/manifest.py tests/test_trace_bundle.py tests/test_serve_trace_run.py
git commit -m "feat: add serve lifecycle trace bundles"
```

---

## Chunk 4: Wire the new pipeline through the CLI and scripts

### Task 7: Route the CLI serve command through the new pipeline

**Files:**
- Modify: `src/aiinfra_e2e/cli.py`
- Modify: `src/aiinfra_e2e/serve/__init__.py`
- Test: `tests/test_openai_api_smoke.py`

- [ ] **Step 1: Write failing CLI compatibility tests**

Add tests asserting that:
- old serve YAML still loads
- `serve_command` now produces a resolved pipeline object before execution
- trace-enabled serve can still start from the CLI without breaking old invocation shape

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_openai_api_smoke.py -v`
Expected: FAIL because the CLI still calls the old direct wrapper path.

- [ ] **Step 3: Write minimal compatibility adapter**

Update `serve_command` to:
- load old or new serve config shape
- translate into `ServeUserConfig`
- inspect model
- plan serve
- plan resources
- render command
- execute via the wrapped vLLM process manager

Preserve the current `python -m aiinfra_e2e.cli serve --config ...` interface.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_openai_api_smoke.py -v`
Expected: PASS for CLI compatibility and smoke assertions.

- [ ] **Step 5: Commit**

```bash
git add src/aiinfra_e2e/cli.py src/aiinfra_e2e/serve/__init__.py tests/test_openai_api_smoke.py
git commit -m "refactor: route serve cli through resolved pipeline"
```

### Task 8: Update shell entrypoints to consume resolved pipeline outputs

**Files:**
- Modify: `scripts/e2e_gpu.sh`
- Modify: `scripts/serve_trace_gpu23.sh`
- Test: `tests/test_script_entrypoints.py`
- Test: `tests/test_serve_trace_run.py`

- [ ] **Step 1: Write failing script-entrypoint tests**

Add or update tests to assert:
- `e2e_gpu.sh` no longer duplicates port-selection/GPU-selection logic inline
- `serve_trace_gpu23.sh` can still force `CUDA_VISIBLE_DEVICES=2,3`, but delegates to the new trace-enabled serve path

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_script_entrypoints.py tests/test_serve_trace_run.py -v`
Expected: FAIL because scripts still implement old inline logic.

- [ ] **Step 3: Rewrite shell scripts as thin wrappers**

`e2e_gpu.sh` should:
- validate configs
- invoke the CLI pipeline
- preserve current demo behavior
- stop owning port/GPU selection business logic directly

`serve_trace_gpu23.sh` should:
- keep the current operator-friendly env defaults
- call the new trace-enabled serve flow
- continue producing the expected trace bundle surface

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_script_entrypoints.py tests/test_serve_trace_run.py -v`
Expected: PASS with thin-wrapper behavior locked in.

- [ ] **Step 5: Commit**

```bash
git add scripts/e2e_gpu.sh scripts/serve_trace_gpu23.sh tests/test_script_entrypoints.py tests/test_serve_trace_run.py
git commit -m "refactor: slim serve shell entrypoints"
```

---

## Chunk 5: Docs, examples, and final verification

### Task 9: Update docs and review artifacts

**Files:**
- Modify: `README.md`
- Modify: `docs/reviews/serve-vllm-deep-dive.md`
- Modify: `docs/reviews/manual-serve-command-codepath-review.md`
- Modify: `docs/reviews/README.md`
- Modify: `docs/reviews/manifest.json`

- [ ] **Step 1: Write doc-facing assertions first**

Create a short checklist in the implementation session (or update doc tests if present) covering:
- new serve abstraction is explained in README
- trace bundle layout is documented
- old qwen-specific assumptions are clearly demoted to examples, not architecture

- [ ] **Step 2: Update docs minimally but completely**

Document:
- new serve pipeline layers
- generic HF text/chat model support boundary
- GPU auto-selection policies
- trace bundle directory layout and learning workflow
- compatibility with legacy serve configs

- [ ] **Step 3: Run targeted verification**

Run:
- `pytest tests/test_openai_api_smoke.py tests/test_gpu_selection.py tests/test_resource_plan.py tests/test_trace_bundle.py tests/test_serve_trace_run.py tests/test_script_entrypoints.py -v`
Expected: PASS

- [ ] **Step 4: Run broader verification**

Run:
- `pytest -v`
Expected: PASS

Then run the closest available executable serve validation, depending on environment:
- `python -m aiinfra_e2e.cli serve --config <small/local test config>`
- or the existing optional smoke path if GPU runtime is available

Expected:
- serve starts
- `/v1/models` succeeds
- trace bundle artifacts are written

- [ ] **Step 5: Commit**

```bash
git add README.md docs/reviews/serve-vllm-deep-dive.md docs/reviews/manual-serve-command-codepath-review.md docs/reviews/README.md docs/reviews/manifest.json
git commit -m "docs: explain generic hf serve pipeline and trace bundles"
```

---

## Commit Strategy

Use small, reviewable commits in this order:
1. `refactor: add serve pipeline domain models`
2. `feat: add generic hf model inspection`
3. `feat: add structured serve planning`
4. `feat: add serve resource planning`
5. `refactor: render vllm commands from resolved serve plans`
6. `feat: add serve lifecycle trace bundles`
7. `refactor: route serve cli through resolved pipeline`
8. `refactor: slim serve shell entrypoints`
9. `docs: explain generic hf serve pipeline and trace bundles`

## Notes for the implementing agent

- Preserve the current `serve` CLI surface first; migration should be additive before it becomes canonical.
- Do not let `extra_args` become the primary abstraction. Keep it as an escape hatch only.
- Keep multimodal detection explicit, but v1 support boundary should remain text/chat only.
- Reuse existing proven patterns:
  - `load_yaml()` strict schema validation
  - `write_run_manifest()` stable JSON manifest style
  - `trace_run.py` artifact-writing helpers
  - `test_openai_api_smoke.py` as serve contract anchor
- Any environment snapshot or command artifact must redact or whitelist sensitive env vars if secrets are present.
- If the design spec file does not yet exist, write it before implementation and use it as the plan’s architectural reference.

Plan complete and saved to `docs/superpowers/plans/2026-04-24-generic-hf-serve-plan.md`. Ready to execute?
