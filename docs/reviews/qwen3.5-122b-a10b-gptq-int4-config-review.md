# qwen3.5-122b-a10b-gptq-int4 config review

## What this file is

This review explains the copied source artifact at `docs/reviews/qwen3.5-122b-a10b-gptq-int4-config.json`.

That JSON came from the local model directory:

- `/mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4/config.json`

The point of this document is to do two jobs at once:

1. explain what the model config says about the model itself
2. explain what those settings mean if you want to attach this model to the current repo's `serve` path without changing core code

This report separates three kinds of statements on purpose:

- **config facts** — things stated directly in `config.json`
- **model-package facts** — things stated in the model `README.md`
- **repo interpretation** — what those facts imply for `src/aiinfra_e2e/config.py` and `src/aiinfra_e2e/serve/vllm_server.py`

## Source artifact

The copied JSON is a mirror of the source config, not a rewritten or annotated variant:

- `docs/reviews/qwen3.5-122b-a10b-gptq-int4-config.json`

That matters because the review should stay auditable. A reader can inspect the raw file and then compare it against the interpretations below.

## Top-level reading guide

At the highest level, this config says the model is:

- a **Qwen3.5 multimodal MoE model**
- with a **language model core** described in `text_config`
- with a **vision encoder** described in `vision_config`
- packaged in **Transformers format**
- quantized with **GPTQ 4-bit** under `quantization_config`

The top-level fields that matter most are:

- `architectures = ["Qwen3_5MoeForConditionalGeneration"]`
- `model_type = "qwen3_5_moe"`
- `text_config`
- `vision_config`
- `quantization_config`
- `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id`

The package README adds two useful framing facts:

- license metadata is `apache-2.0`
- the intended pipeline is `image-text-to-text`

So before we even get into the nested fields, we already know this is **not** a plain text-only dense causal LM. It is a multimodal MoE model with quantized weights.

## Language-model architecture signals

### 1. This is a Qwen3.5 MoE language model, not a dense decoder

The strongest signal is:

```json
"architectures": ["Qwen3_5MoeForConditionalGeneration"]
```

and inside `text_config`:

```json
"model_type": "qwen3_5_moe_text"
```

That tells you the language side is part of a sparse Mixture-of-Experts architecture. The package README reinforces this by saying the model is **122B total with 10B activated**.

In practical terms, that means the full parameter footprint is very large even though only part of the model is active per token. For serving, this usually improves the quality/latency tradeoff compared to a same-size dense model, but it does **not** make deployment trivial. Routing logic, expert weights, KV cache, and multimodal components still put real pressure on GPU memory.

### 2. Hidden size and layer count show a relatively narrow-but-deep MoE backbone

`text_config` includes:

- `hidden_size = 3072`
- `num_hidden_layers = 48`
- `num_attention_heads = 32`
- `num_key_value_heads = 2`
- `head_dim = 256`

The hidden size is much smaller than what people often associate with a 100B+ dense model. That is your hint that total capacity is coming from MoE experts rather than from a huge dense residual stream.

The `num_key_value_heads = 2` field is also notable. It implies a grouped-query or multi-query style KV layout rather than full independent KV heads per attention head. That is a common efficiency move because KV cache growth is one of the biggest runtime costs in long-context serving.

### 3. The layer schedule is hybrid, not uniform attention everywhere

`layer_types` alternates many `linear_attention` layers with periodic `full_attention` layers. The pattern is effectively three linear-attention layers followed by one full-attention layer, repeated across the stack.

That means the model is trying to get most of the throughput and long-context efficiency benefits of cheaper attention-like layers while still keeping full-attention refresh points often enough to preserve quality.

For a learner, the important point is: this is not a vanilla transformer block repeated 48 times.

For a maintainer, the implication is: upstream runtime support matters a lot. A serving engine needs to understand this hybrid architecture correctly; otherwise the model is not just slow, it may fail to load entirely.

### 4. MoE routing is explicit in the config

Key fields:

- `num_experts = 256`
- `num_experts_per_tok = 8`
- `moe_intermediate_size = 1024`
- `shared_expert_intermediate_size = 1024`
- `router_aux_loss_coef = 0.001`

This means each token is routed to a subset of experts rather than every expert. The config also suggests the presence of a shared expert path in addition to routed experts.

This is one reason the package README describes the model as **A10B**: about 10B parameters are activated for a token even though the full model is much larger.

Operationally, that matters for serving in two ways:

1. **parallelism still matters** — the total model asset is large enough that single-device deployment is unlikely to be the real target
2. **quantization alone is not the whole story** — 4-bit weights help, but activation memory and KV cache still dominate many failure modes

### 5. Long context is native in the config

`max_position_embeddings = 262144`

That means the model is natively configured for a 262,144-token context window. The package README repeats this and notes that longer context can be attempted with RoPE scaling.

This is powerful, but it also explains why a naive serve launch can run out of memory even if the model weights fit. Long context increases KV-cache pressure dramatically.

For this repo, that means a bare `model_id` plus defaults may be syntactically valid, but a practical serve config may still need tighter control over memory and parallelism.

### 6. RoPE settings hint at the model's positional strategy

Inside `rope_parameters`:

- `rope_type = "default"`
- `rope_theta = 10000000`
- `partial_rotary_factor = 0.25`
- `mrope_interleaved = true`
- `mrope_section = [11, 11, 10]`

The short version is that this model uses a more specialized rotary-position setup than older basic LLM configs.

You do not need to understand every implementation detail to use it, but you should notice the operational implication: if you later override context behavior for ultra-long serving, you are changing a meaningful architectural setting, not just a cosmetic limit.

## Vision and multimodal signals

### 1. This package includes a vision encoder

The presence of `vision_config` means this model is multimodal at the package level, not text-only with optional external tooling.

Key vision fields include:

- `depth = 27`
- `hidden_size = 1152`
- `num_heads = 16`
- `patch_size = 16`
- `temporal_patch_size = 2`
- `out_hidden_size = 3072`

This shows there is a dedicated visual stack whose output is projected into the language-model hidden space.

### 2. Special multimodal token IDs are part of the contract

Top-level fields include:

- `image_token_id = 248056`
- `video_token_id = 248057`
- `vision_start_token_id = 248053`
- `vision_end_token_id = 248054`

These IDs are not random metadata. They are part of how the tokenizer / model pair represent image and video content inside the sequence.

That matters because it confirms again that this package is designed for multimodal use. If you only want plain text serving, you are serving a multimodal checkpoint in a text-only usage mode.

### 3. What this means for the current repo

The repo's current `serve` path wraps `vllm.entrypoints.openai.api_server` and is centered around text-generation compatibility plus LoRA support. Nothing in `ServeConfig` models multimodal policy directly.

That does **not** mean the model cannot be served. It means the repo is not adding special abstraction around image/video inputs at the config-model level. You are essentially depending on upstream vLLM compatibility for multimodal request handling rather than on repo-owned multimodal orchestration.

## Quantization signals

### 1. The config is explicit: this is GPTQ 4-bit

Inside `quantization_config`:

- `bits = 4`
- `quant_method = "gptq"`
- `group_size = 128`
- `sym = true`

This is the clearest operational section in the file. It tells you the checkpoint is not stored as ordinary bf16/fp16 weights.

### 2. Why the config still shows `dtype = "bfloat16"`

Inside `text_config`, `dtype` is `bfloat16`, which can look contradictory at first.

It is not really a contradiction. The model architecture and many runtime tensors are still described in terms of bf16-capable execution, while the stored weights are quantized under GPTQ. In other words:

- **config architecture dtype** is about the expected compute/runtime representation
- **quantization_config** is about how weights are stored and loaded

### 3. Dynamic exclusions matter

The `dynamic` section under `quantization_config` excludes or treats specially several module classes:

- `lm_head`
- `model.language_model.embed_tokens`
- regex-like entries matching attention, shared expert, MTP, and visual modules

This tells you the checkpoint is not a simplistic “everything uniformly quantized to 4-bit” package. Some components are deliberately left alone or handled differently.

For maintainers, the important consequence is that runtime support has to match the checkpoint's quantization expectations. You do not want to assume that any generic GPTQ loader behavior is equivalent.

## What this means for this repo's serve path

This section ties the model facts above to the current repo code.

### 1. `ServeConfig` accepts this model path directly

`src/aiinfra_e2e/config.py` defines:

```python
class ServeConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = 8000
    startup_timeout_seconds: float = 180.0
    model_id: str | None = None
    served_model_name: str | None = None
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9100
    tensor_parallel_size: int = 1
    max_loras: int = 1
    max_lora_rank: int = 64
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    adapters: list[LoRAAdapterConfig] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
```

So the repo can point at the local model directory directly through:

```yaml
model_id: /mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4
```

No core code change is required for that part.

### 2. The wrapper always builds a vLLM command

`src/aiinfra_e2e/serve/vllm_server.py` builds:

```python
command = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--host",
    config.host,
    "--port",
    str(config.port),
    "--model",
    config.model_id,
    "--enable-lora",
    "--tensor-parallel-size",
    str(config.tensor_parallel_size),
    "--max-loras",
    str(config.max_loras),
    "--max-lora-rank",
    str(config.max_lora_rank),
]
```

That means this repo is not wrapping a custom inference engine. It is shaping a vLLM child process.

### 3. `extra_args` is the repo's escape hatch

The same function appends:

```python
if config.extra_args:
    command.extend(config.extra_args)
```

That is the key extension point. If upstream serving guidance for this model requires additional vLLM flags, this repo can pass them without any Python edits.

For this model, the most important such flag is the quantization/runtime selector recommended by the package README's vLLM examples.

### 4. The upstream package README matters here

The local model README gives a concrete vLLM example of the form:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --quantization moe_wna16
```

That is important because it is **not** the same as a generic old GPTQ example using `--quantization gptq`.

For this specific package, the vendor README is telling us that the intended vLLM runtime path uses `--quantization moe_wna16` for Qwen3.5 serving.

So the safest interpretation is:

- the checkpoint package is described in `config.json` as GPTQ-quantized
- but the upstream serving recipe for this exact model on vLLM uses `--quantization moe_wna16`

That means repo users should trust the model-package serving recipe over a generic GPTQ assumption.

### 5. `tensor_parallel_size` is the first practical deployment knob

The repo default is:

```python
tensor_parallel_size: int = 1
```

For a model described as 122B total / 10B activated with 262K native context, that default is very unlikely to be the realistic production setting.

Even with quantization, this model is large enough that you should expect multi-GPU serving to be the norm. The package README itself shows `--tensor-parallel-size 4` in the vLLM example.

So a YAML file that omits `tensor_parallel_size` may be valid, but it is more of a schema-minimum than a deployment-minimum.

### 6. `startup_timeout_seconds` may need to increase

The repo default startup timeout is 180 seconds.

That is fine for smaller models, but large multimodal MoE checkpoints often take longer to initialize, shard, and warm up. If the first run is slow, the wrapper may time out while the engine is still legitimately starting.

So this field is not always part of the strict minimum config, but it is part of the practical one.

### 7. `--enable-lora` is always on in this repo

The wrapper unconditionally adds `--enable-lora`.

That does not mean you must attach LoRA adapters. It means the child vLLM process is started with runtime LoRA support enabled, and the wrapper environment also sets:

```python
env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
```

For this task, the important point is simply that the repo's serve path is opinionated. You are not launching the exact upstream minimal CLI. You are launching the repo's LoRA-capable vLLM wrapper.

## Minimal YAML examples

### Schema-minimal example

This is the smallest config that matches the repo's model and escape-hatch contract:

```yaml
model_id: /mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4
extra_args:
  - --quantization
  - moe_wna16
```

This is useful as a teaching example because it shows the true minimum:

- the repo needs `model_id`
- upstream-model-specific runtime flags go through `extra_args`

### More realistic repo example

For actual serving, a more realistic starting point is:

```yaml
host: 0.0.0.0
port: 8000
model_id: /mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4
served_model_name: qwen3.5-122b-a10b-gptq-int4
startup_timeout_seconds: 600
tensor_parallel_size: 4
gpu_memory_utilization: 0.9
extra_args:
  - --quantization
  - moe_wna16
```

This version is still compatible with the repo's `ServeConfig`, but it acknowledges what the model package and the architecture are telling us:

- this is a large model
- multi-GPU serving is the likely target
- startup can be slow
- memory pressure matters

## Practical caveats

1. **Valid YAML is not the same as a successful launch.** The repo can accept this config even if the host does not have enough GPU memory or enough compatible GPUs.
2. **Do not over-read `config.json` as a promise of runtime support.** The config describes the checkpoint; the runtime still depends on the installed vLLM version and its Qwen3.5 support.
3. **This package is multimodal.** If you only use text requests, you are using a multimodal checkpoint in a narrower mode.
4. **Long context changes the memory story.** A 262K native context is impressive, but KV cache can dominate deployment cost long before raw weights become your only concern.
5. **Use the package-specific serving recipe when generic advice conflicts.** For this exact model package, the local README's vLLM example is the strongest serving hint we have read.

## Short summary

This config file describes a **multimodal, hybrid-attention, MoE Qwen3.5 checkpoint** with **GPTQ 4-bit packaged weights** and a **262K native context window**.

For the current repo, the main operational takeaway is simple: the model can be wired into the existing `serve` path by setting `model_id` to the local directory and passing the model-specific vLLM quantization/runtime flag through `extra_args`. But because this is a very large multimodal MoE model, the schema-minimal YAML is only a teaching minimum; a realistic serve config will usually also need multi-GPU tensor parallelism, more startup time, and explicit memory tuning.
