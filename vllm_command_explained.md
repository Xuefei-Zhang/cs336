# `vllm serve` 命令讲解

## 原始命令

```bash
/home/minghou/github_proj/vtg_ai_tools/services/ai-server/.venv/bin/python3 \
/home/minghou/github_proj/vtg_ai_tools/services/ai-server/.venv/bin/vllm serve \
/mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4 \
--tensor-parallel-size 2 \
--quantization moe_wna16 \
--max-model-len 131072 \
--max-num-seqs 4 \
--max-num-batched-tokens 65536 \
--gpu-memory-utilization 0.9 \
--enable-prefix-caching \
--enable-expert-parallel \
--disable-custom-all-reduce \
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--host 0.0.0.0 \
--port 8000 \
--served-model-name Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
```

---

## 这条命令整体在做什么

这条命令的作用是：

**启动一个 vLLM 在线推理服务，把 `Qwen3.5-122B-A10B-GPTQ-Int4` 这个模型加载到 GPU 上，并通过 OpenAI 兼容风格的接口对外提供服务。**

从参数上看，这是一套偏生产/服务化的启动方式，而不是临时跑一次脚本。它做了几件关键事情：

1. 用 `vllm serve` 启动常驻服务，而不是单次推理。
2. 使用 **2 张 GPU 做 tensor parallel**，所以模型会拆到两张卡上运行。
3. 打开了较长上下文长度（`131072`），说明希望支持超长上下文请求。
4. 打开了 prefix caching、tool calling、reasoning parser 等能力，说明它不只是普通文本补全，而是面向较复杂 agent / coding / tool-use 场景。
5. 监听 `0.0.0.0:8000`，说明局域网内其他机器或本机其他进程都可以访问它。

---

## 按部分拆解

## 1. Python 解释器与 vLLM 可执行入口

```bash
/home/minghou/github_proj/vtg_ai_tools/services/ai-server/.venv/bin/python3 \
/home/minghou/github_proj/vtg_ai_tools/services/ai-server/.venv/bin/vllm serve
```

### 含义

- 前半段是 Python 解释器路径。
- 后半段是虚拟环境里的 `vllm` 命令入口。
- `serve` 表示启动的是 **服务模式**。

### 说明

这说明运行者没有用系统全局 Python，而是用了项目自己的 `.venv` 环境。这样做通常是为了：

- 固定依赖版本
- 避免和系统环境冲突
- 保持项目部署可复现

---

## 2. 模型路径

```bash
/mnt/md127/LLM/Qwen3.5-122B-A10B-GPTQ-Int4
```

### 含义

这是要加载的模型目录。

### 说明

从名字可以读出几个信息：

- `Qwen3.5-122B-A10B`：这是一个很大的 Qwen 3.5 系列模型
- `GPTQ-Int4`：说明它做过 **GPTQ 4-bit 量化**

### 为什么要量化

122B 级别模型非常大，如果不量化，对显存要求会更高。使用 Int4 量化通常是为了：

- 把模型塞进更有限的 GPU 显存
- 降低部署成本
- 在可接受精度损失下换取可部署性

---

## 3. `--tensor-parallel-size 2`

```bash
--tensor-parallel-size 2
```

### 含义

表示使用 **2 路张量并行**。

### 说明

这意味着：

- 模型不会完整放在一张 GPU 上
- 而是切分到两张 GPU 上共同完成前向计算
- 这正对应你前面看到的：
  - `VLLM::Worker_TP0_EP0`
  - `VLLM::Worker_TP1_EP1`

### 作用

对于超大模型，这是很常见的做法，因为单卡往往装不下或带宽/吞吐不够。

---

## 4. `--quantization moe_wna16`

```bash
--quantization moe_wna16
```

### 含义

告诉 vLLM 这套模型使用特定量化方式来运行，这里是 `moe_wna16`。

### 说明

从名字看，这不是最普通的 `awq` / `gptq` 字符串，而是和 **MoE 模型推理路径** 更相关的量化后端/配置形式。

它的核心目的仍然是：

- 让大模型更适合部署
- 控制显存占用
- 兼顾吞吐和精度

### 结合上下文的理解

因为这条命令还带了 `--enable-expert-parallel`，所以可以合理理解为：

- 这套服务在按 **MoE 模型** 的方式使用 vLLM
- 同时显式指定了相应量化路径

---

## 5. `--max-model-len 131072`

```bash
--max-model-len 131072
```

### 含义

把模型允许的最大上下文长度设置为 **131072 tokens**。

### 说明

这是一个非常大的上下文窗口，大约可以理解成“支持超长上下文”。

### 代价

上下文越长：

- KV cache 占用越大
- 显存压力越高
- 请求调度更复杂
- 单请求成本更高

这也是为什么你会看到显存占用很重。

---

## 6. `--max-num-seqs 4`

```bash
--max-num-seqs 4
```

### 含义

限制 vLLM 同时调度的序列数上限为 **4**。

### 说明

它控制的是并发请求在引擎里的并行调度规模。

### 为什么可能设得比较小

对于一个超大模型、超长上下文配置来说，把并发数压低是合理的，因为：

- 每个请求都可能吃很多 KV cache
- 同时并发太多会把显存打满
- 小一些的 `max-num-seqs` 更稳

这通常说明部署者更看重：

- 稳定性
- 单请求可完成性
- 避免 OOM

而不是极限并发数。

---

## 7. `--max-num-batched-tokens 65536`

```bash
--max-num-batched-tokens 65536
```

### 含义

限制一个 batch 中最多处理 **65536 个 token**。

### 说明

这是 vLLM 调度层面的重要参数，用来控制：

- 一次批处理能塞多少 token
- 吞吐和延迟的平衡
- 显存占用上限

### 理解方式

它不是“单个请求最大 token”，而是更接近：

> 调度器在同一轮里最多愿意处理多少 token 工作量

这个值越大，理论上吞吐可能更高，但显存和调度压力也更大。

---

## 8. `--gpu-memory-utilization 0.9`

```bash
--gpu-memory-utilization 0.9
```

### 含义

告诉 vLLM：尽量把每张 GPU 的大约 **90% 显存** 用于模型与缓存分配。

### 说明

这解释了为什么你看到 GPU 0/1 的显存占用非常高。

vLLM 启动服务时通常会：

- 加载模型权重
- 预留 KV cache
- 分配运行时 buffer

所以只要服务成功启动，显存经常就是“长期高位”。

### 重要理解

**显存高，不代表此刻 GPU 正在高算力工作。**

因此你看到：

- 显存 66GB+
- GPU-Util 0%

并不矛盾。更像是：

> 模型已经热加载完毕，当前在等待请求。

---

## 9. `--enable-prefix-caching`

```bash
--enable-prefix-caching
```

### 含义

启用前缀缓存。

### 说明

如果很多请求有相同前缀，比如：

- 相同 system prompt
- 相同工具描述
- 相同长上下文开头

那么 vLLM 可以复用前缀部分的计算结果，减少重复计算。

### 适合的场景

这个参数非常适合：

- agent 系统
- 对话系统
- 带长 prompt 模板的服务
- 多轮工具调用

也就是说，这条命令明显不是“随便开个模型”，而是偏向真实应用服务。

---

## 10. `--enable-expert-parallel`

```bash
--enable-expert-parallel
```

### 含义

启用 **expert parallel**。

### 说明

这通常是为 **MoE（Mixture of Experts）模型**准备的。MoE 模型不是每次都激活全部参数，而是只激活部分专家模块。

启用 expert parallel 的目的通常是：

- 让不同 expert 的计算/存储分布到不同 GPU 或不同并行路径
- 更高效地跑 MoE 模型

### 结合本命令的意义

由于模型名里有 `A10B`，再叠加 `moe_wna16` 与这个参数，可以推测部署者是在按 MoE 推理方式配置这套服务。

---

## 11. `--disable-custom-all-reduce`

```bash
--disable-custom-all-reduce
```

### 含义

禁用 vLLM 自定义的 all-reduce 实现。

### 说明

在多卡张量并行里，GPU 之间需要做通信与聚合，all-reduce 是常见操作。

关闭 custom all-reduce 常见原因是：

- 某些硬件/驱动/通信环境下更稳定
- 避免兼容性问题
- 避免某些自定义实现带来的异常或性能抖动

### 解读

这通常不是“功能需求”，更像是部署经验参数：

> 为了稳定，显式关掉某条更激进的优化路径。

---

## 12. `--reasoning-parser qwen3`

```bash
--reasoning-parser qwen3
```

### 含义

为模型输出启用与 `qwen3` 格式相匹配的 reasoning parser。

### 说明

这通常说明服务端希望更好地处理模型的推理型输出格式，而不是把输出纯粹当普通文本。

### 使用场景

比较适合：

- reasoning model
- agent 推理链路
- 结构化解析模型输出

这表示部署者不仅关心“能生成字”，还关心“怎样更稳定地消费模型输出”。

---

## 13. `--enable-auto-tool-choice`

```bash
--enable-auto-tool-choice
```

### 含义

允许模型自动决定是否调用工具。

### 说明

这一般是给 tool calling / agent 系统准备的。也就是说，服务端希望模型能够：

- 判断要不要调用工具
- 决定调用哪个工具
- 返回结构化工具调用信息

### 含义上的信号

这进一步说明，这个服务不是纯聊天接口，而更像是：

- 智能 agent 后端
- coding assistant 后端
- 可调工具的 LLM 服务

---

## 14. `--tool-call-parser qwen3_coder`

```bash
--tool-call-parser qwen3_coder
```

### 含义

指定工具调用输出的解析器为 `qwen3_coder`。

### 说明

这表示服务端希望模型按某种特定格式输出 tool call，并由后端正确解析。

`qwen3_coder` 这个名字说明它更偏向：

- Qwen3 coder 风格
- 代码/工具调用场景
- 结构化函数调用输出

### 结合前一项看

和 `--enable-auto-tool-choice` 配合后，这套服务很像一个：

> 面向编程助手或 agent 工具编排的模型服务

---

## 15. `--host 0.0.0.0`

```bash
--host 0.0.0.0
```

### 含义

监听所有网卡地址。

### 说明

这意味着：

- 不只是本机 `localhost` 能访问
- 局域网内其他机器也可能访问
- 通常用于服务化部署

如果只想本机访问，常见会设成 `127.0.0.1`。这里设成 `0.0.0.0`，说明它明确要对外提供服务。

---

## 16. `--port 8000`

```bash
--port 8000
```

### 含义

服务监听在 **8000 端口**。

### 说明

这是 vLLM / API 服务里很常见的端口选择。

结合前面的 `--host 0.0.0.0`，可以理解为：

- 这个模型服务对外暴露在 `*:8000`
- 其他服务或客户端会通过这个端口调用它

---

## 17. `--served-model-name Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`

```bash
--served-model-name Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
```

### 含义

指定对外暴露给客户端的模型名。

### 说明

客户端在请求 API 时，看到或填写的模型名会是这个值，而不是必须使用本地磁盘路径。

这样做的好处是：

- API 接口更规范
- 客户端配置更稳定
- 本地模型路径可以和外部调用名解耦

---

# 这组参数透露出的部署意图

从整条命令看，部署者的目标大概是：

## 1. 跑一个超大模型在线服务

不是离线测试，而是常驻服务。

## 2. 以显存换低延迟

通过高显存占用，把模型、缓存、运行时状态长期留在 GPU 上。

## 3. 支持长上下文

`131072` 的上下文窗口说明它想处理超长 prompt / 对话 / 文档场景。

## 4. 面向 agent / tool-use / coder 场景

因为启用了：

- reasoning parser
- auto tool choice
- tool call parser
- prefix caching

这组合非常像智能体后端服务，而不只是普通聊天。

## 5. 偏保守稳定的生产配置

例如：

- `max-num-seqs 4`
- `disable-custom-all-reduce`

这类参数更像是在控制风险，避免 OOM 或兼容性问题。

---

# 为什么会看到“显存很高但 GPU 利用率是 0%”

这条命令本身就解释了那个现象。

原因通常是：

1. 模型已经被加载到 GPU 上
2. KV cache 已经按大上下文预留
3. 服务正在监听端口、等待请求
4. 当前采样瞬间没有活跃推理

所以你看到的是：

- **显存很高**：因为服务已经热启动
- **GPU-Util 很低甚至 0%**：因为这一刻没有正在跑的 token 生成

这是一种正常的在线推理服务状态。

---

# 最终总结

这条命令启动的是一套 **2 卡张量并行的 vLLM 大模型在线服务**，服务对象是：

- `Qwen3.5-122B-A10B-GPTQ-Int4`

它的特点是：

- 使用量化模型降低部署成本
- 使用 2 张 GPU 共同承载模型
- 支持超长上下文
- 开启 prefix cache
- 支持 reasoning / tool calling / coder 风格解析
- 面向局域网或外部服务开放在 `8000` 端口

如果用一句话概括：

> 这不是一个“临时跑模型”的命令，而是一套面向 agent / 工具调用 / 长上下文场景的 vLLM 服务化部署命令。

---

# 和你的命令对比

你的命令是：

```bash
/home/xuefeiz2/self/cs336/.venv/bin/python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 46335 \
--model Qwen/Qwen2.5-7B-Instruct \
--enable-lora \
--tensor-parallel-size 1 \
--max-loras 4 \
--max-lora-rank 64 \
--served-model-name qwen2.5-7b-instruct \
--gpu-memory-utilization 0.9
```

## 先给一个整体判断

如果把两条命令放在一起看，差异非常明显：

- `minghou` 那条更像是 **面向重型 agent / tool-use / 长上下文的大模型生产服务**
- 你的这条更像是 **轻量一些、单卡部署、支持 LoRA 的通用 OpenAI API 服务**

也就是说，你们两条命令虽然都在跑 vLLM，但服务目标并不一样。

---

## 核心对比表

| 维度 | `minghou` 的命令 | 你的命令 |
|---|---|---|
| 启动方式 | `vllm serve` | `python -m vllm.entrypoints.openai.api_server` |
| 模型 | `Qwen3.5-122B-A10B-GPTQ-Int4` | `Qwen/Qwen2.5-7B-Instruct` |
| 模型规模 | 超大模型 | 中小很多的 7B 模型 |
| GPU 并行 | `--tensor-parallel-size 2` | `--tensor-parallel-size 1` |
| GPU 使用方式 | 两张卡共同承载一个模型实例 | 单卡承载一个模型实例 |
| 量化 / MoE | `--quantization moe_wna16` + `--enable-expert-parallel` | 无对应参数 |
| 长上下文 | `--max-model-len 131072` | 未显式设置 |
| 调度控制 | `--max-num-seqs 4`、`--max-num-batched-tokens 65536` | 未显式设置 |
| 前缀缓存 | `--enable-prefix-caching` | 无 |
| 推理/工具调用能力 | `--reasoning-parser qwen3`、`--enable-auto-tool-choice`、`--tool-call-parser qwen3_coder` | 无 |
| LoRA | 无显式 LoRA 参数 | `--enable-lora` |
| LoRA 容量 | 无 | `--max-loras 4`、`--max-lora-rank 64` |
| 监听端口 | `8000` | `46335` |
| 对外暴露模型名 | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | `qwen2.5-7b-instruct` |

---

## 1. 启动入口的区别

### `minghou` 的命令

```bash
vllm serve ...
```

这是 vLLM 提供的高层服务入口，语义上就是：

> “直接启动一个模型服务。”

### 你的命令

```bash
python -m vllm.entrypoints.openai.api_server ...
```

这是通过 Python 模块入口直接启动 OpenAI API server。

### 差异怎么理解

两者本质上都能提供 API 服务，但风格不同：

- `vllm serve` 更像新一些、更直接的服务入口
- `vllm.entrypoints.openai.api_server` 更像显式地启动 OpenAI 兼容 API 服务

你的写法更“底层入口明确”，`minghou` 的写法更“服务化命令风格”。

---

## 2. 模型规模差异非常大

### `minghou`

```bash
Qwen3.5-122B-A10B-GPTQ-Int4
```

这是超大模型，而且还是量化版。

### 你

```bash
Qwen/Qwen2.5-7B-Instruct
```

这是 7B instruct 模型，规模小得多。

### 实际影响

这会直接影响：

- 显存占用
- 推理延迟
- 并行需求
- 部署复杂度
- 适合的业务场景

简单说：

- `122B` 那条命令天然更重、更贵、更复杂
- `7B` 这条命令更轻、更灵活、更适合快速服务化部署

---

## 3. 多卡 vs 单卡

### `minghou`

```bash
--tensor-parallel-size 2
```

表示一个模型实例拆到两张卡上跑。

### 你

```bash
--tensor-parallel-size 1
```

表示整个模型实例只在一张卡上跑。

### 含义

这和你们前面看到的 GPU 现象完全对应：

- `minghou` 的服务占 **GPU 0 + GPU 1**
- 你的服务占 **GPU 2**

所以两边不是同一个部署方式：

- 他是 **双卡张量并行服务**
- 你是 **单卡服务**

---

## 4. `minghou` 更偏向复杂 agent / reasoning / tool-use

`minghou` 那条命令里有这些参数：

```bash
--enable-prefix-caching
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

这些组合起来，说明它不仅是个“聊天模型服务”，而更像：

- agent 后端
- 工具调用后端
- coding assistant 后端
- 推理链更复杂的服务

而你的命令里没有这些参数。

### 这意味着什么

你的服务更像是：

> 一个标准、干净、单卡的 OpenAI-compatible 推理服务

它更通用，也更轻一些。

---

## 5. 你的命令更偏向 LoRA 可扩展性

你的命令里有：

```bash
--enable-lora
--max-loras 4
--max-lora-rank 64
```

这组参数是 `minghou` 那条没有的。

### 说明

你的服务显式支持 LoRA 动态加载或多 LoRA 场景。

### 含义

这通常适合：

- 同一个底模上挂不同 LoRA
- 快速实验不同 adapter
- 面向多任务/多角色微调版本服务化

### 对比解读

所以你这条命令的重点不是复杂 reasoning/tool-calling，而更像：

> 在一个较轻量的基础模型上，提供带 LoRA 能力的灵活 API 服务。

---

## 6. `minghou` 对上下文和调度控制更激进

### `minghou` 额外显式设置了

```bash
--max-model-len 131072
--max-num-seqs 4
--max-num-batched-tokens 65536
```

### 你没有显式设置这些

这通常意味着你的服务更多依赖模型默认值或 vLLM 默认调度参数。

### 怎么理解这个差别

`minghou` 那套更像是在为特定工作负载做精细调参：

- 超长上下文
- 小并发、稳运行
- 控制 batch token 上限

而你的命令更像：

- 尽量简洁
- 用默认配置先把服务跑起来
- 把重点放在 LoRA 支持上

---

## 7. 为什么 `minghou` 那条更吃显存

因为它同时具备这些特征：

- 超大模型
- 双卡张量并行
- 长上下文 `131072`
- 高显存利用率 `0.9`
- prefix cache
- MoE / expert parallel

而你的命令虽然也设了：

```bash
--gpu-memory-utilization 0.9
```

但模型本身只有 7B，且单卡，所以整体资源压力小得多。

换句话说：

- 你们都在“尽量把 GPU 用满”
- 但由于模型规模和配置完全不同，最终显存表现差很多

---

## 8. 两条命令各自更像什么场景

### `minghou` 的命令更像

- 重型生产模型服务
- 长上下文智能体后端
- 带 tool calling 的复杂业务服务
- 为大模型能力最大化做的部署

### 你的命令更像

- 轻量级 OpenAI API 服务
- 单卡模型服务
- 支持 LoRA 的实验/研发/多变体服务
- 更容易部署和维护的通用推理接口

---

## 9. 最后一句话对比

如果只用一句话区分：

> `minghou` 那条命令是在部署一套“更重、更复杂、更像生产级 agent 后端”的大模型服务；你的命令是在部署一套“更轻、更灵活、支持 LoRA 的单卡 OpenAI API 服务”。
