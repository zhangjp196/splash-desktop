# Splash 推测解码完整方案：可选 DFlash2、GGUF/MLX、MTP、DSpark 与模型扩展

状态：完整工程方案；P0 的“无 draft”路径已在当前检出中实现（见文末“实现
状态”）。本文把“DFlash2 允许为空”扩展成一条可落地的改造路线：先解除
DFlash2 的硬依赖，再把 draft 抽象成多种 provider，最后用声明式 family pack
扩大模型范围。落地按 P0 → P4 分阶段，每一阶段都可独立发布、可回滚。

---

## 0. 一页结论

- **现状**：DFlash2 不是“可选加速”，而是安装、记录、描述、状态、内存、解码六层
  的硬依赖。只加 `--no-draft` 会让引擎在 `inspectModelRoot` 或
  `Runtime::decodeAsync` 直接失败。
- **核心改造**：在安装记录和模型描述里引入 **SpeculationPlan**，把 `none` 做成
  一等 provider；解码从“固定 DFlash2 图”改成“provider 提出 → target 验证 →
  接受 → 提交”。
- **支持范围**：
  - `none`：纯自回归，GGUF/MLX 目标无需任何 draft；
  - `dflash2`：safetensors 或 GGUF，来源可以是 `z-lab` 等镜像；
  - `mtp`：模型内置 nextn/MTP head，或外部 MTP head；
  - `dspark`：外部 DSpark bundle（并行 backbone + 顺序 Markov head +
    置信度调度），实验性。
- **模型范围**：Tier A 现有调优家族；Tier B 用声明式 family pack 接入标准
  dense/MoE；Tier C 需要新算子的实验架构。不承诺“任意模型零改动可跑”。
- **工期**：P0 1–2 周；P1 1–2 周；P2 2–3 周；P3 3–4 周；P4 持续。
- **最大风险**：无 draft 路径的输出一致性、内存审计的零值处理、缓存命名空间
  隔离、GGUF draft 的 kernel/量化覆盖。

---

## 1. 背景与现状

### 1.1 当前调用链

```text
splash serve --model ... / --model-dir ...
        │
        ▼
install/models.py  Selection.of()
        │  选择链接 = hash(model, revision, language_only, draft_model, mmproj)
        ▼
install/upstream.py  prepare() / prepare_local()
        │  解析 target 与 draft 的 revision，下载，检查，发布 assembly
        ▼
install/assembly.py  model.json (version 1)
        │  sources.target + sources.draft + files[]
        ▼
engine  inspectModelRoot(root)
        │  读 model.json / config.json / draft/config.json
        ▼
model::ModelDescriptor  target + draft + stateLayout
        │
        ▼
model::loadModel()  DraftCheckpointLoader + DFlashDraft
        │
        ▼
engine::RuntimeResources  requireLoadedModel + MemoryPlan + MemoryAudit
        │
        ▼
model::Runtime::decodeAsync()  DFlash 提议 8 行 → target 验证 → 接受
```

### 1.2 六处硬依赖

| # | 层 | 位置 | 当前行为 | 影响 |
|---|---|---|---|---|
| 1 | 家族注册 | `install/families.py` | 每个 `ModelFamily` 都有 `draft_repo`，没有则为 `None` 并在安装时报错 | 没有 draft 的模型无法进入安装流程 |
| 2 | 解析/下载 | `install/upstream.py::_resolve_draft`、`install/local.py::_draft` | 没有显式 draft 就解析家族 draft；家族没有就 `ModelError` | 离线、无 draft、只有 target 的场景被拒绝 |
| 3 | 安装记录 | `install/assembly.py::RECORD_KEYS`、`assembly_record` | 固定包含 `sources.draft`，字段精确匹配 | 无法表达“没有 draft”，也无法表达 provider/format |
| 4 | 模型描述 | `runtime/model/ModelDescriptor.mm::inspectModelRoot`、`valid()` | 强制 `draft/`；强制 draft 的 hidden/vocab/mask/target layers 与 target 一致 | 引擎拒绝无 draft 的 assembly |
| 5 | 加载/状态/内存 | `ModelFactory.cpp`、`StateLayout.hpp`、`RuntimeResources.mm`、`MemoryAudit.cpp` | 总是构造 `DraftCheckpointLoader`；draft rings/window 必占内存；draft 权重为 0 视为统计不完整 | 即使跳过描述检查，也会在加载或内存审计失败 |
| 6 | 解码 | `runtime/model/Runtime.cpp::decodeAsync` | 固定 `encodeDraftBatchGraph` + 8 行 verify + draft commit | 没有纯自回归解码路径 |

### 1.3 为什么不能只加一个开关

假设只在安装器放行“没有 draft”：

- 引擎 `inspectModelRoot` 第一行就要求 `draft/`；
- 即使去掉该检查，`ModelDescriptor::valid()` 仍要求 `draft.hiddenSize`、
  `draft.targetLayerCount`；
- 即使描述通过，`loadModel` 仍会构造 `DraftCheckpointLoader`，目录不存在即失败；
- 即使加载跳过，`CompositeStateLayout` 仍要求 draft rings/window 合法；
- 即使状态清零，`MemoryPlan` 仍返回 `draft_weight_bytes_required`；
- 即使内存放行，`decodeAsync` 没有“只跑 target”的图。

所以必须把“推测”从数据结构、内存、调度、解码四处同时改成可选。本文的方案就是
把这四处收敛到一个 **SpeculationPlan** 上：描述、内存、调度、解码都只问 plan。

---

## 2. 目标、非目标与验收标准

### 2.1 功能目标

- **F1 可选 draft**：任意受支持的 GGUF/MLX target 可以在没有 draft 的情况下
  安装、加载、服务。
- **F2 显式策略**：用户可以要求 `auto`、`none`、`dflash2`、`mtp`、`dspark`。
- **F3 多格式 draft**：DFlash2 draft 可以是 safetensors 或 GGUF。
- **F4 多来源 draft**：`z-lab`、`incoai`、本地目录、本地 GGUF 文件、模型内置
  MTP 张量、外部 MTP/DSpark bundle。
- **F5 无 draft 的正常服务**：输出、采样、约束、停止条件与 target 单独运行一致。
- **F6 可见性**：启动日志、`/status`、`/v1/models` 明确报告 provider、来源、
  回退原因。
- **F7 可隔离**：有 draft 与无 draft 的同一 target 不共享状态、缓存命名空间、
  内存计划。
- **F8 可扩展**：新增一个 Tier B 家族不需要改引擎源码，只加 family pack。
- **F9 可回滚**：旧的 v1 安装、旧缓存、旧引擎行为有明确处理。
- **F10 可测试**：每个 provider 都有确定性的 golden 测试，不依赖主观感受。

### 2.2 非目标

- 不做通用 kernel 生成器。新架构仍需要算子、tokenizer profile、chat format。
- 不在安装时训练、转换或重新量化 draft（除了引擎已支持的格式读取）。
- 不加载远端 Python 或模型侧代码；draft 始终是数据。
- 不做分布式、多机、跨设备卸载。
- 不承诺“所有 Hugging Face 模型直接可跑”；只承诺受支持 target 的 draft 维度
  可选、可换。

### 2.3 验收标准

| 编号 | 场景 | 通过条件 |
|---|---|---|
| A1 | `--model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --speculation none` | 安装、加载、服务成功；`/status` 的 `speculation.provider == "none"` |
| A2 | `--model-dir <MLX 4bit> --no-draft` | 同上；不访问 Hub 的 draft 仓库 |
| A3 | 无 draft 的 greedy 输出 | 与“有 draft 但所有 proposal 被拒绝”的 target-only oracle 逐 token 一致 |
| A4 | 无 draft 的 seeded 采样 | 与 target-only oracle 的分布一致；固定 seed 逐 token 一致 |
| A5 | 无 draft 的约束/停止条件 | 与现有有 draft 路径行为一致 |
| A6 | GGUF DFlash2 draft | Q8_0/BF16 draft 与 Q4_K target 配对成功；draft 权重按 GGUF 类型加载 |
| A7 | 不兼容 draft | 默认回退到 `none` 并给出原因；`--require-draft` 时启动失败 |
| A8 | 内置 MTP | DeepSeek 风格 GGUF 用 `nextn_predict_layers` 成功提议；不匹配时回退 |
| A9 | 旧 v1 安装 | 仍可加载，行为与当前一致 |
| A10 | 旧缓存 | 不会被无 draft 模型命中；反之亦然 |
| A11 | 性能 | `none` 的 decode tok/s 不低于 target-only 基线；draft provider 不劣化现有 5 家族 |
| A12 | 回归 | 现有 5 家族的 golden 输出、内存审计、启动日志不变 |

---

## 3. 术语与不变量

### 3.1 术语

- **SpeculationPlan**：一次安装中“如何推测”的不可变记录，属于安装身份。
- **Provider**：算法。`none`、`dflash2`、`mtp`、`dspark`。
- **Format**：权重容器。`none`、`safetensors`、`gguf`。
- **Source**：权重来源。`none`、`builtin`（target 内）、`external`
  （独立仓库/目录）。
- **Alignment**：draft 与 target 必须共享的属性集合。
- **Proposal width**：一个周期最多提出的行数，含 anchor。DFlash2 为 8，
  `none` 为 1，MTP 为 `k`，DSpark 为 `n`。
- **Fallback**：provider 不可用时的行为。`none`（降级）或 `required`（失败）。
- **Family pack**：声明式家族描述，含 model type、维度谓词、张量命名、
  tokenizer/chat profile、kernel shape 行。

### 3.2 不变量

- **I1 target 自足**：target 本身必须足以加载并服务；draft 永远不是加载前提。
- **I2 plan 不可变**：一次安装的 plan 写入 `model.json` 后不再改写；回退原因
  属于运行时状态。
- **I3 不静默替换**：不允许用另一个家族、另一个 revision、另一种量化的 draft
  顶替；只能降级为 `none`。
- **I4 不执行远端代码**：只读配置、header、张量。
- **I5 身份包含 plan**：provider/format/source/proposal width 进入安装哈希和
  持久缓存命名空间。
- **I6 输出语义不变**：推测只影响速度，不影响 greedy/seeded/约束/停止的语义。
- **I7 零值合法**：`none` 的 draft 状态、权重、窗口是合法的 0，不是“缺失”。
- **I8 先校验后下载**：plan check 在任何权重下载之前完成。

---

## 4. 总体架构

### 4.1 分层

```text
┌──────────────────────────────────────────────────────────────────┐
│ CLI / 桌面 App / prepare_local_model                             │
│  --speculation, --draft-model, --draft-format, --require-draft   │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ SpeculationResolver                                              │
│  优先级：CLI → target 声明 → family registry → none              │
│  输出：SpeculationPlan（Python dataclass）                       │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Installer                                                        │
│  target 解析/下载/检查；draft 解析/下载/检查；record；publish    │
│  assembly: config.json target/ tokenizer/ vision/ [draft/]       │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Engine descriptor                                                │
│  inspectModelRoot → ModelDescriptor{target, speculation, vision} │
│  valid() 只校验 plan 声明的部分                                  │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Loader                                                           │
│  target loader + 可选 draft loader（safetensors/GGUF/MTP/DSpark）│
│  LoadedModel{target, speculator, vision}                         │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Runtime                                                          │
│  Speculator::addPropose → target verify → accept → addCommit     │
│  none: 1 行 verify；dflash2: 8 行；mtp: k 行；dspark: n 行       │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 生命周期

1. **Resolve**：CLI/配置/家族 → `SpeculationPlan`。
2. **Plan check**：只读 target 配置/GGUF header、draft 配置/header、alignment；
   不下载权重。
3. **Install**：下载 target；若 plan 有 draft，下载 draft；写 `model.json` v2；
   发布 assembly。
4. **Load check**：读张量名/类型/形状；不匹配则按 fallback 策略处理。
5. **Run**：`Speculator` 驱动解码；`none` 走 1 行自回归。
6. **Report**：启动日志、`/status`、`/v1/models` 报告实际 provider 与回退原因。

### 4.3 模块职责

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `install` 下的 `speculation.py` | plan 类型、解析优先级、候选选择、alignment 比较 | 不下载、不读权重 |
| `install/families.py` | 家族默认 provider 候选、chat/vision 元数据 | 不判断仓库名兼容性 |
| `install/upstream.py` | 解析 revision、检查、下载、发布 | 不实现 provider 算法 |
| `install/local.py` | 本地文件/目录的 plan 解析 | 不发网络请求 |
| `install/assembly.py` | record schema、verify、GC | 不解释 provider 语义 |
| `runtime/model/ModelDescriptor.*` | plan 的 C++ 表示与校验 | 不下载 |
| `runtime/model/Speculator.*` | provider 接口与实现 | 不管理请求生命周期 |
| `runtime/model/Runtime.*` | 解码图、状态提交 | 不决定 provider |
| `runtime/engine/*` | 调度、内存、缓存 | 不读 draft 文件 |

---

## 5. 数据模型

### 5.1 `model.json` v2

```json
{
  "version": 2,
  "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
  "family": "Qwen3.8-27B",
  "target_format": "gguf",
  "vision_format": "none",
  "speculation": {
    "version": 1,
    "provider": "dflash2",
    "format": "gguf",
    "source": {
      "kind": "external",
      "repo": "z-lab/Qwen3.8-27B-DFlash2-GGUF",
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "files": {
        "draft/Qwen3.8-27B-DFlash2-Q8_0.gguf": "<sha256>"
      }
    },
    "proposal_rows": 8,
    "state": "draft-rings",
    "fallback": "none",
    "alignment": {
      "family": "Qwen3.8-27B",
      "hidden_size": 5120,
      "vocabulary_size": 151936,
      "mask_token_id": 151643,
      "target_layers": 64,
      "target_hidden_reduction": "none",
      "tokenizer_digest": "<sha256>"
    }
  },
  "sources": {
    "target": {
      "repo": "unsloth/Qwen3.8-27B-GGUF",
      "revision": "abcdef0123456789abcdef0123456789abcdef01"
    }
  },
  "files": {}
}
```

`provider: "none"` 时：

```json
"speculation": {
  "version": 1,
  "provider": "none",
  "format": "none",
  "source": {"kind": "none"},
  "proposal_rows": 1,
  "state": "none",
  "fallback": "none"
}
```

### 5.2 字段规则

| 字段 | 取值 | 规则 |
|---|---|---|
| `version` | `2` | 有 `speculation` 时为 2；读取器接受 1 |
| `speculation.version` | `1` | plan schema 版本，独立于 record 版本 |
| `provider` | `none`/`dflash2`/`mtp`/`dspark` | 未知值拒绝，消息点名引擎版本 |
| `format` | `none`/`safetensors`/`gguf` | 与 provider 匹配；`none` 必须配 `none` |
| `source.kind` | `none`/`builtin`/`external` | `builtin` 不允许 `repo`/`revision` |
| `source.repo` | 仓库 ID | 仅 `external`；本地路径用 `path` |
| `source.revision` | commit | 仅 `external` repository |
| `source.files` | assembly 路径 → digest | 只读展示与校验，不单独信任 |
| `proposal_rows` | `1..8`（DFlash2 固定 8） | `none` 必须为 1；MTP/DSpark 由 provider 定义 |
| `state` | `none`/`draft-rings`/`mtp-cache`/`dspark-state` | 与 provider 匹配 |
| `fallback` | `none`/`required` | 用户显式请求 `--require-draft` 时为 `required` |
| `alignment` | 对象 | 有 draft 时必填；`none` 可省略 |
| `alignment.tokenizer_digest` | sha256 | 外部 MTP/DSpark 必填；DFlash2 建议填 |

### 5.3 C++ 类型

```cpp
namespace splash::model {

enum class SpeculationProvider : uint8_t { None = 1, DFlash2, Mtp, DSpark };
enum class DraftFormat : uint8_t { None = 1, Safetensors, Gguf };
enum class DraftSourceKind : uint8_t { None = 1, Builtin, External };
enum class DraftStateKind : uint8_t { None = 1, DraftRings, MtpCache, DSparkState };
enum class FallbackPolicy : uint8_t { None = 1, Required };

struct SpeculationAlignment final {
  std::string family;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t maskTokenId = 0;
  uint32_t targetLayers = 0;
  std::string targetHiddenReduction;  // "none" | "stream_mean"
  std::string tokenizerDigest;      // empty: not required
  bool operator==(const SpeculationAlignment &) const = default;
};

struct SpeculationPlan final {
  SpeculationProvider provider = SpeculationProvider::None;
  DraftFormat format = DraftFormat::None;
  DraftSourceKind sourceKind = DraftSourceKind::None;
  DraftStateKind state = DraftStateKind::None;
  FallbackPolicy fallback = FallbackPolicy::None;
  uint32_t proposalRows = 1;
  SpeculationAlignment alignment;
  std::string sourceName;   // 展示用：repo 或路径
  std::string revision;     // 展示用：commit
  std::string filesDigest;  // record 的 files 摘要，进缓存身份

  [[nodiscard]] constexpr bool hasDraft() const noexcept {
    return provider != SpeculationProvider::None;
  }
  [[nodiscard]] bool valid() const noexcept;
};

} // namespace splash::model
```

`ModelDescriptor`：

```cpp
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  SpeculationPlan speculation;   // 取代 DFlashDraftLayout draft
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;  // draft 可为零
  TargetSource targetSource{};
  VisionSource visionSource{};
  std::string sourceIdentity;
  RopeScaling ropeScaling;

  [[nodiscard]] bool hasDraft() const noexcept { return speculation.hasDraft(); }
  [[nodiscard]] bool hasVision() const noexcept { return visionSource != VisionSource::None; }
  [[nodiscard]] bool valid() const noexcept;
};
```

物化后的 `DFlashDraftLayout` 仍存在，作为 `SpeculationPlan` 在 DFlash2 下的
细节；`ModelDescriptor` 可以同时持有 `DFlashDraftLayout dflash` 和
`SpeculationPlan speculation`，但 `valid()` 只按 provider 检查对应字段。

### 5.4 Python 类型

```python
# speculation.py
from dataclasses import dataclass, field
from enum import Enum

class Provider(str, Enum):
    NONE = "none"
    DFLASH2 = "dflash2"
    MTP = "mtp"
    DSPARK = "dspark"

class DraftFormat(str, Enum):
    NONE = "none"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"

class SourceKind(str, Enum):
    NONE = "none"
    BUILTIN = "builtin"
    EXTERNAL = "external"

class Fallback(str, Enum):
    NONE = "none"
    REQUIRED = "required"

@dataclass(frozen=True)
class Alignment:
    family: str
    hidden_size: int
    vocabulary_size: int
    mask_token_id: int | None = None
    target_layers: int = 0
    target_hidden_reduction: str = "none"
    tokenizer_digest: str | None = None

@dataclass(frozen=True)
class Plan:
    provider: Provider
    format: DraftFormat
    source_kind: SourceKind
    proposal_rows: int
    state: str
    fallback: Fallback
    alignment: Alignment | None = None
    source_name: str | None = None
    revision: str | None = None

    def as_record(self) -> dict: ...
```

### 5.5 v1 兼容

读取 v1 时合成：

```python
if record["version"] == 1:
    speculation = {
        "version": 1,
        "provider": "dflash2" if record["sources"].get("draft") else "none",
        "format": "safetensors" if record["sources"].get("draft") else "none",
        "source": {"kind": "external"} if record["sources"].get("draft") else {"kind": "none"},
        "proposal_rows": 8 if record["sources"].get("draft") else 1,
        "state": "draft-rings" if record["sources"].get("draft") else "none",
        "fallback": "none",
    }
```

v1 的 `sources.draft` 在 v2 中仍被接受，但写回时统一写 `speculation`。旧 assembly
不重写，只读兼容。

### 5.6 身份与哈希

三处哈希必须包含 plan：

1. **assembly 哈希**：`model.json` 的 SHA-256，天然包含 plan。
2. **selection link**：`Selection.of()` 的哈希列表加入
   `speculation`、`draft_format`、`require_draft`。
3. **持久缓存命名空间**：`persistentCacheNamespace()` 增加一行
   `speculation <provider> <format> <proposal_rows> <state> <files_digest>`；
   `none` 写 `speculation none none 1 none`。

这样有 draft 与无 draft 的同一 target 不会共享缓存，也不会互相恢复状态。

---

## 6. Provider 规格

### 6.1 `none`

**目标**：纯自回归解码，作为所有 provider 的参考实现。

**算法**：

```text
每个 decode step，对每个 lane：
  1. 把 anchor token 写入 InputTokens[0]
  2. 只跑 target 的 verify 图，VerifyRows = 1
  3. 采样下一 token，RetainedCount = 1
  4. 只提交第 0 行；draft 状态无操作
```

**状态**：`DraftStateLayout{}` 全零；`laneBytes()`、`cachedBytes()` 不增加
draft 部分。

**内存**：`draftWeightsBytes = 0`；不解码 draft 图；不分配 draft rings/window；
不捕获 target hidden rows。

**验证**：`proposalRows == 1`；`accepted == 0`；`drafted_tokens == 0`；
`draft_acceptance_rate == 0`。

**关键实现**：`NoneSpeculator` 写 `ProposedTokens[0] = anchor`、
`VerifyRows = 1`；`encodeVerifyCommit` 跳过 `encodeDraftStateCommitBatch`；
`finalizeDecode` 用 `verifyRows == 1` 正常走 target policy。

**风险**：target 的 verify 图是按 8 行物理形状编译的；只提交 1 行时，第 1–7 行
仍会被计算。P0 接受这个浪费（简单、正确），P1 再评估是否新增 1 行 arena。

### 6.2 `dflash2` / safetensors（现有路径）

- **配置**：`config.json`，`architectures == ["DFlash2DraftModel"]`，
  `dflash_config.target_layer_ids`、`mask_token_id`、`block_size`。
- **权重**：BF16 `model.safetensors` 或 index 分片，可选
  `mask_embedding.safetensors`。
- **加载**：`DraftCheckpointLoader` → `planDraftImages` → af4g64 planes。
- **提交**：`DFlashDraft::addContextCommit`，draft rings 接收保留行。
- **proposalRows**：8。
- **改动**：只把 `readDraftLayout` 的入口从“全局必须有 draft”改为
  “provider == dflash2 时读取”，其余不动。

### 6.3 `dflash2` / GGUF

**为什么需要**：DFlash2 已有社区 GGUF 转换（如 `z-lab`、`neuralforgequantum`
镜像），量化 target 配量化 draft 才能在小内存 Mac 上工作。

**配置来源**（二选一，优先前者）：

1. GGUF header 的 `dflash2.*` / `draft.*` KV，映射到与 `config.json` 相同的
   字段；
2. draft 目录下的 `config.json`。

**张量映射**（示意，落地按实际发布命名校对）：

| config/布局 | GGUF 名称 |
|---|---|
| `fc.weight` | `draft.fc.weight` |
| `hidden_norm.weight` | `draft.hidden_norm.weight` |
| `norm.weight` | `draft.norm.weight` |
| `candidate_selector.hidden_projection.weight` | `draft.selector.hidden_projection.weight` |
| `candidate_selector.predecessor_codebook` | `draft.selector.predecessor_codebook` |
| `candidate_selector.successor_codebook` | `draft.selector.successor_codebook` |
| `layers.N.*` | `draft.blk.N.*` |
| `mask_embedding` | `draft.mask_embedding` |

**类型支持**（P1 首批）：BF16、F16、Q8_0、Q4_K；norm 为 F32/BF16。
不在列表的类型 → 回退原因 `unsupported_draft_type`，不做 host 反量化。

**加载**：新增 `GgufDraftSource`，复用 `GgufFile` / `GgufPreparation` 的
metadata、split、tensor rows 读取；计划阶段校验每个张量的名称、类型、形状。

**split**：遵守 `NNNNN-of-MMMMM` 规则；安装为 `draft/` 下的 part 文件；
`findDraftGgufs()` 合并读取。

**proposalRows**：8。

### 6.4 `dflash2` / `z-lab`

`z-lab/<MODEL>-DFlash2` 与 `z-lab/<MODEL>-DFlash2-GGUF` 是普通来源，不需要
专门运行时路径。安装器只做三件事：

1. family registry 允许多个候选仓库，按顺序尝试：
   `incoai/<MODEL>-DFlash2` → `z-lab/<MODEL>-DFlash2`；
2. 显式 `--draft-model z-lab/...` 永远优先，并原样记录；
3. `--draft-format auto` 时，`-GGUF` 后缀选择 GGUF；否则按 target 的格式偏好。

**禁止**：按仓库名判断兼容性。兼容性只看 config/header/张量，与当前
`families.py` 的规则一致。

### 6.5 `mtp` / builtin

**输入**：target 自带的 MTP/nextn head。

- GGUF：`<arch>.nextn_predict_layers`，张量名
  `blk.<N>.nextn.*` / `mtp.*`；
- MLX：`model.mtp.*`、`model.layers.<N>.mtp.*`；
- Model Optimizer：`mtp.*` 模块（当前明确不读）。

**Plan**：

```json
{
  "provider": "mtp",
  "format": "gguf",
  "source": {"kind": "builtin"},
  "proposal_rows": 2,
  "state": "mtp-cache",
  "alignment": {
    "family": "DeepSeek-V4",
    "hidden_size": 7168,
    "vocabulary_size": 129280,
    "target_layers": 61
  }
}
```

**算法**：

```text
propose(k)：
  for i in 0..k-1:
    h = MTPLayer(hidden_or_embedding(prev_token), kv_cache)
    logits = target.lm_head(h)
    next = sample_or_argmax(logits)
    prev_token = next
verify：target 对 anchor + k 个 proposal 跑 k+1 行
commit：target KV + MTP 自己的 KV
```

**状态**：`MtpStateLayout`，每层 K/V 或 MLA latent，按 head 的实际结构；
`CompositeStateLayout` 增加 `mtp` 可选部分，`none` 时为零。

**约束**：head 必须完整；缺张量、层数不符、head 维度不符 → 回退；不允许用
target 的普通层“近似”MTP。

### 6.6 `mtp` / external

独立仓库或目录只放 MTP head，必须带：

```json
{
  "architectures": ["MtpHead"],
  "base_model": "deepseek-ai/DeepSeek-V4",
  "alignment": {
    "family": "DeepSeek-V4",
    "hidden_size": 7168,
    "vocabulary_size": 129280,
    "target_layers": 61,
    "base_model_digest": "<sha256 of target config or GGUF header>",
    "tokenizer_digest": "<sha256 of tokenizer.json>"
  }
}
```

`base_model_digest` 或 `tokenizer_digest` 不匹配即不兼容，不做“最接近匹配”。

### 6.7 `dspark`

参考 [DSpark](https://arxiv.org/abs/2607.05147)：并行 backbone + 轻量顺序
Markov head，按置信度和引擎吞吐调度验证长度。

**Bundle 布局**：

```text
dspark/
  config.json          # ABI、维度、对齐、proposal 宽度
  backbone.safetensors # 并行 draft backbone
  markov.safetensors   # 顺序 head
  tokenizer.json       # 可选，用于 digest 校验
```

**ABI**：

```json
{
  "abi": "dspark-1",
  "proposal_rows": 8,
  "confidence": {"kind": "prefix-survival", "calibration": "..."},
  "alignment": {...}
}
```

**算法**：

1. `backbone` 一次前向产生 block 的并行 proposal；
2. `markov` head 按顺序修正块内依赖；
3. 置信度映射到现有 `VerifyRows` / `VerifyBudget`；
4. target 验证与接受复用现有路径；
5. `addCommit` 保存 Markov 状态。

**门槛**：实验性 `--speculation dspark`；ABI 未知拒绝；在 release prompts 上
acceptance 不低于 DFlash2 才转正。

### 6.8 Provider 能力矩阵

| 能力 | none | dflash2/safetensors | dflash2/gguf | mtp/builtin | mtp/external | dspark |
|---|---|---|---|---|---|---|
| target 必需 | 是 | 是 | 是 | 是 | 是 | 是 |
| draft 仓库 | 否 | 是 | 是 | 否 | 是 | 是 |
| 量化 draft | — | 否（BF16） | 是 | 随 target | 是 | 是 |
| 额外下载 | 0 | draft | draft | 0 | head | bundle |
| 状态 | 0 | rings+window | rings+window | mtp KV | mtp KV | markov |
| 验证宽度 | 1 | 1–8 | 1–8 | 1–k | 1–k | 1–n |
| 成熟度 | P0 | 现有 | P1 | P2 | P2 | P3 |

---

## 7. 解析与优先级

### 7.1 CLI

| 选项 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `--speculation` | `auto`/`none`/`dflash2`/`mtp`/`dspark` | `auto` | 显式 provider |
| `--no-draft` | flag | — | `--speculation none` 的别名 |
| `--draft-model` | `OWNER/REPO` 或本地路径 | — | 显式 draft 来源（保留现有语义） |
| `--draft-format` | `auto`/`safetensors`/`gguf` | `auto` | 容器选择 |
| `--require-draft` | flag | — | fallback = required |
| `--draft-revision` | commit/tag/branch | 默认分支 | draft 独立 revision |
| `--list-speculation` | flag | — | 打印可选 provider 与原因 |

`--no-draft` 与 `--draft-model` 互斥；`--speculation none` 与
`--draft-model` 互斥；`--require-draft` 与 `--speculation none` 互斥。

### 7.2 优先级

1. `--speculation none` / `--no-draft` → 直接 `none`。
2. `--speculation dflash2|mtp|dspark` → 该 provider；缺来源则报错或回退。
3. `--draft-model` → 由内容探测 provider/format，覆盖家族默认。
4. target 自身声明：
   - MLX `config.json` 的 `speculation` 对象；
   - GGUF 的 `splash.speculation` KV；
   - 内置 MTP 的 `nextn_predict_layers`（仅当用户选 `mtp` 或 `auto`）。
5. family registry 的默认候选。
6. `none`。

`auto` 的语义：能解析到兼容 draft 就用；解析不到就用 `none`，并记录原因。
`dflash2`/`mtp`/`dspark` 的语义：必须用该 provider，解析不到则按 `fallback`
处理（默认回退 `none`，`--require-draft` 则失败）。

### 7.3 冲突规则

- `--speculation none --draft-model X`：参数错误。
- `--speculation dflash2` + 路径是 MTP head：参数错误（provider 不匹配）。
- `--draft-format gguf` + 仓库只有 safetensors：解析错误，不自动换格式。
- `--require-draft` + `--speculation none`：参数错误。
- `--model-dir` + `--draft-model` 指向 target 目录内部：拒绝（与现有
  `_gguf_in`/`_split_parts` 的规则一致）。

### 7.4 target 声明示例

MLX `config.json`：

```json
"speculation": {
  "provider": "dflash2",
  "format": "safetensors",
  "repository": "z-lab/Qwen3.8-27B-DFlash2"
}
```

GGUF KV（安装器读取，不写入 target）：

```text
splash.speculation.provider = "dflash2"
splash.speculation.repository = "z-lab/Qwen3.8-27B-DFlash2-GGUF"
splash.speculation.format = "gguf"
```

### 7.5 family registry 候选

```python
ModelFamily(
    name="Qwen3.8-27B",
    speculation=(
        DFlash2Candidate("incoai/Qwen3.8-27B-DFlash2", preferred="safetensors"),
        DFlash2Candidate("z-lab/Qwen3.8-27B-DFlash2", preferred="safetensors"),
        DFlash2Candidate("z-lab/Qwen3.8-27B-DFlash2-GGUF", preferred="gguf"),
    ),
    chat_format=QWEN,
)
```

解析时按顺序尝试；第一个“可解析且兼容”的候选胜出。全部失败 → `none` 或错误，
取决于 `fallback`。

### 7.6 Selection 身份

`Selection` 增加：

```python
speculation: str          # auto/none/dflash2/mtp/dspark
draft_format: str         # auto/safetensors/gguf
require_draft: bool
```

`selection_link()`、`local_selection_link()`、`prepared_selection_link()` 的
哈希输入都加入这些字段。这样 `--model X` 与 `--model X --no-draft` 是两个
安装，互不覆盖。

---

## 8. 校验与回退

### 8.1 Plan check（下载前）

| Provider | 检查 |
|---|---|
| `none` | 只检查 target；无 draft 字段 |
| `dflash2` | draft config/header 的 architecture、hidden、vocab、mask、target layers、attention/selector/window、block size |
| `dflash2`/GGUF | 上述 + GGUF header 的 `dflash2.*` 或 sidecar config + 张量名/类型列表 |
| `mtp`/builtin | target 的 `nextn_predict_layers`、head 张量完整性、head 维度、层数 |
| `mtp`/external | head config、alignment、base digest、tokenizer digest |
| `dspark` | bundle config、ABI、alignment、backbone/markov 张量列表 |

Plan check 由 `engine model-check` 执行，输入是 target 的 config/GGUF metadata
和 draft 的 config/header，不读权重。

### 8.2 Load check（权重读入前）

- 每个期望张量存在、类型支持、形状匹配；
- 没有多余的必要张量缺失；
- draft 的 vocabulary 与 target 的 head 一致；
- MTP/DSpark 的层数与 config 一致；
- GGUF draft 的 split 完整；
- 不支持的量化类型 → `unsupported_draft_type`。

### 8.3 Alignment 比较

| 字段 | 比较方式 | 缺失时 |
|---|---|---|
| `family` | 精确 | 必填 |
| `hidden_size` | 精确 | 必填 |
| `vocabulary_size` | 精确 | 必填 |
| `mask_token_id` | 精确（DFlash2） | 必填 |
| `target_layers` | 精确 | 必填 |
| `target_hidden_reduction` | 精确 | 取 target 默认 |
| `tokenizer_digest` | 精确 | 外部 MTP/DSpark 必填 |
| `base_model_digest` | 精确 | 外部 MTP 必填 |

### 8.4 回退矩阵

| 原因 | `fallback=none` | `fallback=required` | 状态 |
|---|---|---|---|
| draft 仓库不可达，且无缓存 | 警告 + `none` | 启动失败 | `fallback_reason=unreachable` |
| draft 文件缺失/损坏 | 警告 + `none` | 启动失败 | `missing_files` |
| draft 格式不支持 | 警告 + `none` | 启动失败 | `unsupported_format` |
| draft 配置不兼容 | 警告 + `none` | 启动失败 | `incompatible_config` |
| draft 张量不匹配 | 警告 + `none` | 启动失败 | `incompatible_tensors` |
| 引擎无该 provider | 警告 + `none` | 启动失败 | `unsupported_provider` |
| 内置 MTP 不完整 | 警告 + `none` | 启动失败 | `incomplete_mtp` |
| DSpark ABI 未知 | 警告 + `none` | 启动失败 | `unsupported_abi` |

**绝不**因为 draft 不可用而换一个“相似”的 draft；只允许 `none`。

### 8.5 错误消息规范

```text
Warning: cannot use the Qwen3.8-27B DFlash2 draft z-lab/Qwen3.8-27B-DFlash2-GGUF@<rev>; serving without speculation: draft config dflash_config target_layer_ids names a layer past the target's
```

启动失败：

```text
error: cannot prepare /models/Qwen3.8-27B-GGUF: the DFlash2 draft z-lab/... is incompatible with Qwen3.8-27B: draft config hidden_size 4096 does not match the target's 5120
```

### 8.6 `--require-draft`

用于 CI、基准、发布验证：任何回退都失败，确保测的是 draft 路径，而不是
悄悄测了 `none`。

---

## 9. 安装器改造

### 9.1 `install/families.py`

- `ModelFamily` 增加 `speculation: tuple[Candidate, ...]`；
- 保留 `draft_repo` 属性（取第一个 DFlash2 候选）以兼容现有调用；
- 增加 `MtpCandidate`、`DFlash2Candidate` 数据类；
- 家族可以声明 `supports_builtin_mtp`。

```python
@dataclass(frozen=True)
class Candidate:
    provider: Provider
    repo: str | None
    format: DraftFormat
    builtin: bool = False

@dataclass(frozen=True)
class ModelFamily:
    name: str
    speculation: tuple[Candidate, ...]
    chat_format: ChatFormat
    vision: bool = True
    supports_builtin_mtp: bool = False

    @property
    def draft_repo(self) -> str | None:
        return next((c.repo for c in self.speculation
                     if c.provider is Provider.DFLASH2 and c.repo), None)
```

### 9.2 `install/upstream.py`

新增/修改：

- `_resolve_speculation(selection, family, installed, repo) -> Plan | None`：
  按优先级解析，返回 `None` 表示 `none`；
- `_install()`：先解析 plan，再决定是否解析 draft；
- `_draft()` → `_speculation_files(plan, repo, installed)`：按 format 分派；
- `draft_files()` → `dflash2_safetensors_files()` / `dflash2_gguf_files()` /
  `mtp_files()` / `dspark_files()`；
- `assembly_record()`：写 `speculation` 对象（v2）；
- `_changes()`：plan 变化也算变化（provider/format/revision/files）；
- `_start_installed()`：plan 不兼容时按 fallback 决定“重装为 none”还是失败。

伪代码：

```python
def _install(selection, repo, installed, plan, projector):
    target = inspect_target(...)
    if plan is None:
        plan = resolve_plan(selection, target.family, installed, repo)
    files = {}
    if plan.has_draft:
        files |= download_draft(plan, target, installed)
    files |= download_target(target, repo)
    publish(..., assembly_record(selection.model, target, repo, plan), files)
```

### 9.3 `install/local.py`

- `prepare_local_model(target, draft=None, ..., speculation="auto", draft_format="auto")`
  （新增关键字参数，默认保持现有行为）。
- `draft=None` + `speculation="none"` → 无 draft；
- `draft` 是目录：探测 `config.json` / `*.gguf`，决定 provider/format；
- `draft` 是文件（GGUF）：直接作为 GGUF draft；
- 本地文件不访问 Hub；draft 缺失时按 `fallback` 处理，不再无条件报错。

### 9.4 `install/assembly.py`

- `RECORD_KEYS` 增加 `"speculation"`；
- `SPECULATION_KEYS` 定义 schema；
- `verify()` 接受 version 1 和 2；
- v2 有 `speculation` 时校验其内部字段；
- `provider != none` 要求 `draft/` 存在且至少一个权重文件；
- `provider == none` 要求没有 `draft/`（或允许空目录，但拒绝非空）；
- GGUF target 的 `metadata` 逻辑不变；
- GC/pins 不变：有 draft 时 pin 两个来源，无 draft 时只 pin target。

### 9.5 `install/cached.py`

- `cached_targets()` 不变；
- 新增 `cached_speculations()`：扫描 `models/.resolved/*/model.json` 的
  `speculation`，用于离线提示与测试；
- 选择链接的缓存键加入 plan 字段。

### 9.6 `install/models.py`

- `Selection` 增加 `speculation`、`draft_format`、`require_draft`；
- `Selection.of()` 参数与哈希；
- 新增 `parse_speculation`、`parse_draft_format`；
- `selection_link()`、`local_selection_link()`、`prepared_selection_link()`
  纳入新字段；
- `--model` 的 `OWNER/REPO` 语法不变。

### 9.7 `install/launcher.py` 与 `server/serve_options.py`

- `serve_options.py` 的 `model` 组增加四个选项；
- `launcher.py` 的 `_ensure_installed()` 传递新参数；
- `parse_args()` 的互斥检查；
- `serve_record` 的启动记录增加 `speculation`（用于桌面 app）。

示例：

```python
model.add_argument("--speculation", choices=("auto", "none", "dflash2", "mtp", "dspark"),
                   default="auto")
model.add_argument("--no-draft", action="store_true")
model.add_argument("--draft-format", choices=("auto", "safetensors", "gguf"), default="auto")
model.add_argument("--require-draft", action="store_true")
```

### 9.8 安装器文件清单

| 文件 | 改动 | 风险 |
|---|---|---|
| `install/families.py` | 家族候选、provider 元数据 | 低 |
| `install` 下的 `speculation.py` | 新增：plan 类型、解析、alignment | 中 |
| `install/upstream.py` | plan 解析、draft 分派、record | 高 |
| `install/local.py` | 本地无 draft、GGUF draft | 中 |
| `install/assembly.py` | v2 schema、verify | 中 |
| `install/cached.py` | 缓存扫描 | 低 |
| `install/models.py` | Selection、CLI | 中 |
| `install/launcher.py` | 参数传递 | 低 |
| `server/serve_options.py` | 选项定义 | 低 |
| `dev/tests/install/*` | 单测 | 低 |

---

## 10. 运行时改造

### 10.1 `ModelDescriptor`

`ModelDescriptor.mm`：

```cpp
ModelDescriptor inspectModelRoot(const std::filesystem::path &root) {
  // ...
  std::optional<std::filesystem::path> draftConfig;
  if (std::filesystem::exists(root / "draft"))
    draftConfig = root / "draft" / "config.json";
  auto plan = readSpeculationPlan(record, draftConfig);
  ModelDescriptor result = describeSourceModel(..., plan);
  if (!result.valid()) throw std::invalid_argument("incompatible target and speculation plan");
  return result;
}
```

`valid()`：

```cpp
bool ModelDescriptor::valid() const noexcept {
  if (name.empty() || !speculation.valid()) return false;
  // 只校验 plan 声明的部分
  if (speculation.provider == SpeculationProvider::DFlash2)
    return validDraft(target, speculation, dflash);
  if (speculation.provider == SpeculationProvider::Mtp)
    return validMtp(target, speculation, mtp);
  if (speculation.provider == SpeculationProvider::DSpark)
    return validDSpark(target, speculation, dspark);
  return validTarget(target);  // none
}
```

`describeSourceModel` 的 `draft` 参数改为
`const std::optional<SpeculationPlan> &plan`；`readDraftLayout` 只在
`provider == dflash2` 时调用。

### 10.2 `ModelFactory`

```cpp
struct LoadedModel final {
  ModelDescriptor descriptor;
  std::shared_ptr<WeightImages> images;
  TargetWeights target;
  std::variant<std::monostate, DFlashDraftWeights, MtpWeights, DSparkWeights> draft;
  QwenVisionWeights vision;
  std::unique_ptr<Speculator> speculator;
  std::string manifestFingerprintSha256;
};
```

`loadModel()`：

```cpp
if (descriptor.speculation.hasDraft()) {
  result.draft = loadDraft(descriptor, root, images, backend);
  result.speculator = makeSpeculator(descriptor, result);
} else {
  result.speculator = std::make_unique<NoneSpeculator>();
}
```

### 10.3 `StateLayout`

```cpp
struct CompositeStateLayout final {
  GdnStateLayout target;
  DraftStateLayout draft;   // 全零合法
  MtpStateLayout mtp;       // 可选
  DSparkStateLayout dspark; // 可选

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.validOrNone() && mtp.validOrNone() &&
           dspark.validOrNone() && onlyPlannedState();
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return kLaneGdnCells * target.cellBytes() + draft.ringBytes() +
           draft.windowBytes() + mtp.laneBytes() + dspark.laneBytes();
  }
};
```

`DraftStateLayout::valid()` 拆成：

```cpp
[[nodiscard]] constexpr bool validOrNone() const noexcept {
  return layers == 0 ? kvHeads == 0 && headDimension == 0 && contextWidth == 0
                     : valid();
}
```

### 10.4 内存计划与审计

- `ModelMemoryProfile::validationError()`：`draftWeightsBytes == 0` 只在
  `speculation.hasDraft()` 时是错误；否则合法。
- `fixedRuntimeBytes()`：零值相加不变。
- `EngineMemoryBreakdown::toStatusJson()`：增加 `speculation` 对象。
- `MemoryAudit`：`MissingMeasurement` 只在 plan 要求 draft 时检查 draft 分类。
- `RuntimeResources::requireLoadedModel()`：

```cpp
if (!loaded.targetActualAllocatedBytes() ||
    (loaded.descriptor.hasDraft() && !loaded.draftBytes()) ||
    (loaded.descriptor.hasVision() && !loaded.vision.actualAllocatedBytes) ||
    ...)
```

### 10.5 `Speculator` 接口

```cpp
namespace splash::model {

enum class SpeculationKind : uint8_t { None = 1, DFlash2, Mtp, DSpark };

class Speculator {
public:
  virtual ~Speculator() = default;
  [[nodiscard]] virtual SpeculationKind kind() const noexcept = 0;
  [[nodiscard]] virtual uint32_t proposalRows() const noexcept = 0;
  [[nodiscard]] virtual const DraftStateLayout &draftState() const noexcept = 0;

  virtual void addPropose(CommandGraph &graph, const SpeculatorBatch &batch) = 0;
  virtual void addVerifyRows(CommandGraph &graph, const SpeculatorBatch &batch) = 0;
  virtual void addCommit(CommandGraph &graph, const SpeculatorBatch &batch) = 0;

  [[nodiscard]] virtual uint64_t weightsBytes() const noexcept = 0;
  [[nodiscard]] virtual bool needsTargetHidden() const noexcept = 0;
};

} // namespace splash::model
```

`SpeculatorBatch` 持有 lanes、logical positions、maximum retained、
`DecodeArena` slices、`DFlashDecodeBuffers` 等；具体字段按现有
`encodeDraftBatchGraph` 的参数收敛。

### 10.6 解码循环改造

**现在**：

```cpp
encodeDraftBatchGraph(commandGraph, entries, positions, maximumRetained, constrained);
encodeBatchVerifyInput(commandGraph, width);
encodeTargetVerifyBatchForward(commandGraph, entries, items, *reads);
encodeVerifyCommit(commandGraph, entries, items, maximumRetained);
```

**改造后**：

```cpp
speculator->addPropose(commandGraph, batch);
speculator->addVerifyRows(commandGraph, batch);
encodeBatchVerifyInput(commandGraph, width);
encodeTargetVerifyBatchForward(commandGraph, entries, items, *reads);
encodeVerifyCommit(commandGraph, entries, items, maximumRetained); // 内部调用 speculator->addCommit
```

`NoneSpeculator::addPropose`：

```cpp
void NoneSpeculator::addPropose(CommandGraph &graph, const SpeculatorBatch &batch) {
  // ProposedTokens[lane][0] = anchor; 其余行写 0 或 anchor
  for (lane : batch.lanes)
    batch.writeProposedToken(lane, 0, batch.anchor(lane));
  batch.setVerifyRows(1);
}
```

`NoneSpeculator::addVerifyRows`：空实现（已经写死 1 行）。

`encodeVerifyCommit`：

```cpp
encodeTargetVerifyBatchPolicy(graph, entries);
encodeBatchAcceptance(graph, entries, maximumRetained);
encodeTargetStateCommit(graph, entries);
speculator->addCommit(graph, batch);  // none: 空
```

`finalizeDecode`：

- `verifyRows` 必须 `>= 1` 且 `<= maximumRetained`；
- `accepted` 对 `none` 为 0；
- `verifyBudget` 在 `proposalRows == 1` 时禁用；
- `draftedTokens` 对 `none` 为 0。

### 10.7 Scheduler / Engine

- `DraftContextPlan` 改为 `std::optional<DraftContextPlan>`；
- `configureDraftStatePlan()` 在 `!model.hasDraft()` 时返回 `std::nullopt`；
- `kMinimumJunctionGain` 用 `draftContextTokens()` 代替固定值：

```cpp
const uint32_t gain = model.hasDraft()
    ? model::ExecutionLimits::draftContextTokens
    : 0;
```

- `addStateBoundary()` 的 `completeAfter` 条件在无 draft 时只看 target；
- 预填充的 `DraftCaptureSpan` 只在 `needsTargetHidden()` 时生成；
- 状态缓存只在 `hasDraft()` 时保存 draft window；
- `VerifyBudget` 只在 `proposalRows() > 1` 时构造。

### 10.8 缓存命名空间

`persistentCacheNamespace()` 增加：

```cpp
canonical << "speculation " << providerName(plan.provider) << ' '
          << formatName(plan.format) << ' ' << plan.proposalRows << ' '
          << stateName(plan.state) << ' ' << plan.filesDigest << '\n';
```

`none` 的 `draft` 行写 `draft 0 0 0 0`，`context` 行写 `context 0 q4 0`，
保证旧缓存不会误命中。

### 10.9 Status / Protocol

`/status` 增加：

```json
"speculation": {
  "provider": "none",
  "format": "none",
  "proposal_rows": 1,
  "state": "none",
  "draft_weights_bytes": 0,
  "fallback_reason": null
}
```

`/v1/models` 增加 `speculation` 字段。协议层 `Speculation` 统计对 `none`
保持全零，不改变现有客户端解析。

### 10.10 运行时文件清单

| 文件 | 改动 | 风险 |
|---|---|---|
| `runtime/model/ModelDescriptor.hpp/.mm` | plan 类型、valid、inspect | 高 |
| `runtime/model/ModelFactory.hpp/.cpp` | 可选加载、Speculator | 高 |
| `runtime/model` 下的 `Speculator.hpp` 与 `Speculator.cpp` | 新增接口与 None | 中 |
| `runtime/model/StateLayout.hpp` | 零值合法、可选 MTP/DSpark | 高 |
| `runtime/model/DraftCheckpoint.*` | GGUF source 抽象 | 高 |
| `runtime/model/Runtime.cpp` | 解码循环、提交 | 高 |
| `runtime/engine/Engine.cpp` | DraftContextPlan 可选、junction | 高 |
| `runtime/engine/MemoryPlan.*` | 零 draft 合法 | 中 |
| `runtime/engine/MemoryAudit.*` | 零值审计 | 中 |
| `runtime/engine/RuntimeResources.mm` | requireLoadedModel、启动日志 | 中 |
| `runtime/engine/Status.*` | 字段 | 低 |
| `runtime/engine/Protocol.*` | 兼容字段 | 低 |

---

## 11. 模型范围扩展

### 11.1 三级模型

| Tier | 定义 | 例子 | 支持方式 |
|---|---|---|---|
| A | 已调优家族，有专用 kernel 与 DFlash2 | Qwen3.8-27B、Qwen3.6-35B-A3B、Ornith、Flash-Next、Flint | 现有路径 |
| B | 标准 dense/MoE，形状落在现有算子内 | Llama 3.x、Qwen2/3 dense、Gemma 3/4、Mistral、DeepSeek MoE | family pack + 现有算子 |
| C | 需要新算子 | MLA、线性注意力、特殊 MoE 路由 | 实验开关，不参与 auto |

### 11.2 Family pack

建议新增 `install` 下的 `families.json`（或 Python 表），字段：

```json
{
  "name": "Llama-3.1-8B",
  "tier": "B",
  "model_types": ["llama"],
  "match": {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256
  },
  "flat_config": true,
  "tensor_prefix": "model.",
  "tokenizer_profile": "llama3",
  "chat_format": "llama3",
  "vision": false,
  "speculation": {
    "providers": ["none", "dflash2"],
    "default": "none"
  },
  "kernels": {
    "attention": ["llama3-4096-32-8-128"],
    "ffn": ["swiglu-14336"]
  }
}
```

引擎侧只增加一个 `GenericDenseLayout`（Tier B），由 pack 的参数实例化；
不再为每个尺寸写一个 C++ layout。现有 5 家族继续走专用 layout。

### 11.3 Tier B 的最小算子集

- RMSNorm；
- RoPE（含 YaRN）；
- GQA/MHA paged attention；
- SwiGLU/GeGLU FFN；
- 可选 MoE（top-k router + shared expert）；
- LM head；
- 量化：GGUF Q4_K/Q5_K/Q6_K/Q8_0/BF16/F16；MLX affine 4/6/8、mxfp4、nvfp4。

不满足这些的模型不进 Tier B。

### 11.4 量化矩阵

| Target | Draft | 状态 |
|---|---|---|
| GGUF Q4_K/Q5_K/Q6_K/Q8_0 | DFlash2 GGUF Q8_0/BF16 | P1 |
| GGUF Q4_K | DFlash2 safetensors BF16 | 现有 |
| MLX 4bit/6bit/8bit | DFlash2 safetensors BF16 | 现有 |
| MLX 4bit | DFlash2 GGUF Q8_0 | P1 |
| 任意 | none | P0 |
| GGUF/MLX | MTP builtin | P2 |
| GGUF/MLX | MTP external | P2 |
| GGUF/MLX | DSpark | P3 |

### 11.5 候选家族（按优先级）

1. Llama 3.1/3.2 dense（Tier B，none）；
2. Qwen2.5/Qwen3 dense（Tier B，none）；
3. Gemma 3/4（Tier B，none；vision 后续）；
4. Mistral Small 3.x（Tier B，none）；
5. DeepSeek-V3/V4 MoE（Tier B + MTP builtin）；
6. DSpark for DeepSeek-V4（Tier C）。

### 11.6 不承诺的部分

- 不做任意架构的自动 kernel 生成；
- 不保证每个 Hugging Face 模型都能加载；
- 不把 Tier C 纳入 `auto` 选择；
- 不为没有 tokenizer/chat profile 的模型猜格式。

---

## 12. CLI、API 与桌面 App

### 12.1 命令示例

```bash
# 无 draft，纯自回归
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --no-draft

# 指定 DFlash2 GGUF
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --draft-model z-lab/Qwen3.8-27B-DFlash2-GGUF \
  --draft-format gguf

# 内置 MTP
splash serve --model deepseek-ai/DeepSeek-V4-GGUF:Q4_K_M --speculation mtp

# 外部 MTP
splash serve --model ... --draft-model /models/DeepSeek-V4-MTP --speculation mtp

# 实验 DSpark
splash serve --model ... --draft-model /models/DeepSeek-V4-DSpark --speculation dspark

# 本地文件无 draft
splash serve --model local/qwen --model-dir /models/Qwen3.8-27B-4bit --no-draft
```

### 12.2 `/status`

```json
{
  "speculation": {
    "provider": "dflash2",
    "format": "gguf",
    "proposal_rows": 8,
    "state": "draft-rings",
    "draft_weights_bytes": 123456789,
    "fallback_reason": null
  }
}
```

### 12.3 `/v1/models`

```json
{
  "id": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
  "speculation": {
    "provider": "dflash2",
    "format": "gguf",
    "proposal_rows": 8
  }
}
```

### 12.4 启动日志

```text
Speculation: DFlash2 (gguf, 8 rows, draft 1.2 GiB)
Speculation: none (autoregressive; no compatible draft found: unsupported_draft_type Q5_K)
```

### 12.5 桌面 App

- Models 编辑器增加 “Speculation” 段：Auto / None / DFlash2 / MTP / DSpark；
- 新增 “Draft format” 与 “Require draft”；
- Live 页显示 provider、acceptance、fallback reason；
- 启动参数透传 `--speculation`、`--draft-format`、`--require-draft`；
- 旧配置的默认值为 `auto`，行为不变。

---

## 13. 兼容性与迁移

### 13.1 旧安装（v1 record）

- 读取器接受 `version: 1`；
- 有 `sources.draft` → 合成 DFlash2/safetensors/external plan；
- 无 `sources.draft` 的 v1 记录不存在，按 `none` 处理（防御性）；
- 不重写旧 assembly，除非用户显式重新安装。

### 13.2 旧缓存

- 旧命名空间不包含 `speculation` 行；
- 新 `none` 路径的命名空间不同，不会命中旧 draft 缓存；
- 旧 draft 缓存仍可被 v1 合成 plan 命中；
- 发布时不清空缓存，只让新命名空间自然未命中。

### 13.3 旧引擎

- 新安装的 v2 record 被旧引擎读取：旧引擎只认 version 1，会拒绝；
- 错误消息提示“reinstall Splash / rebuild it with make”，与现有
  `run_engine` 的 `EX_USAGE` 处理一致；
- 不允许旧引擎静默忽略 `speculation`，否则可能按 v1 语义错误加载。

### 13.4 回滚

- 发布采用“先读兼容、后写新格式”：P0 先让引擎能读 v2，再让安装器写 v2；
- 回滚时旧安装器仍能写 v1；旧引擎仍能读旧 assembly；
- 新 assembly 在回滚后不可用，但不会破坏旧 assembly。

### 13.5 发布顺序

1. 引擎：读取 v2 + `none` 路径（P0）。
2. 安装器：写 v2 + `--no-draft`（P0）。
3. 引擎：GGUF draft（P1）。
4. 安装器：GGUF draft 选择（P1）。
5. MTP（P2）、DSpark（P3）。
6. 文档与桌面 App 同步。

---

## 14. 测试计划

### 14.1 单元测试

新增 `dev/tests/install` 下的 `test_speculation.py`：

- 解析优先级：CLI > config > family > none；
- 冲突规则；
- v1 → v2 合成；
- record 校验（缺字段、未知 provider、format 不匹配）；
- alignment 比较（每个字段的缺失/不等）；
- 候选顺序与 fallback；
- selection link 哈希随 plan 变化。

### 14.2 引擎测试

新增 `dev/tests/engine` 下的 `speculation_test.mm`：

- `none` 的 `ModelDescriptor::valid()`；
- `none` 的 `CompositeStateLayout` 零值；
- `MemoryPlan` 的零 draft；
- `MemoryAudit` 的零分类；
- `Speculator::proposalRows()`；
- `NoneSpeculator` 写 anchor 与 `VerifyRows=1`；
- `encodeVerifyCommit` 跳过 draft commit。

### 14.3 Golden 测试

- 选 3 个 target（dense、MoE、Flint），各跑：
  - `none`；
  - `dflash2` all-rejected（`--verify-rows all` + 强制拒绝）；
  - `dflash2` 正常；
- greedy 输出逐 token 对比；
- seeded 采样同 seed 对比；
- 约束（JSON schema）与停止条件对比。

### 14.4 集成测试

- `splash serve --model ... --no-draft` 启动、发请求、`/status` 字段；
- GGUF draft 下载/加载；
- `z-lab` 候选回退；
- 不兼容 draft 的 fallback 与 `--require-draft`；
- 旧 v1 assembly 加载；
- 缓存命名空间隔离。

### 14.5 性能测试

- `none` 的 decode tok/s、TTFT、内存；
- 每个 provider 的 acceptance、drafted/accepted；
- 回归：现有 5 家族的 benchmark 不下降超过阈值（例如 2%）。

### 14.6 回归矩阵

| 维度 | 取值 |
|---|---|
| target | GGUF dense / GGUF MoE / MLX dense / MLX MoE / Flint |
| provider | none / dflash2-safetensors / dflash2-gguf / mtp-builtin / mtp-external / dspark |
| context | 4K / 32K / auto |
| sampling | greedy / seeded / constrained |
| vision | on / off |
| cache | cold / warm / persistent |

---

## 15. 实施路线

### P0：可选 draft 与 `none` 解码（1–2 周）

**目标**：任何受支持 target 可在无 draft 下服务。

**任务**：

1. `install` 下的 `speculation.py`：Plan 类型、解析、alignment。
2. `families.py`：允许 `speculation=()`，默认 `none`。
3. `upstream.py`/`local.py`：无 draft 安装；record v2。
4. `assembly.py`：v2 schema、verify。
5. `models.py`/`launcher.py`/`serve_options.py`：`--no-draft`、
   `--speculation`、`--require-draft`。
6. `ModelDescriptor`：plan、`valid()`、`inspectModelRoot`。
7. `StateLayout`：零 draft 合法。
8. `MemoryPlan`/`Audit`/`Resources`：零 draft 合法。
9. `Speculator` + `NoneSpeculator`；`Runtime::decodeAsync` 分支。
10. `Engine`：`DraftContextPlan` 可选、junction、verify budget。
11. `Status`：`speculation` 字段。
12. 测试：A1–A5、A9–A10。

**验收**：`--no-draft` 在 GGUF 与 MLX 上启动；greedy/seeded 与 oracle 一致；
内存审计通过；现有 5 家族回归通过。

**风险**：解码图分支、状态可选化、缓存命名空间。

### P1：GGUF draft 与 `z-lab`（1–2 周）

**目标**：DFlash2 draft 支持 GGUF；`z-lab` 作为候选/显式来源。

**任务**：

1. `DraftSource` 抽象；`GgufDraftSource`。
2. GGUF draft 的 config/header 读取、张量映射、类型检查。
3. split draft 支持。
4. `--draft-format`、候选顺序。
5. 测试：A6、A7、GGUF draft golden。

**验收**：Q4_K target + Q8_0 draft 可服务；safetensors target + GGUF draft
可服务；不支持的 draft 类型回退。

### P2：内置/外部 MTP（2–3 周）

**目标**：DeepSeek 风格 MTP head 可用。

**任务**：

1. `MtpLayout`、`MtpWeights`、`MtpSpeculator`。
2. GGUF/MLX/Model Optimizer 的 head 提取。
3. alignment 与外部 head 安装。
4. MTP 状态与 KV 提交。
5. 测试：A8、MTP golden、不匹配回退。

**验收**：内置与外部 MTP 各跑通一个模型；缺 head 时回退。

### P3：DSpark（3–4 周）

**目标**：实验性 DSpark bundle 可用。

**任务**：

1. Bundle ABI、config、alignment。
2. 并行 backbone + Markov head。
3. 置信度调度接入 `VerifyRows`/`VerifyBudget`。
4. 实验开关与状态。
5. 测试：ABI 拒绝、acceptance、性能。

**验收**：在 release prompts 上 acceptance 不低于 DFlash2，才转正。

### P4：family pack 与模型扩展（持续）

**目标**：Tier B 家族零引擎改动接入。

**任务**：

1. `install` 下的 `families.json` schema。
2. `GenericDenseLayout`（Tier B）。
3. tokenizer/chat profile 选择。
4. 首批 3 个家族。
5. 文档与测试。

**验收**：新增一个 Tier B 家族只改 pack 与测试。

### 依赖图

```text
P0 ──► P1 ──► P2 ──► P3
 │
 └────► P4（可与 P2/P3 并行）
```

P0 是硬前置；P1 可与 P2 部分并行；P3 依赖 P2 的 state 抽象；P4 依赖 P0 的
无 draft 路径。

---

## 16. 风险登记册

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 无 draft 输出与 oracle 不一致 | 中 | 高 | golden 测试、固定 seed、约束测试 |
| 解码图 8 行浪费导致 `none` 性能差 | 高 | 中 | P0 接受；P1 评估 1 行 arena |
| 状态可选化引入内存/缓存 bug | 中 | 高 | 零值单测、缓存命名空间隔离 |
| GGUF draft 类型覆盖不足 | 中 | 中 | 首批限定类型；明确回退 |
| MTP head 张量命名差异 | 高 | 中 | 每家族映射表 + 严格校验 |
| DSpark ABI 不稳定 | 高 | 中 | 实验开关、版本拒绝 |
| 旧引擎读新 record 失败 | 中 | 中 | 发布顺序、清晰错误 |
| 模型范围扩张带来质量回归 | 中 | 高 | Tier 分级、`auto` 不选 Tier C |
| 供应链/远端代码 | 低 | 高 | 只读数据、不执行代码 |

---

## 17. 决策记录（ADR）

- **ADR-001**：`none` 是一等 provider，不是“特殊分支”。
  理由：统一 plan、状态、内存、缓存、状态字段；避免 if-else 扩散。
- **ADR-002**：plan 写入 `model.json`，属于安装身份。
  理由：可复现、可缓存、可回滚；回退原因属于运行时。
- **ADR-003**：provider/format 由内容决定，不由仓库名决定。
  理由：`z-lab`、`incoai`、本地目录都应走同一校验。
- **ADR-004**：默认 fallback 为 `none`，`--require-draft` 为显式严格模式。
  理由：可用性优先；基准/CI 需要严格性。
- **ADR-005**：不在 host 反量化不支持的 GGUF draft 类型。
  理由：性能与内存不可控；失败要显式。
- **ADR-006**：模型扩展用 family pack，不做“通用 loader 吞一切”。
  理由：kernel 形状、tokenizer、chat 语义都需要显式声明。
- **ADR-007**：`CompositeStateLayout` 的 draft 部分允许全零。
  理由：无 draft 是正常状态，不是缺失。
- **ADR-008**：缓存命名空间加入 plan。
  理由：有/无 draft 的状态不可互换。

---

## 18. 开放问题

1. `none` 是否值得单独缩小 decode arena（1 行）？还是 P0 先用 8 行物理形状？
2. `--verify-rows` 对 `none`/MTP 是否保留？建议变成 provider 选项或忽略。
3. `tokenizer_digest` 是否对外部 DFlash2 也强制？
4. `z-lab` 的官方镜像列表由 registry 维护，还是只允许显式 `--draft-model`？
5. DSpark 的 Markov 状态放 `CompositeStateLayout`，还是复用 draft window？
6. MTP 的 `proposal_rows` 上限是否固定 8，还是按 head 的 `k` 动态？
7. 无 draft 的持久缓存是否应该默认关闭（因为没有复用价值）？
8. 桌面 App 的默认值：`auto` 还是 `none`？

---

## 19. 附录

### A. 无 draft 的 `model.json`

```json
{
  "version": 2,
  "model": "mlx-community/Qwen3.8-27B-4bit",
  "family": "Qwen3.8-27B",
  "target_format": "mlx-affine",
  "vision_format": "safetensors",
  "speculation": {
    "version": 1,
    "provider": "none",
    "format": "none",
    "source": {"kind": "none"},
    "proposal_rows": 1,
    "state": "none",
    "fallback": "none"
  },
  "sources": {"target": {"repo": "mlx-community/Qwen3.8-27B-4bit", "revision": "..."}},
  "files": {}
}
```

### B. 错误消息

```text
error: --speculation none and --draft-model are exclusive
error: --require-draft needs a draft provider, not --speculation none
error: --draft-format gguf needs a draft repository that publishes GGUF
error: the DFlash2 draft z-lab/... is incompatible with Qwen3.8-27B: draft config hidden_size 4096 does not match the target's 5120
Warning: cannot use the Qwen3.8-27B DFlash2 draft; serving without speculation: unsupported draft type Q5_K
error: the model's record version 2 requires a newer engine; rebuild it with make
```

### C. 迁移检查清单

- [ ] 旧 v1 assembly 能读；
- [ ] 新 v2 assembly 能读；
- [ ] 旧缓存不命中新 `none`；
- [ ] 新 `none` 不命中旧 draft 缓存；
- [ ] `--no-draft` 不访问 draft 仓库；
- [ ] `--require-draft` 在不兼容时失败；
- [ ] 现有 5 家族 golden 不变；
- [ ] 桌面 App 旧配置默认 `auto`；
- [ ] 文档与 `--help` 同步；
- [ ] 发布说明写明“draft 可选”。

### D. 术语表

| 术语 | 含义 |
|---|---|
| SpeculationPlan | 一次安装的推测配置 |
| Provider | 推测算法 |
| Format | 权重容器 |
| Source | 权重来源 |
| Alignment | draft 与 target 的共享属性 |
| Proposal width | 一个周期最多提出的行数 |
| Fallback | provider 不可用时的行为 |
| Family pack | 声明式家族描述 |
| Tier A/B/C | 模型支持等级 |


## 20. 实现状态（当前检出）

- **已实现（P0 的大部分）**：
  - `model.json` version 2 的 `speculation` 对象，provider `none`；
  - 安装器 `--no-draft` / `--speculation none` / `--require-draft`；
  - `auto` 在 draft 不可解析时告警并降级为 `none`；
  - 引擎读取 v2 记录、可选 `draft/`、`SpeculationPlan::none`；
  - 零 draft 状态布局、内存计划与审计；
  - `none` 解码：1 行 verify、无 draft 图、无 draft 提交；
  - `/status` 的 `speculation` 字段；
  - 安装器与引擎的无 draft 测试，以及既有的 CPU 测试。
- **P1 已开始**：
  - `families.py` 的 `draft_mirrors` 与 `draft_candidates()`：先解析
    `incoai`，失败再尝试 `z-lab` 等镜像；
  - `install/draftgguf.py`：安装时把 BF16/F16/F32/Q8_0 的 DFlash2 GGUF 转成
    引擎读取的 BF16 safetensors，`draft/config.json` 保留为 sidecar；
  - 安装器自动识别 draft 仓库中的 GGUF，无需额外参数。
- **P2 已开始（计划、元数据与界面）**：
  - GGUF 的 `nextn_predict_layers` 与 MLX 的 `num_nextn_predict_layers`
    会记录到 `model.json` 的 `mtp: {"layers": N}`；
  - `--speculation mtp` 接受带内置 MTP 头的目标，记录
    `provider: mtp`、`source.kind: builtin`，但不加载 head；
  - 该计划按 `none` 服务，`/status` 报告
    `speculation.provider: mtp` 与 `fallback_reason: mtp execution is not implemented...`；
  - 桌面 App 的 Live 卡片显示“内置 MTP N 层”与回退原因；
  - 外部 MTP 头目录/仓库会作为 plan 数据安装到 `draft/`，其 `alignment`
    会与 target 校验，但同样不执行；
  - MTP head 的实际加载与提议尚未实现。
- **未实现（P1 剩余 / P2 起）**：其它量化 GGUF draft（Q4_K/Q6_K 等）的直接
  读取、MTP 执行、DSpark、family pack。
- **已知取舍**：`none` 仍复用 8 行物理 decode arena 与 1 行 verify；draft
  状态布局为零但 `CompositeStateLayout` 的接口未改成 optional；这些是 P1
  的性能优化点，不影响正确性。
