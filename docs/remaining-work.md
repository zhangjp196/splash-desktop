# 待办与遗留

1.9.3 代码整理与性能优化完成后,以下工作尚未完成。按建议优先级排列,每项都给出位置与复现方法,便于下一个会话直接上手。

## 1. block indexer 的 pool 竞态

位置:`runtime/metal/kernels/shared/block_indexer.metal`(`block_indexer_pool`)。现象:验证环境(`MTL_SHADER_VALIDATION=1`,dev/native.mk:581)下,池化 key 读回**全 0**——pool 对 extent(经 index table 条目里的 GPU 地址)的写没有落地;无验证时同一测试通过。失败在多次运行间非确定(pool 各例有时全过、有时全零),rollback 一例则稳定失败。

已修的一处(2026-11-12):pool 对 pending keys 的读越界——循环的 `t` 跑到 `kBlock - 1 = 3`,而 pending 只有 `kPending = 3` 个槽。语义上 t=3 的读总被丢弃(完成的 block 末 token 必是本步的行,`position >= first`),但边界检查门禁会为此中止 dispatch。已按 `min(t, kPending - 1)` 收住;无验证的门禁通过。

2026-11-12 的二分把症状钉死了:验证下 pool 的 kernel **完整执行**(dispatch 末尾落在绑定缓冲里的标记读得到),但它对 extent 的 addressed 写**整体不落地**——extent 读回全 0,无验证时同一测试通过,`MTL_SHADER_VALIDATION=2` 同样。已排除:pending 读越界(已修)、对齐、extent 大小、barrier 与写函数的形态(换成直接写仍被丢)、addressed 写机制本身(store kernel 经地址写 KV,attention-variants 在验证下通过)、绑定顺序、setTableReach(无人调用,store 也不靠它)。结论方向:验证层对这个 dispatch 的 addressed 写的丢弃,是 Metal 验证层的伪影而非 kernel 缺陷;下一步是给 Apple 的最小复现(qwen4-kernels 的 pool 一例即可),或把该例的 extent 改经 `HostKvExtents` 分配对照一次。相关 kernel 由 Qwen3.8-Flash-Next(d6003397)与 YaRN(40470bbd)引入。

## 2. 已测量但未动的引擎项

- **spilled-KV decode 的逐层 host step**(`runtime/model/KvBlockCache.cpp:341`、`runtime/model/QwenTarget.cpp:448`):每层一个 resolve host step(`SpilledKvReads::verify` → `KvBlockCache::addResolve`),结构账已对上——awake host step 一个加 2 个命令 buffer(`MetalBackend.mm` 的 `addedBuffers`),48 层的 decode 命令占队列 512 个里的 97 个、48 次 runner 唤醒,与原记的"约 96 个额外 buffer、48 次唤醒"相符。host step 串行:GPU 等每层的 resolve 返回才跑下一层,每层的往返是 pickup + work + resume。
  **决策规则(模型环境一次读 /status 即可判)**:每周期 host step 成本 = `model.attention_layers` × (`kv_spill.blockCache.host_steps.pickup_ms.p50` + `work_ms.p50` + `resume_ms.p50`),与 `metrics.decode_cycle_ms` 比;占比小就不动。后端测试给的下限:pickup 9.4 µs + resume 68.7 µs 一步(metal-backend 的 host step 测量),按 48 层算 ≈ 3.7 ms 一周期;resume 里下一 kernel 的启动占大头,能砍多少要靠实测的 resume 分布说。**显然的改法被结构挡住**:block list 由每层 indexer 在该层注意力之前产出,resolve 无法并成每周期一个 host step;替代方案是把目录搬上 GPU 做成 kernel,或把 miss 的读挪到后台(改变失败语义,`SsdReaders::runBatch` 的注释是现成的对照)。`work_ms` 里 miss 的 SSD 读占大头时,先看 `read_ms` 与 `resolve_ms` 的差。
- **在 GPU 运行时预构建下一周期的图**(`Engine.cpp:324-350`):命令完成后的 sweep、调度、准入、建图、编码、提交全部串行,GPU 在此期间空闲。**决策规则**:`loop.max_tick_ms`(串行尾的实测)与 `decode_cycle_ms` 比;占比小就不动。要做的两条路仍是原记的双份 arena lane(内存换并行,arena 是各命令共享的缓冲,下一命令的编码不能复用当前命令正在用的)与参数延迟绑定(复杂度换并行);前者先量 arena 的字节(`memory_plan` 的 `shared_prefill_bytes` + `shared_decode_bytes` 是每命令一份,翻倍就是这么多)。

## 3. 只能在模型环境验证的项

- **decode 图复用的逐位门禁**:在真实 target 上跑 `--verify-rows all` 与 CPU 对比。现有 CPU 测试覆盖了复用的记账逻辑,但未覆盖真实 plan。
- **logprob kernel 对真实词表**:metal 测试在 1,003 词表上对比 CPU 扫描;真实 152K 请求才能确认 float 分母的误差量级。
- **完整 `make test-engine-metal` 门禁**:第 1 项修复前无法通过;`qwen4-kernels` 之后的测试组(attention plan、linear plan、KV page tier、model loading、qwen state、q8、gdn、dflash、draft selector、target sampling、vision、handoff)自本次改动后尚未全部跑过。
- **改动前后 decode/prefill 速率对比**:同 prompt、同并发,方法见 [性能](performance.md)的基准记录。

## 4. 明确不做

`server/protocol.py`、`server/runtime.py` 与 `runtime/model/Runtime.cpp` 中的 `Runtime::Impl` 保持现状。把 1,500 行拆成两个 700 行文件没有可测量的收益,而 `Runtime::Impl` 是全树最热的代码;真正值得拆的边界是 `server.py` 那一处,已经完成。

## 5. 可选草稿的剩余阶段(speculation 提案)

提案的 P0 大部分与 P1 的开头已在树上:安装记录 v2 的 `speculation` 对象与 provider `none`(`install/assembly.py:204`)、`--no-draft`/`--speculation`/`--require-draft`、`auto` 在 draft 不可解析时降级、引擎读取可选 `draft/` 与零 draft 的状态布局和内存审计、`none` 解码、`/status` 的 `speculation` 字段;`families.py` 的 `draft_mirrors`/`draft_candidates()` 先解析 `incoai` 再试 `z-lab` 镜像,`install/draftgguf.py` 把 BF16/F16/F32/Q8_0 的 DFlash2 GGUF 转成引擎读取的 safetensors。内置与外部 MTP(原 P2)已删除(70d957dc)。

尚未实现,按提案的阶段:

- **P1 剩余——其它量化的 GGUF draft**(Q4_K/Q6_K 等):现在量化 draft 被 `install/draftgguf.py:168` 按名拒绝,只有上述四类会转换。需要 draft kernel 对这些投影的覆盖与每类型受支持列表,拒绝时给出类型名。
- **P3——family pack**:声明式注册表与 Tier B 通用布局,出口是"加一个 Tier B 家族不改引擎源码"。

DSpark(提案的 P2)已从本发布移除:安装器不再把 `dspark` 当作一个模式(`install/models.py` 的 `SPECULATION_MODES`),schema 不再接受它的 plan 与 state(`install/assembly.py` 的 `PROVIDERS`),引擎的 `SpeculationKind` 只剩 `None` 与 `DFlash2`,桌面 app 不再显示它的名字。提案原文里的 DSpark 章节随历史保留,不再是待办。

已知取舍(P1 的性能优化点,不影响正确性):`none` 仍复用 8 行物理 decode arena 与 1 行 verify;draft 状态布局为零但 `CompositeStateLayout` 的接口未改成 optional。

完整设计在 git 历史:`git show 68ce4e29^:docs/speculation.zh.md`(中文,1623 行,数据模型、provider 规格、安装器与运行时改造、测试计划、风险登记册、开放问题)与 `speculation.md`(英文,537 行)。
