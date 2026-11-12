# 待办与遗留

1.9.3 代码整理与性能优化完成后,以下工作尚未完成。按建议优先级排列,每项都给出位置与复现方法,便于下一个会话直接上手。

## 1. block indexer 的 pool 竞态

位置:`runtime/metal/kernels/shared/block_indexer.metal`(`block_indexer_pool`,自第 106 行起)。现象:池化的 key 没有旋转到 block 起点;rollback 场景下,被拒绝行已完成的 block 会丢失下一 step 的 key。

该问题只在原生门禁所用的验证环境(`MTL_SHADER_VALIDATION=1`,dev/native.mk:581)下出现:

```bash
# 通过
./build/engine-tests/qwen4-kernels build/engine-tests/production-and-test.metallib

# 10 个失败
MTL_SHADER_VALIDATION=1 ./build/engine-tests/qwen4-kernels \
    build/engine-tests/production-and-test.metallib
```

结果依赖验证层,说明是**同步缺陷**而非 kernel 算术错误:最可能是某行完成 block 的写入与下一步读取的旋转之间缺少 `threadgroup_barrier`,或读取了同一命令中后续 dispatch 才写入的内存。相关 kernel 由 Qwen3.8-Flash-Next(d6003397)与 YaRN(40470bbd)引入,可缩小排查范围。

## 2. 内存规划测试中的自动磁盘 tier

位置:`dev/tests/engine/runtime_resources_test.mm:1105`,报错 "the automatic tier did not start beside its staging, or the staging cost no context"。与第 1 项不同,它在**默认模式也失败**,因此是本机内存规划的真实结果,而非竞态:

```bash
./build/engine-tests/runtime-resources build/engine-tests/production-and-test.metallib
```

测试用 `start().minimumRequired` 构造预算,所以如果本机的规划把自动 tier 判到阈值以下,这里如实失败。需要先判断:是测试的预算不合理,还是 tier 的阈值不合理。

## 3. OpenAI dialect 的成对重复

位置:`server/dialects/openai.py:145-234`。`complete`/`complete_choices` 与 `text_completion`/`text_completion_choices` 近乎逐行重复;且 `complete_choices` 为每个 choice 构建完整响应(usage、metrics、timings、compaction)后只保留第一个。抽一个"单 choice 构建"函数即可同时消除重复与浪费。

## 4. 已测量但未动的引擎项

- **spilled-KV decode 的逐层 host step**(`runtime/model/KvBlockCache.cpp:359`、`runtime/model/QwenTarget.cpp:448`):每个注意力层把命令拆成独立 buffer,满宽度下每周期约 96 个额外 buffer、48 次 runner 唤醒,外加逐层目录查找。`KvReadStats`(`KvBlockCache.cpp:398-401`)已记录 resolve/pickup/resume 分布——**先测量再动**,因为 block list 由每层 indexer 在注意力之前产出。
- **在 GPU 运行时预构建下一周期的图**(`Engine.cpp:324-340`):命令完成后的调度、准入、建图、编码、提交全部串行,GPU 在此期间空闲。要做需要双份 arena lane 或参数延迟绑定。

## 5. 只能在模型环境验证的项

- **decode 图复用的逐位门禁**:在真实 target 上跑 `--verify-rows all` 与 CPU 对比。现有 CPU 测试覆盖了复用的记账逻辑,但未覆盖真实 plan。
- **logprob kernel 对真实词表**:metal 测试在 1,003 词表上对比 CPU 扫描;真实 152K 请求才能确认 float 分母的误差量级。
- **完整 `make test-engine-metal` 门禁**:第 1 项修复前无法通过;`qwen4-kernels` 之后的测试组(attention plan、linear plan、KV page tier、model loading、qwen state、q8、gdn、dflash、draft selector、target sampling、vision、handoff)自本次改动后尚未全部跑过。
- **改动前后 decode/prefill 速率对比**:同 prompt、同并发,方法见 [性能](performance.md)的基准记录。

## 6. 明确不做

`server/protocol.py`、`server/runtime.py` 与 `runtime/model/Runtime.cpp` 中的 `Runtime::Impl` 保持现状。把 1,500 行拆成两个 700 行文件没有可测量的收益,而 `Runtime::Impl` 是全树最热的代码;真正值得拆的边界是 `server.py` 那一处,已经完成。

## 7. 可选草稿的剩余阶段(speculation 提案)

提案的 P0 大部分与 P1 的开头已在树上:安装记录 v2 的 `speculation` 对象与 provider `none`(`install/assembly.py:204`)、`--no-draft`/`--speculation`/`--require-draft`、`auto` 在 draft 不可解析时降级、引擎读取可选 `draft/` 与零 draft 的状态布局和内存审计、`none` 解码、`/status` 的 `speculation` 字段;`families.py` 的 `draft_mirrors`/`draft_candidates()` 先解析 `incoai` 再试 `z-lab` 镜像,`install/draftgguf.py` 把 BF16/F16/F32/Q8_0 的 DFlash2 GGUF 转成引擎读取的 safetensors。内置与外部 MTP(原 P2)已删除(70d957dc)。

尚未实现,按提案的阶段:

- **P1 剩余——其它量化的 GGUF draft**(Q4_K/Q6_K 等):现在量化 draft 被 `install/draftgguf.py:168` 按名拒绝,只有上述四类会转换。需要 draft kernel 对这些投影的覆盖与每类型受支持列表,拒绝时给出类型名。
- **P3——family pack**:声明式注册表与 Tier B 通用布局,出口是"加一个 Tier B 家族不改引擎源码"。

DSpark(提案的 P2)已从本发布移除:安装器不再把 `dspark` 当作一个模式(`install/models.py` 的 `SPECULATION_MODES`),schema 不再接受它的 plan 与 state(`install/assembly.py` 的 `PROVIDERS`),引擎的 `SpeculationKind` 只剩 `None` 与 `DFlash2`,桌面 app 不再显示它的名字。提案原文里的 DSpark 章节随历史保留,不再是待办。

已知取舍(P1 的性能优化点,不影响正确性):`none` 仍复用 8 行物理 decode arena 与 1 行 verify;draft 状态布局为零但 `CompositeStateLayout` 的接口未改成 optional。

完整设计在 git 历史:`git show 68ce4e29^:docs/speculation.zh.md`(中文,1623 行,数据模型、provider 规格、安装器与运行时改造、测试计划、风险登记册、开放问题)与 `speculation.md`(英文,537 行)。
