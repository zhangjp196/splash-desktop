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
- **改动前后 decode/prefill 速率对比**:同 prompt、同并发,方法见原 `performance.md`(已移至 git 历史)。

## 6. 明确不做

`server/protocol.py`、`server/runtime.py` 与 `runtime/model/Runtime.cpp` 中的 `Runtime::Impl` 保持现状。把 1,500 行拆成两个 700 行文件没有可测量的收益,而 `Runtime::Impl` 是全树最热的代码;真正值得拆的边界是 `server.py` 那一处,已经完成。
