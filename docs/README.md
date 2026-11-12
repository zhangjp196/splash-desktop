# 文档

面向使用与运维:

- [服务器](server.md):构建与运行、每个选项、网络与鉴权、文件布局、引擎重启与休眠。
- [API](api.md):四个生成 API 与控制端点的请求字段、采样、工具调用、结构化输出、图像与 PDF、错误。
- [模型](models.md):上游加载与准备、GGUF 与 MLX 目标、草稿、视觉、本地模型目录。
- [性能](performance.md):1.0 发布的基准记录、GGUF 对 llama.cpp 的精度与速度对比、24 GB Mac 上的小 GGUF。

面向维护:

- [内部结构](internals.md):代码与 API 边界、每个模块的职责、内核与权重加载、准入与驱逐。
- [校验](validate.md):`make check` 与各真实模型门禁,以及发布前的检查单。
- [打包](package.md):发布归档、桌面 app、Homebrew、测试发布通道。
- [待办与遗留](remaining-work.md):1.9.3 代码整理与性能优化之后尚未完成的工作,以及复现每个问题的方法。

以上七篇取自 `68ce4e29` 之前的提交,于 2026-11-12 对照当前树核对过:每篇点名的模块、选项、make 目标与端点都还在,根 README 指向它们的 16 处链接与七篇之间的互链全部可达。`speculation.md` 与 `speculation.zh.md` 未取回:它们是提案,其中仍未实现的部分已提炼进 [待办与遗留](remaining-work.md)第 6 项,完整设计从历史取回(`git show 68ce4e29^:docs/speculation.zh.md`)。
