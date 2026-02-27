## Vulcan 架构与目录结构概览

本文件总结 Vulcan 项目的整体架构与目录结构，配合 `reconstruction_plan.md` 以及各阶段的 `phase*.md` 使用。

- 核心代码集中在 `src/vulcan/` 下，按领域分为：
  - `vulcan.framework`：训练框架（模型、数据集、损失、优化器、表示等）；
  - `vulcan.lang`：语言/代码解析与分析；
  - `vulcan.datacollection`：数据/漏洞收集工具；
  - `vulcan.cli`：训练/验证/benchmark/导出的命令行入口；
  - `vulcan.services`：后端服务与 API 封装。
- 顶层辅助目录：
  - `tools/`：薄封装 CLI 脚本，调用 `vulcan.cli.*`；
  - `scripts/`：启动/运维脚本，调用 `vulcan.services.*`；
  - `tests/`：按包结构镜像划分的测试目录；
  - `docs/`：用户与开发文档。

更多细节（包括每个子模块的职责与迁移方案），可参考根目录的 `reconstruction_plan.md` 以及 `phase1.md` ~ `phase4.md`。 

