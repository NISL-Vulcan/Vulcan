# Vulcan 14 次 Commit 覆盖矩阵与结论

## 执行口径

- 基线：当前 `HEAD`，按提交影响域做回归覆盖（非逐提交 checkout）。
- 环境：`conda run -n vulcan`。
- 主命令：
  - `python -m pytest -q`
  - `python -m pytest --cov=src/vulcan --cov-report=term-missing --cov-report=xml:tests/reports/coverage.xml -q`
- 结果：`308 passed, 6 skipped`，`TOTAL 64%`。

## Commit -> 模块 -> 测试映射

### 1) `c63d824` Phase 1：确立包结构但不移动脚本

- 影响域：`src/vulcan/*` 包结构、导入路径基座。
- 覆盖测试：
  - `tests/test_smoke/test_imports.py`
  - `tests/test_framework/test_common.py`
  - `tests/test_services/test_backend.py`
  - `tests/test_datacollection/test_import.py`
- 结论：覆盖了 src 布局导入与包级入口稳定性。

### 2) `942e508` Phase 2：统一导入路径与运行方式

- 影响域：`framework/*`、`tools/cli`、`lang/cParser` 相关导入统一。
- 覆盖测试：
  - `tests/test_framework/test_model.py`
  - `tests/test_framework/test_dataset.py`
  - `tests/test_framework/test_utils.py`
  - `tests/test_framework/test_representations_vectorizer.py`
  - `tests/test_cli.py`
- 结论：关键导入链路与工厂分发已覆盖。

### 3) `601d78f` Phase 3：拆分 CLI 与库逻辑

- 影响域：`src/vulcan/cli/train.py`、`src/vulcan/cli/val.py`、`tools/*`。
- 覆盖测试：
  - `tests/test_cli.py`（`ordered_load`、`convert_output`、`train/val/benchmark/export` 的 `cli_main` 调用链）
- 结论：CLI 入口参数解析与调用路径已覆盖。

### 4) `f96d7c1` Phase 4：整理服务层与运维脚本

- 影响域：`src/vulcan/services/*` 基础封装与后端入口。
- 覆盖测试：
  - `tests/test_services/test_backend.py`
  - `tests/test_services/test_apis.py`
- 结论：服务层代理与后端加载入口关键分支已覆盖。

### 5) `997eab9` Phase 5：测试与文档结构完善

- 影响域：文档与测试目录结构完善。
- 覆盖测试：
  - 全量 `tests/` 可发现性验证（`python -m pytest -q` 通过）
- 结论：结构性改动已由全量回归验证。

### 6) `164315c` 新增 pyproject、清理 sys.path hack、修正 train 依赖

- 影响域：打包入口与 `vulcan.cli.train` 依赖关系。
- 覆盖测试：
  - `tests/test_cli.py`（`train.cli_main` 流程）
  - `tests/test_smoke/test_imports.py`
- 结论：核心运行路径在安装态/包态导入下可用。

### 7) `9d6a0a1` 迁移测试、拆分 utils、文档角色调整

- 影响域：`framework/utils/*`、测试迁移后行为一致性。
- 覆盖测试：
  - `tests/test_framework/test_utils.py`
  - `tests/test_framework/test_utils_ddp.py`
  - `tests/test_framework/test_utils_training.py`
  - `tests/test_framework/test_utils_converting.py`
  - `tests/test_framework/test_utils_vocabulary.py`
  - `tests/test_framework/test_utils_clean_gadget.py`
- 结论：utils 族改动已被参数化与异常分支覆盖。

### 8) `09fa630` pytest 工具链与 smoke tests

- 影响域：测试执行链路、基础 smoke 验证。
- 覆盖测试：
  - `tests/test_smoke/test_imports.py`
  - 全量 `python -m pytest -q`
- 结论：测试工具链可用并持续回归。

### 9) `82d214d` src 布局生效、安装态导入修复、可选组件容错

- 影响域：`framework/__init__.py`、`representations/*`、`services/apis.py`。
- 覆盖测试：
  - `tests/test_framework/test_common.py`
  - `tests/test_framework/test_representations_vectorizer.py`
  - `tests/test_services/test_apis.py`
  - `tests/test_framework/test_models_extra.py`（可选依赖 skip 路径）
- 结论：可选依赖缺失时 skip/降级行为可复现。

### 10) `6dff628` 2.0 大规模服务层重构与脚本迁移

- 影响域：`services/*_app.py`、`backend_server_app.py`、`datacollection/*`、`cli/*`。
- 覆盖测试：
  - `tests/test_services/test_backend.py`
  - `tests/test_services/test_apis.py`
  - `tests/test_datacollection/test_import.py`
  - `tests/test_datacollection/test_data_collector.py`
  - `tests/test_cli.py`
- 结论：核心服务封装与数据收集流程基础路径已覆盖。

### 11) `53afc36` gitignore 与跟踪文件清理

- 影响域：仓库卫生与忽略文件。
- 覆盖测试：不属于单元测试对象（无运行时逻辑）。
- 结论：在覆盖矩阵标记为“非代码逻辑提交”。

### 12) `a57021c` 环境依赖与配置调整

- 影响域：`pyproject.toml`、`vulcan.yaml`、多模型/数据集导入细节。
- 覆盖测试：
  - `tests/test_framework/test_model.py`
  - `tests/test_framework/test_models_extra.py`
  - `tests/test_framework/test_dataset.py`
  - `tests/test_smoke/test_imports.py`
- 结论：依赖调整后的核心工厂装配路径已覆盖。

### 13) `49a1fec` 细节修复及训练测试

- 影响域：`framework/dataset.py`、`framework/model.py`、`datasets/vddata.py`、`metrics.py`、`cli/train.py`。
- 覆盖测试：
  - `tests/test_framework/test_dataset.py`
  - `tests/test_framework/test_dataset_dataloader.py`
  - `tests/test_framework/test_model.py`
  - `tests/test_framework/test_metrics.py`
  - `tests/test_framework/test_pretrained.py`
  - `tests/test_cli.py`
- 结论：数据流、模型装配与指标路径覆盖完整。

### 14) `65b3291` 纯净版（删除训练测试数据）

- 影响域：样例数据与产物清理，smoke 配置。
- 覆盖测试：
  - `tests/test_framework/test_dataset.py`
  - `tests/test_framework/test_dataset_dataloader.py`
  - `tests/test_framework/test_utils_clean_gadget.py`
- 结论：当前测试均基于最小夹具与 mock，不依赖大体积训练产物。

## 覆盖完整性结论

- 14 次提交均已在矩阵中可追溯。
- 其中 `53afc36` 属于仓库卫生提交，按“非代码逻辑”处理。
- 可选依赖导致的 `skip` 已被显式记录，不影响主流程回归结论。
- `framework/representations/ast_graphs.py` 已通过 stub + 访客/构建器用例补齐到约 `91%` 覆盖。
- `framework/models/mvdetection.py` 通过 stub 化 `torch_geometric` 依赖与模型侧用例，已从 `10%` 提升到 `100%`，并修复了 `CustomGraphConvolutionLayer__` 初始化的 `super()` 调用缺陷。

## 当前残余风险

- `services/backend_server_app.py` 与 `dataset_optimization_*_app.py` 体量大，当前以入口/代理路径覆盖为主，深层路由逻辑覆盖仍可继续提升。
- `services/backend_server_app.py` 已从 `0%` 提升到 `30%`，当前主要覆盖状态/摘要函数、轻路由、状态日志接口与配置/启动接口分支，深层训练与子任务编排分支仍需继续补测。
- `services/data_collection_api_app.py` 与 `services/dataset_optimization_api_app.py` 已从 `0%` 提升到 `52%` 与 `27%`，服务层 0% 热点进一步减少。
- `services/dataset_optimization_server_app.py` 与 `services/resource_updater.py` 已从 `0%` 提升到 `31%` 与 `76%`，服务层剩余 0% 进一步收敛。
- `framework/datasets/linevul.py` 与 `framework/datasets/regvd.py` 已补齐核心转换函数覆盖（聚焦回归下分别到 `58%` 与 `64%`），后续可继续补 `Dataset.__init__/__getitem__` 的文件读取与采样分支。
- `services/backend_server.py` 已补齐启动封装路径（聚焦回归 `100%`），后续主要增量仍在 `backend_server_app.py` 深层接口。
- `cli/benchmark.py` 与 `cli/export.py` 已通过依赖隔离补测（分别提升到 39% 和 72%），但 `benchmark.main` 的 GPU 性能循环与真实设备分支仍未全覆盖。
- `tests/test_services/test_backend.py` 与 `tests/test_services/test_apis.py` 的多语言文案断言已做兼容，当前全量回归基线为 `305 passed, 6 skipped`、`TOTAL 61%`。
