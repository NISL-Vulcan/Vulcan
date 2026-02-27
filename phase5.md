## Phase 5：测试与文档结构完善

本阶段对应 `reconstruction_plan.md` 中 **Phase 5：测试与文档结构完善** 的设计目标：

- 在项目根目录下建立标准的 `tests/` 目录结构，并按包结构划分子目录；
- 在 `docs/` 目录下补充基础文档骨架（架构说明与使用指南）；
- 不强制一次性写完所有测试用例和详细文档，而是先搭好“挂载点”，方便后续迭代完善。

---

### 1. 测试目录结构：`tests/`

根据 `reconstruction_plan.md` 中的规划，`tests/` 目录需要大致镜像 `src/vulcan/` 的结构。本阶段完成了以下结构搭建：

- 新增测试包根目录：
  - `tests/__init__.py`

- 新增按领域划分的子目录（每个目录一个 `__init__.py` 占位）：
  - `tests/test_framework/__init__.py`
    - 未来用于放置与 `vulcan.framework` 相关的单元/集成测试。
  - `tests/test_lang/__init__.py`
    - 未来用于放置与 `vulcan.lang` 相关的解析与分析测试。
  - `tests/test_datacollection/__init__.py`
    - 未来用于放置与 `vulcan.datacollection` 相关的数据采集测试。
  - `tests/test_services/__init__.py`
    - 未来用于放置与 `vulcan.services`（后端服务与 API）相关的测试。

> 当前阶段只创建了包结构和占位文件，没有添加具体 `test_*.py` 测试用例，避免在依赖环境尚未完全就绪时强行引入可能失败的导入测试。后续可按模块优先级逐步在对应子目录下补充测试文件。  

典型未来结构示例（尚未创建的文件，仅作为参考）：

```text
tests/
  test_framework/
    test_models.py
    test_datasets.py
  test_lang/
    test_cparser.py
  test_datacollection/
    test_nvd_fetcher.py
  test_services/
    test_backend_healthcheck.py
```

---

### 2. 文档结构完善：`docs/`

在原有 `docs/readme.md` 的基础上，本阶段为整体架构与使用方式新增了两份文档骨架：

#### 2.1 `docs/architecture.md`

新建文件 `docs/architecture.md`，用于从宏观上描述项目的架构与目录布局。主要内容包括：

- 简要说明核心代码集中在 `src/vulcan/` 下，并按领域拆分为：
  - `vulcan.framework`
  - `vulcan.lang`
  - `vulcan.datacollection`
  - `vulcan.cli`
  - `vulcan.services`
- 顶层辅助目录的角色：
  - `tools/`：薄封装 CLI；
  - `scripts/`：启动与运维脚本；
  - `tests/`：测试目录；
  - `docs/`：文档目录。
- 引导读者进一步查看 `reconstruction_plan.md` 以及 `phase1.md` ~ `phase4.md` 获取更详细的重构与设计信息。

该文档目前是架构的高层总览，后续可以在每个模块下逐步补充更细的说明（如子模块职责、关键类与函数等）。

#### 2.2 `docs/usage.md`

新建文件 `docs/usage.md`，作为项目的基础“使用指南”，主要内容包括：

- **环境准备**：

  ```bash
  cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan
  pip install -r requirements_backend.txt
  export PYTHONPATH=src:$PYTHONPATH
  ```

- **训练与验证运行方式**：

  ```bash
  # 训练
  python tools/train.py --cfg configs/custom.yaml

  # 验证
  python tools/val.py --cfg configs/custom.yaml

  # benchmark
  python tools/benchmark.py --cfg configs/custom.yaml
  ```

  对应的 Python 入口（来自 Phase 3 的 `vulcan.cli` 拆分）：

  ```python
  from vulcan.cli.train import cli_main as train_cli
  from vulcan.cli.val import cli_main as val_cli
  from vulcan.cli.benchmark import cli_main as benchmark_cli
  ```

- **后端服务与 API 运行方式**：

  ```bash
  # 启动后端
  python scripts/start_backend.py

  # 启动综合服务
  python scripts/start_services.py
  ```

  对应的服务封装入口（来自 Phase 4 的 `vulcan.services`）：

  ```python
  from vulcan.services.backend_server import run_backend
  from vulcan.services.apis import DataCollectionApp, DatasetOptimizationJobs
  ```

- 文档末尾引导读者前往 `docs/architecture.md` 与 `reconstruction_plan.md` 获取更深入的结构与重构说明。

---

### 3. 与 `reconstruction_plan.md` Phase 5 的一致性

对照重构总方案中 Phase 5 的描述，本阶段已完成：

- 在根目录下建立了 `tests/` 目录，并按包结构镜像划分为 `test_framework`、`test_lang`、`test_datacollection`、`test_services` 四个子目录；
- 在 `docs/` 中新增了：
  - `architecture.md`：作为架构与目录结构的高层说明；
  - `usage.md`：作为基于 `src/vulcan` 布局的使用指南；
- 保持现阶段改动“轻量但清晰”：只搭建结构和骨架，不预先假设完整的测试与文档内容。

这些改动与 `reconstruction_plan.md` 中“**tests 目录镜像包结构**”和“**docs 中集中项目结构与使用说明**”的要求对齐，同时不会影响现有的运行方式或引入新的强制依赖。后续如需扩展测试覆盖率或完善文档，只需在当前搭好的框架内增量补充即可。 

