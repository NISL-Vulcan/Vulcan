## Phase 1：确立包结构但不移动脚本

> 本阶段的目标和范围与 `reconstruction_plan.md` 中 **Phase 1：确立包结构但不移动脚本** 一节保持严格一致。

本阶段根据重构总方案（见 `reconstruction_plan.md`）以及 Real Python 项目布局最佳实践（参考 `https://realpython.com/ref/best-practices/project-layout/`），在不移动任何脚本目录（如 `tools/`、`scripts/`）的前提下，完成以下目标：

- 在仓库中引入统一的 `src/vulcan` 包结构，用于承载所有可复用的库代码；
- 将原有的核心库目录（`framework/`、`langParser/`、`dataCollection/tools/`）迁移到 `src/vulcan/` 下对应位置；
- 为后续阶段（拆分 CLI 与库逻辑、测试与文档完善等）奠定基础。

---

### 1. 新建统一包结构目录

在项目根目录下创建 `src/vulcan` 目录，并添加空的 `__init__.py` 文件，使其成为一个标准的 Python 包：

- 新增目录：
  - `src/`
  - `src/vulcan/`
- 新增文件：
  - `src/vulcan/__init__.py`

命令示例（已执行）：

```bash
mkdir -p src/vulcan
touch src/vulcan/__init__.py
```

---

### 2. 迁移核心库代码目录

#### 2.1 `framework/` → `src/vulcan/framework/`

将原先位于项目根目录下的 `framework/` 整体迁移为 `src/vulcan/framework/`，保持其内部结构不变：

- 原路径：
  - `framework/`
- 新路径：
  - `src/vulcan/framework/`

命令示例（已执行）：

```bash
mv framework src/vulcan/framework
```

迁移后内部仍包含（示意）：

- `src/vulcan/framework/__init__.py`
- `src/vulcan/framework/dataset.py`
- `src/vulcan/framework/datasets/`
- `src/vulcan/framework/errors/`
- `src/vulcan/framework/losses.py`
- `src/vulcan/framework/metrics.py`
- `src/vulcan/framework/model.py`
- `src/vulcan/framework/models/`
- `src/vulcan/framework/optimizers.py`
- `src/vulcan/framework/preprocess.py`
- `src/vulcan/framework/pretrained.py`
- `src/vulcan/framework/representations/`
- `src/vulcan/framework/schedulers.py`
- `src/vulcan/framework/utils/` 等。

> 注意：本阶段仅改变物理路径，尚未细分 `framework/utils/` 中的具体模块，保持与原项目行为一致，后续阶段再做更细致的重构。

#### 2.2 `langParser/` → `src/vulcan/lang/`

将原先的语言解析相关代码整体迁移至 `src/vulcan/lang/`：

- 原路径：
  - `langParser/`
- 新路径：
  - `src/vulcan/lang/`

命令示例（已执行）：

```bash
mv langParser src/vulcan/lang
```

迁移后内部结构示意：

- `src/vulcan/lang/CodeAnalysis/`
  - `__init__.py`
  - `analyzer.py`
  - `implementations/`
  - `interfaces/`
  - `utils/`
- `src/vulcan/lang/cParser/`
  - `grammar/`
  - `src/`
  - `test_source/`
- 以及原 `langParser` 下的其他文件（如 `example.py` 等）。

> 说明：本阶段保持原有子目录命名（如 `CodeAnalysis`、`cParser`）不变，仅调整到统一的 `vulcan.lang` 命名空间下。更进一步的命名规范化（如 `analysis/`、`cparser/`）和 API 整理将留待后续阶段处理。

#### 2.3 `dataCollection/tools/` → `src/vulcan/datacollection/tools/`

`dataCollection/` 目录中包含既有代码又可能有数据。本阶段先只迁移其中的“工具代码”子目录 `tools/` 到统一包结构下：

- 原路径：
  - `dataCollection/tools/`
- 新路径：
  - `src/vulcan/datacollection/tools/`

命令示例（已执行）：

```bash
mkdir -p src/vulcan/datacollection
mv dataCollection/tools src/vulcan/datacollection/
```

迁移后，`src/vulcan/datacollection/tools/` 下仍包含原有子目录：

- `src/vulcan/datacollection/tools/common/`
- `src/vulcan/datacollection/tools/nvd/`
- `src/vulcan/datacollection/tools/osv/`
- `src/vulcan/datacollection/tools/research/`

顶层的 `dataCollection/` 目录仍然保留，用于后续承载数据或中间产物；其中的代码部分已开始收拢到包内。

---

### 3. 暂未移动的脚本与其他目录

根据 Phase 1 的约束，本阶段 **没有移动或重命名任何脚本目录或脚本文件**，包括但不限于：

- `tools/` 目录下的脚本：
  - `tools/train.py`
  - `tools/val.py`
  - `tools/export.py`
  - `tools/benchmark.py`
  - `tools/clean_cuda.py`
- `scripts/` 目录下的各类运行/服务/调试脚本：
  - `scripts/backend_server.py`
  - `scripts/start_backend.py`
  - `scripts/start_services.py`
  - `scripts/data_collection_api.py`
  - `scripts/data_collector.py`
  - `scripts/dataset_optimization_api.py`
  - `scripts/dataset_optimization_server.py`
  - `scripts/debug_optimization.py`
  - `scripts/debug_optimization_simple.py`
  - `scripts/quick_test.py`
  - `scripts/demo.py`
  - `scripts/diagnose_issue.py`
  - `scripts/resource_updater.py`
  - `scripts/split_dataset.py`
  - `scripts/trans_pyg_data.py`
  - `scripts/Normalization/` 等。

这些脚本依然位于项目根目录结构的原始位置，其内部对 `framework` 等模块的导入关系将在后续阶段逐步统一为通过 `vulcan` 包访问。

---

### 4. 已知后续工作与影响说明

#### 4.1 导入路径的后续调整

由于 `framework/`、`langParser/`、`dataCollection/tools/` 已迁移至 `src/vulcan/`，原有代码中通过相对项目根目录导入它们的方式（如 `from framework...`）在不调整 Python 路径时将无法直接工作。

下一阶段的工作将包括：

- 在运行环境中将 `src/` 添加到 `PYTHONPATH` 或通过安装当前项目（`pip install -e .`）的方式，让 `vulcan` 成为标准可导入包；
- 逐步将脚本中的导入从：
  - `from framework.xxx import ...`
  - `from langParser.xxx import ...`
  - 等形式
  调整为：
  - `from vulcan.framework.xxx import ...`
  - `from vulcan.lang.xxx import ...`
  - `from vulcan.datacollection.tools.xxx import ...`
- 同时清理由于历史路径硬编码带来的问题（例如某些 `sys.path.append(...)` 对旧路径的依赖）。

#### 4.2 与 Real Python 最佳实践的对齐情况

到目前为止，项目已经完成了以下与 Real Python 项目布局最佳实践一致的调整（参考 `https://realpython.com/ref/best-practices/project-layout/`）：

- **将可导入代码集中到包中**：核心库代码现在统一收拢至 `src/vulcan/` 下；
- **为引入 `src/` 布局奠定基础**：顶层结构已具备 `src/` + 顶级包的雏形，后续可继续引入 `tests/`、更规范的文档与依赖管理；
- **为分离“库代码”与“脚本入口”铺路**：脚本目录暂未移动，但其依赖的核心实现已经统一迁入库包内，后续可以将 CLI 薄化为仅调用 `vulcan` 内函数。

---

### 5. 当前状态小结

- 新建了统一包目录：`src/vulcan/`（含 `__init__.py`）；
- 将原来的：
  - `framework/` → `src/vulcan/framework/`
  - `langParser/` → `src/vulcan/lang/`
  - `dataCollection/tools/` → `src/vulcan/datacollection/tools/`
  完成了物理迁移；
- 未移动任何脚本目录/文件；
- 尚未大规模修改导入路径和脚本逻辑，这部分将作为后续阶段（Phase 2 及以后）的主要工作内容。

