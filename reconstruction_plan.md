## Vulcan 项目目录与程序结构重构总方案

本文档基于 Real Python 的项目布局最佳实践（特别是 `src/` 布局与顶层结构建议，参考 `https://realpython.com/ref/best-practices/project-layout/`），为 Vulcan 项目设计一个目标结构，并给出从当前结构到目标结构的迁移思路与分阶段实施计划。

---

### 1. 目标总体目录结构（建议形态）

以 **`src/` 布局 + 单一顶级包 `vulcan`** 为核心：

```text
Vulcan/
├── pyproject.toml                 # （中长期目标，可与现有 requirements_backend.txt 并存一段时间）
├── requirements_backend.txt
├── README.md
├── LICENSE
├── vulcan.yaml                    # 全局运行配置，可部分迁移到 pyproject.toml
├── src/
│   └── vulcan/
│       ├── __init__.py
│       ├── core/                  # 核心通用组件（错误、配置、通用工具等）
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── errors.py
│       │   └── utils.py
│       ├── framework/             # 训练框架（模型/数据集/损失/优化器等）
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── datasets/
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   ├── model.py
│       │   ├── models/
│       │   ├── optimizers.py
│       │   ├── preprocess.py
│       │   ├── pretrained.py
│       │   ├── representations/
│       │   ├── schedulers.py
│       │   └── training/
│       ├── lang/                  # 代码/语言解析相关
│       │   ├── __init__.py
│       │   ├── analysis/          # 原 CodeAnalysis
│       │   ├── cparser/           # 原 cParser
│       │   └── examples/
│       ├── datacollection/        # 漏洞/数据采集相关逻辑
│       │   ├── __init__.py
│       │   ├── common/
│       │   ├── nvd/
│       │   ├── osv/
│       │   └── research/
│       ├── services/              # 运行中的服务端/后端相关封装
│       │   ├── __init__.py
│       │   ├── backend_server.py
│       │   ├── dataset_optimization_server.py
│       │   └── apis.py
│       └── cli/                   # 统一的命令行入口（train/val/benchmark 等）
│           ├── __init__.py
│           ├── train.py
│           ├── val.py
│           ├── export.py
│           └── benchmark.py
├── tools/                         # 纯 CLI 脚本入口（薄封装，调用 src/vulcan）
│   ├── train.py
│   ├── val.py
│   ├── export.py
│   ├── benchmark.py
│   └── clean_cuda.py
├── scripts/                       # 运行/部署/维护脚本（可含 Python + Shell）
│   ├── start_backend.py
│   ├── start_services.py
│   ├── backend_server.py
│   ├── data_collection_api.py
│   ├── data_collector.py
│   ├── dataset_optimization_api.py
│   ├── dataset_optimization_server.py
│   ├── resource_updater.py
│   ├── split_dataset.py
│   ├── trans_pyg_data.py
│   ├── check_backend.py
│   ├── auto_update_and_dynamic_ratio.py
│   ├── diagnose_issue.py
│   ├── debug_optimization.py
│   ├── debug_optimization_simple.py
│   ├── quick_test.py
│   ├── demo.py
│   └── Normalization/
├── docs/                          # 正式用户/开发文档
│   ├── readme.md
│   └── architecture.md
├── notebooks/                     # 教程与实验 Notebook
│   └── tutorials.ipynb
├── tests/                         # 单元/集成测试（按包结构镜像）
│   ├── test_framework/
│   ├── test_lang/
│   ├── test_datacollection/
│   └── test_services/
└── dataCollection/                # 若仍需使用，可保留为数据根目录（raw/processed 等）
    └── ...
```

> 目标：  
> - **清晰顶层结构**：`src/`、`tests/`、`docs/`、`tools/`、`scripts/` 一眼可见。  
> - **所有可导入代码进包**：统一集中在 `src/vulcan/` 下。  
> - **CLI 与业务逻辑分离**：`tools/`、`scripts/` 只做入口和 orchestrator。  
> - **测试目录独立**：`tests/` 与主包结构大致镜像，便于运行和排除打包。

---

### 2. 现有顶层目录与文件的迁移映射（设计）

#### 2.1 根目录文件

- `README.md`
  - **保留**为根目录 README，负责项目简介、安装与快速上手；
  - 详细文档迁移/扩展到 `docs/` 下。

- `LICENSE`
  - **保留**在根目录不动。

- `requirements_backend.txt`
  - **短期**：继续在根目录使用；
  - **长期**：核心依赖可迁到 `pyproject.toml` 的 `[project]` / `[project.optional-dependencies]` 中。

- `vulcan.yaml`
  - **保留**为运行配置文件；
  - 部分与构建/工具相关的配置可迁入 `pyproject.toml` 中对应工具的配置。

- `install.sh`
  - **保留**在根目录，或视情况迁入 `scripts/install.sh`。

---

#### 2.2 `framework/` → `src/vulcan/framework/`

原内容（已在 Phase 1 迁移）：

- `framework/__init__.py`
- `framework/config_templates.py`
- `framework/dataset.py`
- `framework/datasets/`
- `framework/errors/`
- `framework/losses.py`
- `framework/metrics.py`
- `framework/model.py`
- `framework/models/`
- `framework/optimizers.py`
- `framework/preprocess.py`
- `framework/pretrained.py`
- `framework/representations/`
- `framework/schedulers.py`
- `framework/utils/`

**迁移目标：**

- 顶层整体：
  - `framework/` → `src/vulcan/framework/`

- 细分建议：
  - `framework/__init__.py` → `src/vulcan/framework/__init__.py`
  - `framework/config_templates.py`
    - 若为框架级公共配置模板 → `src/vulcan/framework/config_templates.py` 或 `src/vulcan/core/config.py`
  - `framework/dataset.py` → `src/vulcan/framework/dataset.py`
  - `framework/datasets/` → `src/vulcan/framework/datasets/`
  - `framework/errors/`
    - 若错误类型只服务于框架内部 → `src/vulcan/framework/errors/`
    - 若为全局错误类型 → 抽到 `src/vulcan/core/errors/`
  - `framework/losses.py` → `src/vulcan/framework/losses.py`
  - `framework/metrics.py` → `src/vulcan/framework/metrics.py`
  - `framework/model.py` → `src/vulcan/framework/model.py`
  - `framework/models/` → `src/vulcan/framework/models/`
  - `framework/optimizers.py` → `src/vulcan/framework/optimizers.py`
  - `framework/preprocess.py` → `src/vulcan/framework/preprocess.py`
  - `framework/pretrained.py` → `src/vulcan/framework/pretrained.py`
  - `framework/representations/` → `src/vulcan/framework/representations/`
  - `framework/schedulers.py` → `src/vulcan/framework/schedulers.py`

##### `framework/utils/` 的拆分

`framework/utils/` 容易成为“工具大杂烩”，建议按领域拆分：

- `framework/utils/training.py`
  - → `src/vulcan/framework/training/__init__.py` 或 `src/vulcan/framework/training/loops.py`

- `framework/utils/vocabulary.py`
  - → `src/vulcan/framework/vocabulary.py` 或 `src/vulcan/data/vocabulary.py`

- `framework/utils/converting.py`
  - → `src/vulcan/data/converters.py` 或保留为 `src/vulcan/framework/converting.py`（视其用途而定）

- `framework/utils/clean_gadget.py`
  - 若与漏洞/代码解析关系更大 → 可移动到 `src/vulcan/datacollection/clean_gadget.py` 或 `src/vulcan/lang/utils/clean_gadget.py`

- `framework/utils/utils.py`
  - 对其中函数进行归类：
    - 有明确领域归属的，移入对应模块；
    - 真正通用的小工具，可进入 `src/vulcan/core/utils.py`；
  - 目标是将此文件瘦身到极小，避免成为杂物堆。

---

#### 2.3 `langParser/` → `src/vulcan/lang/`

原内容（已在 Phase 1 迁移）：

- `langParser/CodeAnalysis/`
  - `__init__.py`
  - `analyzer.py`
  - `implementations/`
  - `interfaces/`
  - `utils/`
- `langParser/cParser/`
  - `grammar/`
  - `src/`
  - `test_source/`
- `langParser/example.py` 等。

**迁移目标：**

- 顶层：
  - `langParser/` → `src/vulcan/lang/`

- 建议的进一步整理：
  - `langParser/CodeAnalysis/` → `src/vulcan/lang/analysis/`
  - `langParser/cParser/` → `src/vulcan/lang/cparser/`
    - 对 `test_source/` 中的示例，可迁到 `tests/test_lang/cparser/test_source/` 或 `src/vulcan/lang/cparser/examples/`
  - `langParser/example.py` → `src/vulcan/lang/examples/example.py` 或顶层 `examples/` 目录。

> 当前实现中，仅做了整体移动到 `src/vulcan/lang/`，命名规范化可以在后续阶段逐步完成。

---

#### 2.4 `dataCollection/tools/` → `src/vulcan/datacollection/tools/`

原内容（工具代码）：

- `dataCollection/tools/common/`
- `dataCollection/tools/nvd/`
- `dataCollection/tools/osv/`
- `dataCollection/tools/research/`

**迁移目标：**

- 顶层：
  - `dataCollection/tools/` → `src/vulcan/datacollection/tools/`

- 顶层 `dataCollection/`：
  - 保留，用于实际数据（原始数据/中间产物/缓存等），逐步 **不再放库代码**。

---

#### 2.5 `tools/`：CLI 入口与库逻辑分离

当前 `tools/` 下的脚本（如 `train.py`、`val.py`、`export.py`、`benchmark.py`、`clean_cuda.py`）通常既做参数解析，又包含一部分业务逻辑。

**目标结构：**

- 在 `src/vulcan/cli/` 引入对应模块：
  - `src/vulcan/cli/train.py`（定义 `main()`，调用 `vulcan.framework.training.*`）
  - `src/vulcan/cli/val.py`
  - `src/vulcan/cli/export.py`
  - `src/vulcan/cli/benchmark.py`
  - `src/vulcan/cli/clean_cuda.py`（如需要）

- `tools/` 中保留“极薄”的启动脚本：

  ```python
  # tools/train.py
  from vulcan.cli.train import main

  if __name__ == "__main__":
      main()
  ```

这与 Real Python 中“bin/scripts 目录只做入口，业务逻辑在包内”的建议一致（参见其对 `bin/`、`scripts/` 的说明）。

---

#### 2.6 `scripts/`：服务层与运维脚本

当前 `scripts/` 中包括：

- 服务类脚本：
  - `backend_server.py`
  - `dataset_optimization_server.py`
  - 以及相关 `*_api.py`
- 运维/管理脚本：
  - `start_backend.py`
  - `start_services.py`
  - `check_backend.py`
  - `resource_updater.py`
  - `split_dataset.py`
  - `trans_pyg_data.py`
- 调试 / demo：
  - `debug_optimization.py`
  - `debug_optimization_simple.py`
  - `quick_test.py`
  - `demo.py`
  - `diagnose_issue.py`
  - `Normalization/` 等。

**迁移与重构方向：**

- 将长期运行的服务逻辑集中到 `src/vulcan/services/`：
  - `scripts/backend_server.py` → `src/vulcan/services/backend_server.py`
  - `scripts/dataset_optimization_server.py` → `src/vulcan/services/dataset_optimization_server.py`
  - 相关 API：
    - `scripts/data_collection_api.py`、`scripts/dataset_optimization_api.py` → `src/vulcan/services/apis.py`（或按子模块拆分）。

- `scripts/` 中保留的脚本仅负责：
  - 解析命令行参数；
  - 调用 `vulcan.services` 中的函数，例如：
    - `start_backend.py` → 调用 `vulcan.services.backend_server.run()`
    - `start_services.py` → 批量启动多个服务。

- 调试/实验脚本：
  - 短期内保留在 `scripts/`;
  - 重要、稳定的 demo 将来可迁入 `examples/` 目录。

---

#### 2.7 文档与 Notebook：`docs/` 与 `notebooks/`

- `docs/readme.md`
  - 推荐重命名/拆分为：
    - `docs/overview.md`：总体说明；
    - `docs/user_guide.md`：使用说明；
    - `docs/architecture.md`：架构与重构方案（可包含本文件内容）。

- `notebooks/tutorials.ipynb`
  - 保留在 `notebooks/` 下，用于实验与交互式教程；
  - 其中成熟且稳定的内容，可抽取一部分转成 `docs/tutorials/*.md`。

---

### 3. 分阶段实施计划

#### Phase 1：确立包结构但不移动脚本（已部分完成）

目标：

- 新建统一包结构 `src/vulcan/`；
- 将纯库代码从顶层目录迁至 `src/vulcan/`，但不移动 `tools/`、`scripts/` 等脚本。

主要操作：

- 创建目录和包：
  - `mkdir -p src/vulcan`
  - `touch src/vulcan/__init__.py`

- 迁移核心库代码：
  - `mv framework src/vulcan/framework`
  - `mv langParser src/vulcan/lang`
  - `mkdir -p src/vulcan/datacollection`
  - `mv dataCollection/tools src/vulcan/datacollection/`

- 记录：在根目录添加 `phase1.md`，说明本阶段的物理迁移及影响（已完成）。

当前状态：

- 包结构已具备；
- 内部 import 仍大量使用 `from framework...` 等旧形式，需在 Phase 2 中统一为 `from vulcan.framework...`。

---

#### Phase 2：统一导入路径与运行方式

目标：

- 将所有库内部及脚本对 `framework`、`langParser`、`dataCollection.tools` 的导入统一为 `vulcan` 包路径；
- 定义推荐的运行方式（例如 `python -m vulcan.cli.train` 或 `pip install -e .` 后直接使用暴露的命令）。

主要工作：

1. **环境配置**
   - 临时方式：在开发环境中将仓库根目录下的 `src/` 加入 `PYTHONPATH`；
   - 推荐方式：引入 `pyproject.toml`，以可编辑安装模式运行：

     ```bash
     pip install -e .
     ```

2. **统一导入为 `vulcan.*`**
   - 所有 `src/vulcan/**` 内部的导入：
     - `from framework.xxx import ...` → `from vulcan.framework.xxx import ...`
     - `import framework.representations.common as common` → `from vulcan.framework.representations import common as common`
   - 脚本中的导入（`tools/`、`scripts/`）：
     - `from framework.models import ...` → `from vulcan.framework.models import ...`
     - `from framework.dataset import get_dataset` → `from vulcan.framework.dataset import get_dataset`
     - `from framework.config_templates import ConfigTemplateManager` → `from vulcan.framework.config_templates import ConfigTemplateManager` 等。

3. **清理历史硬编码路径**
   - 如 `src/vulcan/lang/cParser/src/cfg_from_stdin.py` 等文件中对旧工程的绝对路径引用：
     - `/.../astBugDetection/langParser/ControlFlowGraph/...`
   - 替换为相对当前包的导入或基于配置/环境变量的路径。

输出：

- 更新的代码 import；必要时补充一份 `running_guide.md`，说明如何设置环境与运行。

---

#### Phase 3：拆分 CLI 与库逻辑

目标：

- 将训练/验证/导出/benchmark 等逻辑抽象为 `vulcan` 包中的可复用函数；
- 使 `tools/` 仅保留非常薄的一层入口。

主要工作：

- 在 `src/vulcan/cli/` 中为每个主要脚本建立对应模块：
  - `cli/train.py` 暴露 `main(args)` 或 `run_train(config)`。
  - `cli/val.py`、`cli/export.py`、`cli/benchmark.py` 类似。

- 将 `tools/train.py` 等改为只负责：
  - 解析参数；
  - 调用 `vulcan.cli.train.main`。

- 若引入 `pyproject.toml`，可通过 `[project.scripts]` 暴露命令：

  ```toml
  [project.scripts]
  vulcan-train = "vulcan.cli.train:main"
  ```

输出：

- 更清晰的 CLI 层；
- 训练等关键逻辑可以被其他 Python 代码直接 `import` 使用与测试。

---

#### Phase 4：整理服务层与运维脚本

目标：

- 将后端服务及 API 封装到 `vulcan.services` 包中；
- 让 `scripts/` 只保留 orchestration/部署脚本。

主要工作：

- 在 `src/vulcan/services/` 中抽象：
  - `backend_server.py`、`dataset_optimization_server.py` 的核心类与函数；
  - `data_collection_api.py`、`dataset_optimization_api.py` 等统一到 `apis.py` 或子模块。

- `scripts/start_backend.py` / `start_services.py` 等仅调用对应服务：

  ```python
  from vulcan.services.backend_server import run_backend
  ```

- 对于调试脚本（`debug_optimization.py` 等）：
  - 抽出其中可复用逻辑到 `vulcan.services` 或 `vulcan.framework`；
  - 保留脚本用作开发调试入口，或迁入 `examples/`。

---

#### Phase 5：测试与文档结构完善

目标：

- 建立 `tests/` 目录，逐步补齐关键模块单元测试；
- 梳理文档结构，与新目录布局保持一致。

主要工作：

- 在根目录创建：
  - `tests/test_framework/`
  - `tests/test_lang/`
  - `tests/test_datacollection/`
  - `tests/test_services/`

- 将原 `framework/representations/*_test.py`、`lang` 相关测试/样例迁移为正式测试文件；

- 在 `docs/` 中新增：
  - `architecture.md`：总结目录结构、模块职责、本重构方案；
  - `usage.md`：CLI 命令、API 示例；
  - 若需要，补充 `dev_guide.md` 指导开发者如何扩展模型/数据集。

---

### 4. 与 Real Python 最佳实践的对齐总结

与 `https://realpython.com/ref/best-practices/project-layout/` 中推荐的项目结构对比，本方案已经或计划做到：

- **使用清晰的顶层结构**：配置、代码、测试、文档、脚本各有明确位置；
- **将可导入代码集中在包中**：`src/vulcan/` 作为唯一的顶级包；
- **选择 `src/` 布局**：避免因当前工作目录而导致的隐式导入问题；
- **将测试放在独立的 `tests/` 目录中**：测试结构大致镜像 `vulcan` 包结构；
- **在较大应用中分层**：区分 CLI（`cli`）、服务层（`services`）、领域逻辑（`framework`、`lang`、`datacollection`）、通用核心（`core`）。

这些调整将显著提升项目的可维护性、可测试性以及新成员上手时对结构的理解效率。接下来的具体实施可以严格按 Phase 1–5 渐进式推进，确保任何时刻项目都能保持可运行。 

