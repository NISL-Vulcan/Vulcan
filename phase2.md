## Phase 2：统一导入路径与运行方式

本阶段严格对应 `reconstruction_plan.md` 中 **Phase 2：统一导入路径与运行方式** 的设计，目标是：

- 将所有库内部及脚本对 `framework`、`langParser`、`dataCollection.tools` 的导入统一为 `vulcan.*` 包路径；
- 为后续运行提供清晰、统一的方式（基于 `src/` 布局的 `PYTHONPATH` 或可编辑安装）；
- 对历史硬编码路径进行初步清理或标记。

下文首先说明本阶段的代码修改与约定的运行方式，然后列出具体变更点。

---

### 1. 运行方式约定（基于 src 布局）

根据 `reconstruction_plan.md` 的 Phase 2 设计，本阶段给出两种推荐运行方式：

- **方式 A：直接使用 `PYTHONPATH=src` 运行仓库内脚本**

  在项目根目录 `/home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan` 下：

  ```bash
  # 一次性在当前命令行中设置 PYTHONPATH 并运行
  cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan
  export PYTHONPATH=src:$PYTHONPATH

  # 训练
  python tools/train.py --cfg configs/custom.yaml

  # 验证
  python tools/val.py --cfg configs/custom.yaml

  # benchmark
  python tools/benchmark.py --cfg configs/custom.yaml
  ```

  说明：
  - `vulcan` 包从 `src/` 下导入，因此需要将 `src` 加入 `PYTHONPATH`；
  - 脚本内部已经统一使用 `from vulcan.framework...` 等导入方式。

- **方式 B：后续引入 `pyproject.toml` 后，以可编辑安装方式运行（规划中）**

  该方式在 `reconstruction_plan.md` 中已规划，但目前尚未创建 `pyproject.toml`，因此仅作为后续演进方向，不在本阶段实际执行：

  ```bash
  pip install -e .
  # 然后可通过 python -m vulcan.cli.train 等方式运行
  ```

---

### 2. 统一导入路径的具体修改

#### 2.1 顶层 CLI 脚本（`tools/`）

- `tools/train.py`

  将原有导入：

  ```python
  from framework.models import *
  from framework.datasets import * 
  from framework.model import get_model
  from framework.dataset import get_dataset, get_dataloader
  from framework.losses import get_loss
  from framework.schedulers import get_scheduler
  from framework.optimizers import get_optimizer
  from framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
  ```

  统一替换为通过 `vulcan` 包导入：

  ```python
  from vulcan.framework.models import *
  from vulcan.framework.datasets import * 
  from vulcan.framework.model import get_model
  from vulcan.framework.dataset import get_dataset, get_dataloader
  from vulcan.framework.losses import get_loss
  from vulcan.framework.schedulers import get_scheduler
  from vulcan.framework.optimizers import get_optimizer
  from vulcan.framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
  ```

- `tools/benchmark.py`

  同样将：

  ```python
  from framework.models import *
  from framework.datasets import * 
  from framework.model import get_model
  from framework.dataset import get_dataset, get_dataloader
  from framework.losses import get_loss
  from framework.schedulers import get_scheduler
  from framework.optimizers import get_optimizer
  from framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
  ```

  替换为：

  ```python
  from vulcan.framework.models import *
  from vulcan.framework.datasets import * 
  from vulcan.framework.model import get_model
  from vulcan.framework.dataset import get_dataset, get_dataloader
  from vulcan.framework.losses import get_loss
  from vulcan.framework.schedulers import get_scheduler
  from vulcan.framework.optimizers import get_optimizer
  from vulcan.framework.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
  ```

- `tools/val.py`

  将：

  ```python
  from framework.models import *
  from framework.datasets import *
  from framework.metrics import Metrics
  from framework.utils.utils import setup_cudnn
  from framework.dataset import get_dataset
  from framework.model import get_model
  ```

  替换为：

  ```python
  from vulcan.framework.models import *
  from vulcan.framework.datasets import *
  from vulcan.framework.metrics import Metrics
  from vulcan.framework.utils.utils import setup_cudnn
  from vulcan.framework.dataset import get_dataset
  from vulcan.framework.model import get_model
  ```

- `tools/export.py`

  将：

  ```python
  from framework.models import *
  from framework.datasets import *
  from framework.model import get_model
  ```

  替换为：

  ```python
  from vulcan.framework.models import *
  from vulcan.framework.datasets import *
  from vulcan.framework.model import get_model
  ```

> 说明：Phase 2 仅统一导入路径，尚未按 Phase 3 方案把 CLI 逻辑迁入 `src/vulcan/cli/`，脚本仍然作为直接入口存在于 `tools/` 目录。

---

#### 2.2 后端服务脚本（`scripts/backend_server.py`）

- 在 `scripts/backend_server.py` 中，将：

  ```python
  from framework.config_templates import ConfigTemplateManager
  ```

  替换为：

  ```python
  from vulcan.framework.config_templates import ConfigTemplateManager
  ```

当前文件中仍保留了一行历史路径补丁（用于向 `sys.path` 添加 `framework` 目录）：

```python
sys.path.append(os.path.join(os.path.dirname(__file__), 'framework'))
```

这行代码在 Phase 2 中暂未删除，目的是维持与旧环境的兼容；随着后续将所有脚本迁移为仅依赖 `vulcan` 包（并通过 `PYTHONPATH=src` 或可编辑安装提供导入路径），这类 `sys.path` 手工操作可在后续阶段安全移除。

---

#### 2.3 框架内部模块（`src/vulcan/framework/**`）

根据 `reconstruction_plan.md` 的要求，库内部也统一改为通过 `vulcan.framework...` 导入：

- `src/vulcan/framework/model.py`

  ```python
  from framework.models import *
  ```

  改为：

  ```python
  from vulcan.framework.models import *
  ```

- `src/vulcan/framework/errors/__init__.py`

  ```python
  from framework.errors.dataset_errors import DatasetInitError,DatasetNotMatch,BenchmarkInitError
  from framework.errors.download_errors import DownloadFailed,TooManyRequests
  ```

  改为：

  ```python
  from vulcan.framework.errors.dataset_errors import DatasetInitError,DatasetNotMatch,BenchmarkInitError
  from vulcan.framework.errors.download_errors import DownloadFailed,TooManyRequests
  ```

- `src/vulcan/framework/utils/utils.py`

  ```python
  from framework import models
  ```

  改为：

  ```python
  from vulcan.framework import models
  ```

- `src/vulcan/framework/datasets/IVDetect/IVDetectDataset_build.py` 中注释块里的导入：

  ```python
  import framework.datasets.IVDetect.utils.process as process
  ```

  改为：

  ```python
  from vulcan.framework.datasets.IVDetect.utils import process
  ```

  同一文件中仍存在若干指向旧工程 `astBugDetection` 的硬编码路径（如 `/root/astBugDetection/framework/datasets/IVDetect/data.csv`），目前仅标记在代码中，尚未迁移到配置，后续可配合 Phase 4/5 一并清理。

- `src/vulcan/framework/representations/**` 相关文件：

  - `common_test.py`：

    ```python
    import framework.representations.common as common
    ```

    改为：

    ```python
    from vulcan.framework.representations import common
    ```

  - `ast_graphs.py` / `syntax_seq.py` / `llvm_seq.py` / `llvm_graphs.py` 及其 `*_test.py` 文件中，所有：

    ```python
    from framework.representations...
    ```

    统一替换为：

    ```python
    from vulcan.framework.representations...
    ```

  - `vectorizers/word2vec.py`：

    ```python
    from framework.representations.vectorizers.vectorizer import Vectorizer
    ```

    改为：

    ```python
    from vulcan.framework.representations.vectorizers.vectorizer import Vectorizer
    ```

---

#### 2.4 与 `langParser` 相关的历史硬编码路径

在 `src/vulcan/lang/**` 中存在两处旧工程绝对路径，原引用形式为：

```python
sys.path.append('/Users/asteriska/.../astBugDetection/langParser/ControlFlowGraph')
default_path = "/Users/asteriska/.../astBugDetection/langParser/ControlFlowGraph/test_source/1.cpp"
```

分别位于：

- `src/vulcan/lang/cParser/src/cfg_from_stdin.py`
- `src/vulcan/lang/CodeAnalysis/utils/c_utils/src/cfg_from_stdin.py`

本阶段将上述硬编码路径改为**相对当前文件**的路径计算，以配合 `vulcan.lang` 包结构：

```python
here = os.path.dirname(os.path.abspath(__file__))
control_flow_graph_root = os.path.abspath(os.path.join(here, "..", "..", "ControlFlowGraph"))
sys.path.append(control_flow_graph_root)

...

default_path = os.path.join(control_flow_graph_root, "test_source", "1.cpp")
```

这样，无需依赖用户本地旧工程目录，只要 `ControlFlowGraph/test_source/1.cpp` 在当前包结构中存在，就可以正常工作。

---

### 3. 导入自检结果

为验证 Phase 2 导入统一后的基本可用性，在项目根目录执行了：

```bash
cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan
PYTHONPATH=src python - << 'PY'
import importlib
for name in [
    'vulcan',
    'vulcan.framework',
    'vulcan.framework.models',
    'vulcan.framework.datasets',
    'vulcan.lang',
    'vulcan.datacollection.tools']:
    try:
        importlib.import_module(name)
        print(f'OK {name}')
    except Exception as e:
        print(f'FAIL {name}: {e}')
PY
```

当前结果：

- `OK vulcan`
- `OK vulcan.lang`
- `OK vulcan.datacollection.tools`
- `FAIL vulcan.framework` / `vulcan.framework.models` / `vulcan.framework.datasets`：`No module named 'tabulate'`

说明：

- 导入失败的原因是 **运行环境中暂未安装第三方依赖包**（如 `tabulate`），并非导入路径错误；
- 一旦按 `requirements_backend.txt` 安装依赖（如 `pip install -r requirements_backend.txt`），在 `PYTHONPATH=src` 前提下，`vulcan.framework` 及其子包应可正常导入。

---

### 4. Phase 2 小结

- 已完成：
  - 将 `tools/*.py`、`scripts/backend_server.py` 以及 `src/vulcan/framework/**` 中所有使用 `framework` 顶层包名的导入统一为 `vulcan.framework...`；
  - 将 `framework` 相关的 `import` 形式导入改为 `from vulcan.framework...` 风格；
  - 将 `langParser` 相关的两处绝对路径改为基于当前文件目录的相对路径计算；
  - 在 `phase2.md` 中记录了运行方式与上述代码变更。
- 尚未执行（留待 Phase 3+）：
  - 将 CLI 逻辑迁入 `src/vulcan/cli/` 并通过 `[project.scripts]` 提供命令；
  - 完全移除脚本中对 `sys.path` 的手工补丁；
  - 引入 `pyproject.toml` 并改用可编辑安装方式运行。 


