## Phase 4：整理服务层与运维脚本

本阶段对应 `reconstruction_plan.md` 中 **Phase 4：整理服务层与运维脚本** 的设计目标：

- 为主要后端服务与 API 建立统一的 `vulcan.services` 封装层；
- 让 `scripts/` 下的启动/运维脚本只负责进程编排和环境检查，真正的服务接口通过 `vulcan.services` 暴露；
- 在不大规模重写现有大体量脚本的前提下，为未来进一步迁移实现逻辑预留清晰结构。

本阶段采取的是“轻量封装”策略：将现有服务逻辑从调用路径上集中到 `src/vulcan/services/`，但暂不强行搬运全部实现代码，避免大文件重构风险。

---

### 1. 新增 `src/vulcan/services/` 包

在 `src/vulcan/` 下新建服务层包：

- 新增目录与文件：
  - `src/vulcan/services/__init__.py`
  - `src/vulcan/services/backend_server.py`
  - `src/vulcan/services/apis.py`

`src/vulcan/services/__init__.py`：

```python
"""
Vulcan 服务层封装。

本包对后端服务（例如 backend_server）和与训练/数据处理相关的 API
提供统一的 Python 接口，脚本层（scripts/*.py）只保留启动/编排职责。
"""
```

> 说明：后续如有更多服务（例如 dataset_optimization_server 的专门封装），也可以在该包下继续扩展子模块。当前阶段主要先引入统一入口与 minimal 封装。

---

### 2. 后端服务封装：`vulcan.services.backend_server`

新增文件：`src/vulcan/services/backend_server.py`，封装对 `scripts/backend_server.py` 的调用：

```python
from scripts import backend_server as _backend_server


def run_backend(*args: Any, **kwargs: Any) -> Any:
    """
    启动后端服务。

    优先调用 `scripts/backend_server.py` 中的 `main` 或 `start_backend` 函数，
    以保持与现有脚本行为一致。
    """
    if hasattr(_backend_server, "main"):
        return _backend_server.main(*args, **kwargs)
    if hasattr(_backend_server, "start_backend"):
        return _backend_server.start_backend(*args, **kwargs)
    raise RuntimeError("backend_server script has no 'main' or 'start_backend' entrypoint")
```

特点：

- **不直接搬运大体量实现代码**，而是通过导入 `scripts.backend_server` 并调用其公开入口函数来启动服务；
- 为其他模块（包括启动脚本和未来可能的 CLI）提供统一的 Python API：`vulcan.services.backend_server.run_backend()`。

---

### 3. API 统一封装：`vulcan.services.apis`

新增文件：`src/vulcan/services/apis.py`，用于集中暴露当前已有的 API 脚本：

- 源脚本：
  - `scripts/data_collection_api.py`
  - `scripts/dataset_optimization_api.py`

- 封装形式：

```python
from scripts import data_collection_api  # noqa: F401
from scripts import dataset_optimization_api  # noqa: F401

# 为了简化调用，这里可按需提供别名：

DataCollectionApp = data_collection_api.app
DatasetOptimizationJobs = dataset_optimization_api.optimization_jobs
```

这样：

- 其他代码可以通过统一路径访问 API 对象，例如：
  - `from vulcan.services.apis import DataCollectionApp`
  - `from vulcan.services.apis import DatasetOptimizationJobs`
- API 本体暂时仍实现于 `scripts/` 下，但调用入口已经统一到了 `vulcan.services.apis`。

> 后续若希望将 Flask app / 优化任务管理逻辑完全迁入 `src/vulcan/services/` 内，只需将实现从 `scripts/` 移动过来并调整导入，`vulcan.services.apis` 的对外接口可以保持不变。

---

### 4. 启动脚本对服务层的依赖调整

根据 `reconstruction_plan.md` 中 Phase 4 的建议，“`scripts` 只保留 orchestration/部署脚本”，本阶段对两个核心启动脚本进行了轻量改造：

#### 4.1 `scripts/start_backend.py`

原逻辑：

- 自行检查依赖与环境后，直接通过 `subprocess.run([sys.executable, "backend_server.py"], check=True)` 启动后端；

调整后：

- 在文件顶部新增导入：

```python
from vulcan.services.backend_server import run_backend
```

- 将 `start_backend()` 函数中“启动服务器”的部分从直接 `subprocess.run("backend_server.py")` 改为调用服务层封装：

```python
def start_backend():
    """启动后端服务器（通过 vulcan.services.backend_server 封装）。"""
    print("🚀 启动vulcan-Detection后端服务器...")
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 检查环境
    if not check_vulcan_environment():
        return False
    
    # 创建必要的目录
    Path("generated_configs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # 启动服务器（实际逻辑在 vulcan.services.backend_server 中）
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True
```

这样：

- `start_backend.py` 仍保留其“环境检查 + 提示 + 目录创建”等职责；
- 实际如何启动后端服务器的细节，集中到 `vulcan.services.backend_server.run_backend()` 中。

#### 4.2 `scripts/start_services.py`

原逻辑：

- `start_backend_server()` 直接通过 `subprocess.run([sys.executable, "backend_server.py"], check=True)` 启动主后端服务；

调整后：

- 顶部新增：

```python
from vulcan.services.backend_server import run_backend
```

- 将 `start_backend_server()` 重写为：

```python
def start_backend_server():
    """启动主后端服务（通过 vulcan.services.backend_server 封装）。"""
    print("🚀 启动主后端服务 (端口 5000)...")
    print("📋 包含功能:")
    print("  • 配置生成")
    print("  • 模型训练")
    print("  • 模型验证")
    print("  • 数据集优化")
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\n🛑 主后端服务已停止")
    except Exception as e:
        print(f"❌ 主后端服务启动失败: {e}")
```

脚本 `main()` 的外层结构与输出提示保持不变，只是实际启动动作由服务层统一处理。

---

### 5. 当前运行方式与 Phase 4 的影响

Phase 4 之后，原有的使用方式依然可用：

```bash
cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan
export PYTHONPATH=src:$PYTHONPATH

# 启动后端（环境检查 + 服务启动）
python scripts/start_backend.py

# 启动综合服务（目前主要仍是后端服务）
python scripts/start_services.py
```

但内部调用路径已经演进为：

- `scripts/start_backend.py` → `vulcan.services.backend_server.run_backend()` → `scripts.backend_server` 的实现；
- `scripts/start_services.py` → `vulcan.services.backend_server.run_backend()`；
- 对于 API 层逻辑，可通过：
  - `from vulcan.services.apis import DataCollectionApp, DatasetOptimizationJobs` 等方式集中访问。

> 注意：由于 `vulcan.services.backend_server` 目前是对现有 `scripts/backend_server.py` 的封装（而非完全迁移），如果将来重构 `backend_server.py` 的内部接口（例如重命名 `main`/`start_backend`），应同步更新 `run_backend` 的逻辑以维持兼容性。

---

### 6. Phase 4 小结

- 已完成：
  - 引入 `src/vulcan/services/` 包，并提供 `backend_server` 与 `apis` 两个子模块；
  - 将 `scripts/start_backend.py`、`scripts/start_services.py` 调整为通过 `vulcan.services.backend_server.run_backend()` 启动后端服务；
  - 集中暴露数据收集与数据集优化 API 相关对象到 `vulcan.services.apis`，为后续统一调用与进一步迁移实现打基础。
- 有意暂缓的重构（后续可按需进行）：
  - 将 `scripts/backend_server.py` 和 `dataset_optimization_server.py` 的全部服务实现物理迁移到 `src/vulcan/services/` 中，只在 `scripts/` 中保留极薄启动脚本；
  - 将更多运维脚本（如 `data_collector.py`、`resource_updater.py` 等）按职责抽象为服务层函数，并在 `scripts/` 中通过 `vulcan.services` 调用。

整体上，本阶段已经让“如何启动后端服务 / 使用 API”的入口统一到 `vulcan.services` 命名空间下，与 `reconstruction_plan.md` 中关于“服务层与运维脚本分离”的方向保持一致，并且没有破坏现有脚本级工作流。 

