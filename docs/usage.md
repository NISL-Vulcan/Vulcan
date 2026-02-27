## 使用指南（基于 src/vulcan 布局）

### 1. 环境准备

在项目根目录执行：

```bash
cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan

# 推荐：按 pyproject.toml 安装（src/ 布局的标准用法）
python -m pip install -e .

# 可选：开发/测试依赖
python -m pip install -e ".[dev]"
```

### 2. 训练与验证

```bash
# 训练
python tools/train.py --cfg configs/custom.yaml

# 验证
python tools/val.py --cfg configs/custom.yaml

# benchmark
python tools/benchmark.py --cfg configs/custom.yaml
```

对应的 Python 入口位于：

```python
from vulcan.cli.train import cli_main as train_cli
from vulcan.cli.val import cli_main as val_cli
from vulcan.cli.benchmark import cli_main as benchmark_cli
```

### 3. 后端服务与 API

```bash
# 启动后端
python scripts/start_backend.py

# 启动综合服务
python scripts/start_services.py
```

相关服务封装位于：

```python
from vulcan.services.backend_server import run_backend
from vulcan.services.apis import DataCollectionApp, DatasetOptimizationJobs
```

更多结构性说明可参考 `docs/architecture.md` 与 `reconstruction_plan.md`。 

