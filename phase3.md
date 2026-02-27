## Phase 3：拆分 CLI 与库逻辑

本阶段严格对应 `reconstruction_plan.md` 中 **Phase 3：拆分 CLI 与库逻辑** 的设计目标：

- 将训练 (`train`)、验证 (`val`)、benchmark (`benchmark`)、模型导出 (`export`) 等命令行入口逻辑从顶层 `tools/*.py` 中抽离到 `src/vulcan/cli/` 包内；
- 让 `tools/*.py` 只保留极薄的一层命令行封装，方便直接从仓库根目录运行；
/- 保持已有行为（参数、默认配置等）尽量不变，为后续通过 `pyproject.toml` 暴露 CLI 命令打基础。
/
---

### 1. 新增 `src/vulcan/cli/` 包

本阶段在 `src/vulcan/` 下新增 CLI 子包：

- 新增目录与文件：
  - `src/vulcan/cli/__init__.py`
  - `src/vulcan/cli/train.py`
  - `src/vulcan/cli/val.py`
  - `src/vulcan/cli/benchmark.py`
  - `src/vulcan/cli/export.py`

`src/vulcan/cli/__init__.py` 内容仅用于说明用途：

```python
"""
CLI entrypoints for Vulcan.

本包用于承载训练、验证、benchmark、导出等命令行入口逻辑，
顶层 `tools/*.py` 只保留极薄的封装，实际业务逻辑放在这里。
"""
```

---

### 2. 训练入口：`vulcan.cli.train`

#### 2.1 新增 `src/vulcan/cli/train.py`

将原 `tools/train.py` 中的**全部训练逻辑和参数解析**迁移到 `src/vulcan/cli/train.py` 中，并提供：

- `def main(cfg, gpu, save_dir: Path)`：核心训练逻辑，可被脚本或其他 Python 代码直接调用；
- `def cli_main()`：命令行入口，负责解析 `--cfg` 参数、加载配置、设置随机种子与 CUDNN/DDP 环境，然后调用 `main(...)`。

核心特征：

- 继续使用 Phase 2 中统一好的导入：`from vulcan.framework...`；
- 仍然通过 `ordered_load` 实现有序加载 YAML 配置；
- 保留原有训练循环、DDP 处理、AMP、scheduler、TensorBoard 记录与 `evaluate` 调用等逻辑。

> 说明：`train.py` 内部仍然通过 `from tools.val import evaluate` 复用现有的 `evaluate` 函数。后续若需要，可以改为从 `vulcan.cli.val` 中导出 `evaluate` 并在此处引用。

#### 2.2 精简 `tools/train.py`

原 `tools/train.py` 中的具体实现已移入 `vulcan.cli.train`，现在只保留：

```python
from vulcan.cli.train import cli_main


if __name__ == '__main__':
    cli_main()
```

这样：

- 直接运行命令：`python tools/train.py --cfg configs/custom.yaml` 的行为不变；
- 训练逻辑可被外部代码通过 `from vulcan.cli.train import main` 或 `cli_main` 复用。

---

### 3. 验证入口：`vulcan.cli.val`

#### 3.1 新增 `src/vulcan/cli/val.py`

将原 `tools/val.py` 中的验证逻辑迁移到 `src/vulcan/cli/val.py`：

- 保留并迁移：
  - `evaluate(model, dataloader, device)`：带 `@torch.no_grad()` 的验证主函数；
  - `main(cfg)`：执行完整验证流程（加载数据集、加载模型权重、执行验证并打印表格结果）；
- 新增 `cli_main()` 作为命令行入口：

```python
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
```

#### 3.2 精简 `tools/val.py`

`tools/val.py` 现在只包含：

```python
from vulcan.cli.val import cli_main


if __name__ == '__main__':
    cli_main()
```

命令保持一致：

```bash
PYTHONPATH=src python tools/val.py --cfg configs/custom.yaml
```

---

### 4. Benchmark 入口：`vulcan.cli.benchmark`

#### 4.1 新增 `src/vulcan/cli/benchmark.py`

将原 `tools/benchmark.py` 中的 benchmark 逻辑迁移到 `src/vulcan/cli/benchmark.py`：

- 保留：
  - `get_max_cuda_memory()`：统计峰值 CUDA 显存使用情况；
  - `ordered_load(...)`：有序加载配置；
  - `main(cfg, gpu, save_dir)`：核心 benchmark 循环（warmup + 定长 `total_steps`，计算 fps、打印显存占用等）；
  - 使用 `sklearnex.patch_sklearn()` 优化 sklearn；
- 新增 `cli_main()`：解析 `--cfg`，加载配置、设置随机种子和 DDP 环境，然后调用 `main(...)`。

#### 4.2 精简 `tools/benchmark.py`

`tools/benchmark.py` 现在只包含：

```python
from vulcan.cli.benchmark import cli_main


if __name__ == '__main__':
    cli_main()
```

---

### 5. 导出入口：`vulcan.cli.export`

#### 5.1 新增 `src/vulcan/cli/export.py`

将原 `tools/export.py` 中的导出逻辑迁移到 `src/vulcan/cli/export.py`，并做了轻微修正：

- 函数：
  - `export_onnx(model, inputs, file)`：用 ONNX 导出并简化模型；
  - `export_coreml(model, inputs, file)`：尝试用 `coremltools` 导出 CoreML 模型（如未安装则打印提示）；
  - `main(cfg)`：
    - 通过 `get_model(cfg['MODEL'])` 获取模型（修正了原文件中的 `get_mode` 拼写错误）；
    - 加载 `cfg['TEST']['MODEL_PATH']` 对应权重；
    - 构造 `inputs` 并调用 `export_onnx` 与 `export_coreml`。
- 新增 `cli_main()`：

```python
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    
    main(cfg)
```

#### 5.2 精简 `tools/export.py`

`tools/export.py` 现在只包含：

```python
from vulcan.cli.export import cli_main


if __name__ == '__main__':
    cli_main()
```

---

### 6. 运行方式与兼容性

Phase 3 完成后，推荐的运行方式与 Phase 2 一致，只是 CLI 实现位置改为 `vulcan.cli.*`：

- 在项目根目录：

```bash
cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan
export PYTHONPATH=src:$PYTHONPATH

# 训练
python tools/train.py --cfg configs/custom.yaml

# 验证
python tools/val.py --cfg configs/custom.yaml

# benchmark
python tools/benchmark.py --cfg configs/custom.yaml

# 模型导出
python tools/export.py --cfg configs/custom.yaml
```

同时，也可以直接从 Python 代码中复用 CLI 逻辑，例如：

```python
from vulcan.cli.train import main as train_main
from vulcan.cli.val import main as val_main
from vulcan.cli.export import main as export_main
```

> 注：目前在容器中直接导入 `vulcan.cli.*` 仍可能因为未安装依赖（如 `torch`、`tabulate` 等）而失败，属于**环境依赖问题**，与本阶段结构性修改无关。安装 `requirements_backend.txt` 中的依赖后即可正常导入。

---

### 7. Phase 3 小结

- 已完成：
  - 在 `src/vulcan/cli/` 下新增 `train.py`、`val.py`、`benchmark.py`、`export.py`，并拆分出 `main(...)` 与 `cli_main()`；
  - 将 `tools/train.py`、`tools/val.py`、`tools/benchmark.py`、`tools/export.py` 精简为仅调用对应的 `vulcan.cli.*.cli_main`；
  - 保持原有 CLI 参数（`--cfg`）与默认配置路径不变，用户现有使用方式不受影响。
- 尚未做但已在 `reconstruction_plan.md` 中规划的后续工作：
  - 在 `pyproject.toml` 中通过 `[project.scripts]` 暴露命令（如 `vulcan-train = "vulcan.cli.train:cli_main"`）；
  - 进一步将 `evaluate` 等工具函数统一暴露在 `vulcan.cli` 或其他更合适的模块下（当前仍有少量交叉依赖）。

