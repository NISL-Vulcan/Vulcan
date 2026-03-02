## Usage Guide (Based on the `src/vulcan` Layout)

### 1. Environment Setup

Run the following commands in the project root:

```bash
cd /home/aejl3/NISL-Vulcan/NISL-Vulcan-2.0/Vulcan

# Recommended: install from pyproject.toml (standard for `src/` layout)
python -m pip install -e .

# Optional: development/test dependencies
python -m pip install -e ".[dev]"
```

### 2. Training and Validation

```bash
# Training
vulcan-train --cfg configs/custom.yaml

# Validation
vulcan-val --cfg configs/custom.yaml

# benchmark
vulcan-benchmark --cfg configs/custom.yaml
```

The corresponding Python entry points are:

```python
from vulcan.cli.train import cli_main as train_cli
from vulcan.cli.val import cli_main as val_cli
from vulcan.cli.benchmark import cli_main as benchmark_cli
```

### 3. Backend Services and API

```bash
# Start backend
python scripts/start_backend.py

# Start integrated services
python scripts/start_services.py
```

Related service wrappers are:

```python
from vulcan.services.backend_server import run_backend
from vulcan.services.apis import DataCollectionApp, DatasetOptimizationJobs
```

For more structural details, see `docs/architecture.md` and `reconstruction_plan.md`.

