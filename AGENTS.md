# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Vulcan Detection is a deep learning-based vulnerability detection framework. The single essential service is the **Flask backend API server** on port 5000. There is no frontend in this repository and no database -- the system is fully file-based.

### Python version

The project requires Python `>=3.10,<3.12`. The VM has Python 3.10 installed via deadsnakes PPA. A virtualenv lives at `/workspace/.venv` and must be activated before running anything: `source /workspace/.venv/bin/activate`.

### Running tests

```bash
source .venv/bin/activate
python -m pytest tests/
```

There are 15 pre-existing test failures (placeholder 'EN' return values in `backend_server_app.py` functions, and a monkeypatch incompatibility with PyTorch `nn.Module`). 5 tests are skipped due to an optional `extractors` module not being present, and 1 due to Java not being available.

### Running the backend server

```bash
source .venv/bin/activate
python scripts/start_backend.py
```

The server runs on `http://0.0.0.0:5000`. Key endpoints: `/api/health`, `/api/models`, `/api/datasets`, `/api/generate-config`, `/api/start-training`, `/api/list-configs`.

Before starting, ensure the `configs/`, `generated_configs/`, `output/`, and `logs/` directories exist (the startup script creates `generated_configs/` and `output/` but `configs/` and `logs/` must exist beforehand).

### No linter configured

The project does not have any linter (flake8, ruff, pylint, mypy) configured. The only check script is `scripts/run_checks.sh` which runs pytest.

### PyTorch / CUDA

PyTorch 1.11.0 CPU-only is installed. The project expects CUDA but falls back to CPU. The PyTorch Geometric extensions (`torch-scatter`, `torch-sparse`, `torch-cluster`) are installed for the CPU variant of torch 1.11.0.

### Console scripts

After `pip install -e .`, the following CLI commands are available: `vulcan-train`, `vulcan-val`, `vulcan-benchmark`, `vulcan-export`, `vulcan-backend`.
