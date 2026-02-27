"""
后端相关 API 的统一封装。

目前直接复用 `scripts/data_collection_api.py` 和 `scripts/dataset_optimization_api.py`
中的实现，并通过统一的导入路径暴露它们，便于其他模块直接调用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
import os
import runpy

def _repo_root_from_src_layout() -> Path:
    # src/vulcan/services/apis.py -> services -> vulcan -> src -> repo_root
    return Path(__file__).resolve().parents[3]


def _load_legacy_namespace(script_relpath: str) -> Mapping[str, Any]:
    repo_root = _repo_root_from_src_layout()
    script_path = repo_root / script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"cannot find legacy script at {script_path}")

    prev_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        return runpy.run_path(str(script_path), run_name=f"vulcan_legacy_{script_path.stem}")
    finally:
        os.chdir(prev_cwd)


@lru_cache()
def _dataset_optimization_ns() -> Mapping[str, Any]:
    return _load_legacy_namespace("scripts/dataset_optimization_api.py")


@lru_cache()
def _data_collection_ns() -> Mapping[str, Any]:
    return _load_legacy_namespace("scripts/data_collection_api.py")


def __getattr__(name: str):
    # 保持兼容：延迟暴露 legacy 对象，避免包导入时强依赖 scripts/。
    if name == "DataCollectionApp":
        return _data_collection_ns().get("app")
    if name == "DatasetOptimizationJobs":
        return _dataset_optimization_ns().get("optimization_jobs")
    raise AttributeError(name)


__all__ = [
    "start_dataset_optimization",
    "get_dataset_optimization_status",
    "get_dataset_optimization_logs",
    "DataCollectionApp",
    "DatasetOptimizationJobs",
]


def start_dataset_optimization(max_iterations: int = 15):
    """
    启动数据集优化任务的服务封装。

    等价于调用 `scripts.dataset_optimization_api.start_dataset_optimization_api`。
    """
    ns = _dataset_optimization_ns()
    fn = ns.get("start_dataset_optimization_api")
    if fn is None:
        raise RuntimeError("legacy script missing 'start_dataset_optimization_api'")
    return fn(max_iterations=max_iterations)


def get_dataset_optimization_status(job_id: str):
    """
    获取数据集优化任务状态的服务封装。

    等价于调用 `scripts.dataset_optimization_api.get_optimization_status_api`。
    """
    ns = _dataset_optimization_ns()
    fn = ns.get("get_optimization_status_api")
    if fn is None:
        raise RuntimeError("legacy script missing 'get_optimization_status_api'")
    return fn(job_id=job_id)


def get_dataset_optimization_logs(job_id: str):
    """
    获取数据集优化任务日志的服务封装。

    等价于调用 `scripts.dataset_optimization_api.get_optimization_logs_api`。
    """
    ns = _dataset_optimization_ns()
    fn = ns.get("get_optimization_logs_api")
    if fn is None:
        raise RuntimeError("legacy script missing 'get_optimization_logs_api'")
    return fn(job_id=job_id)

