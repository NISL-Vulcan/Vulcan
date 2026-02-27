"""
后端相关 API 的统一封装。

目前直接复用 `scripts/data_collection_api.py` 和 `scripts/dataset_optimization_api.py`
中的实现，并通过统一的导入路径暴露它们，便于其他模块直接调用。
"""

from __future__ import annotations

from scripts import data_collection_api
from scripts import dataset_optimization_api

# 为了简化调用，这里提供一些别名/包装函数，统一服务层入口：

DataCollectionApp = data_collection_api.app
DatasetOptimizationJobs = dataset_optimization_api.optimization_jobs


def start_dataset_optimization(max_iterations: int = 15):
    """
    启动数据集优化任务的服务封装。

    等价于调用 `scripts.dataset_optimization_api.start_dataset_optimization_api`。
    """
    return dataset_optimization_api.start_dataset_optimization_api(max_iterations=max_iterations)


def get_dataset_optimization_status(job_id: str):
    """
    获取数据集优化任务状态的服务封装。

    等价于调用 `scripts.dataset_optimization_api.get_optimization_status_api`。
    """
    return dataset_optimization_api.get_optimization_status_api(job_id=job_id)


def get_dataset_optimization_logs(job_id: str):
    """
    获取数据集优化任务日志的服务封装。

    等价于调用 `scripts.dataset_optimization_api.get_optimization_logs_api`。
    """
    return dataset_optimization_api.get_optimization_logs_api(job_id=job_id)

