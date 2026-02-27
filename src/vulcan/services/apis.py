"""
后端相关 API 的统一封装。

统一从 `src/vulcan/services/*_app.py` 暴露后端相关 Flask app 与任务 API。
"""

from __future__ import annotations

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

    等价于调用 `vulcan.services.dataset_optimization_api_app.start_dataset_optimization_api`。
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.start_dataset_optimization_api(max_iterations=max_iterations)


def get_dataset_optimization_status(job_id: str):
    """
    获取数据集优化任务状态的服务封装。

    等价于调用 `vulcan.services.dataset_optimization_api_app.get_optimization_status_api`。
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.get_optimization_status_api(job_id=job_id)


def get_dataset_optimization_logs(job_id: str):
    """
    获取数据集优化任务日志的服务封装。

    等价于调用 `vulcan.services.dataset_optimization_api_app.get_optimization_logs_api`。
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.get_optimization_logs_api(job_id=job_id)


def __getattr__(name: str):
    # 延迟导入，避免在 import vulcan.services.apis 时就启动/初始化 Flask 相关对象。
    if name == "DataCollectionApp":
        from . import data_collection_api_app

        return data_collection_api_app.app

    if name == "DatasetOptimizationJobs":
        from . import dataset_optimization_api_app

        return dataset_optimization_api_app.optimization_jobs

    raise AttributeError(name)

