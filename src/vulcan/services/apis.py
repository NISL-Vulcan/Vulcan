"""
EN API EN.

EN `src/vulcan/services/*_app.py` EN Flask app EN API.
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
    EN.

    EN `vulcan.services.dataset_optimization_api_app.start_dataset_optimization_api`.
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.start_dataset_optimization_api(max_iterations=max_iterations)


def get_dataset_optimization_status(job_id: str):
    """
    EN.

    EN `vulcan.services.dataset_optimization_api_app.get_optimization_status_api`.
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.get_optimization_status_api(job_id=job_id)


def get_dataset_optimization_logs(job_id: str):
    """
    EN.

    EN `vulcan.services.dataset_optimization_api_app.get_optimization_logs_api`.
    """
    from . import dataset_optimization_api_app

    return dataset_optimization_api_app.get_optimization_logs_api(job_id=job_id)


def __getattr__(name: str):
    # EN,EN import vulcan.services.apis EN/EN Flask EN.
    if name == "DataCollectionApp":
        from . import data_collection_api_app

        return data_collection_api_app.app

    if name == "DatasetOptimizationJobs":
        from . import dataset_optimization_api_app

        return dataset_optimization_api_app.optimization_jobs

    raise AttributeError(name)

