"""
后端相关 API 的统一封装。

目前直接复用 `scripts/data_collection_api.py` 和 `scripts/dataset_optimization_api.py`
中的实现，并通过统一的导入路径暴露它们，便于其他模块直接调用。
"""

from __future__ import annotations

from scripts import data_collection_api  # noqa: F401
from scripts import dataset_optimization_api  # noqa: F401

# 为了简化调用，这里可按需提供别名：

DataCollectionApp = data_collection_api.app
DatasetOptimizationJobs = dataset_optimization_api.optimization_jobs

