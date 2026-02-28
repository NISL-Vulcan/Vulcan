"""
按需惰性导入各个数据集，避免在导入 datasets 包时就加载所有重依赖（如 scipy、transformers 等）。
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ReGVD",
    "Devign_Partial",
    "CodeXGLUE",
    "DWK_Dataset",
    "IVDetectDataset",
    "LineVul",
    "VDdata",
    "vdet_data",
]


_DATASET_ATTRS = {
    "ReGVD": ("vulcan.framework.datasets.regvd", "ReGVD"),
    "Devign_Partial": ("vulcan.framework.datasets.devign_partial", "Devign_Partial"),
    "CodeXGLUE": ("vulcan.framework.datasets.CodeXGLUE", "CodeXGLUE"),
    "DWK_Dataset": ("vulcan.framework.datasets.XFGDataset_build", "DWK_Dataset"),
    "IVDetectDataset": ("vulcan.framework.datasets.IVDetect.IVDetectDataset", "IVDetectDataset"),
    "LineVul": ("vulcan.framework.datasets.linevul", "LineVul"),
    "VDdata": ("vulcan.framework.datasets.vddata", "VDdata"),
    "vdet_data": ("vulcan.framework.datasets.vdet_data", "vdet_data"),
}


def __getattr__(name: str):
    if name not in _DATASET_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, cls_name = _DATASET_ATTRS[name]
    module = import_module(module_name)
    return getattr(module, cls_name)