"""
Lazily import model modules on demand to avoid loading heavy optional dependencies
(e.g., transformers and torch-geometric) during package import.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "GNNReGVD",
    "Devign",
    "LineVul",
    "VulDeepecker",
    "CodeXGLUE_baseline",
    "Russell",
    "VulBERTa_CNN",
    "Concoction",
    "DeepWuKong",
    "IVDmodel",
    "vdet_for_java",
    "TrVD",
]


_MODEL_ATTRS = {
    "GNNReGVD": ("vulcan.framework.models.GNNReGVD", "GNNReGVD"),
    "Devign": ("vulcan.framework.models.devign_re", "Devign"),
    "LineVul": ("vulcan.framework.models.LineVul", "LineVul"),
    "VulDeepecker": ("vulcan.framework.models.VulDeePecker", "VulDeepecker"),
    "CodeXGLUE_baseline": ("vulcan.framework.models.CodeXGLUE_baseline", "CodeXGLUE_baseline"),
    "Russell": ("vulcan.framework.models.Russell_et_net", "Russell"),
    "VulBERTa_CNN": ("vulcan.framework.models.VulBERTa_CNN", "VulBERTa_CNN"),
    "Concoction": ("vulcan.framework.models.Concoction", "Concoction"),
    "DeepWuKong": ("vulcan.framework.models.deepwukong.DWK_gnn", "DeepWuKong"),
    "IVDmodel": ("vulcan.framework.models.IVDetect.IVDetect_model", "IVDmodel"),
    "vdet_for_java": ("vulcan.framework.models.VDET", "vdet_for_java"),
    "TrVD": ("vulcan.framework.models.TrVD", "TrVD"),
}


def __getattr__(name: str):
    if name not in _MODEL_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, cls_name = _MODEL_ATTRS[name]
    module = import_module(module_name)
    return getattr(module, cls_name)