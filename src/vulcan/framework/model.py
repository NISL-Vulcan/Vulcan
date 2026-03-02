from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


# Avoid importing all models at CLI startup, because some models depend on
# optional packages such as torch_geometric/dgl. Load modules on demand.
_MODEL_LOADERS: Dict[str, Tuple[str, str]] = {
    "GNNReGVD": ("vulcan.framework.models.GNNReGVD", "GNNReGVD"),
    "Devign": ("vulcan.framework.models.devign_re", "Devign"),
    "LineVul": ("vulcan.framework.models.LineVul", "LineVul"),
    "VulDeepecker": ("vulcan.framework.models.VulDeePecker", "VulDeepecker"),
    "CodeXGLUE": ("vulcan.framework.models.CodeXGLUE_baseline", "CodeXGLUE_baseline"),
    "Russell": ("vulcan.framework.models.Russell_et_net", "Russell"),
    "VulBERTa_CNN": ("vulcan.framework.models.VulBERTa_CNN", "VulBERTa_CNN"),
    "Concoction": ("vulcan.framework.models.Concoction", "Concoction"),
    "DeepWuKong": ("vulcan.framework.models.deepwukong.DWK_gnn", "DeepWuKong"),
    "IVDetect": ("vulcan.framework.models.IVDetect.IVDetect_model", "IVDmodel"),
    "Vdet_for_java": ("vulcan.framework.models.VDET", "vdet_for_java"),
    "TrVD": ("vulcan.framework.models.TrVD", "TrVD"),
}


def get_model(config: Dict[str, Any]):
    model_name = config["NAME"]
    model_param = config.get("PARAMS") or {}

    if model_name not in _MODEL_LOADERS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {sorted(_MODEL_LOADERS.keys())}")

    module_name, cls_name = _MODEL_LOADERS[model_name]
    module = import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**model_param)
