"""
Explainer implementations for Vulcan.
"""

from .base import BaseExplainer, ExplanationResult
from .coca_dual_view import CocaDualViewExplainer
from .metrics import aggregate_dual_view_metrics

__all__ = [
    "BaseExplainer",
    "ExplanationResult",
    "CocaDualViewExplainer",
    "aggregate_dual_view_metrics",
]

