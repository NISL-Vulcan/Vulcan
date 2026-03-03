from __future__ import annotations

from typing import Any

from .base import ExplanationResult


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_dual_view_metrics(
    results: list[ExplanationResult],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Aggregate Coca-style dual-view explanation metrics.

    PN: removing selected units flips vulnerable prediction to non-vulnerable.
    PS: keeping selected units preserves vulnerable prediction.
    FNS: harmonic mean of PN and PS.
    """
    if not results:
        return {
            "count": 0,
            "pn": 0.0,
            "ps": 0.0,
            "fns": 0.0,
            "avg_selected_units": 0.0,
            "avg_base_score": 0.0,
        }

    positives = [r for r in results if r.base_score >= threshold]
    if not positives:
        return {
            "count": len(results),
            "positive_count": 0,
            "pn": 0.0,
            "ps": 0.0,
            "fns": 0.0,
            "avg_selected_units": _safe_mean([float(len(r.selected_units)) for r in results]),
            "avg_base_score": _safe_mean([r.base_score for r in results]),
        }

    pn_hits = 0
    ps_hits = 0
    for item in positives:
        if item.counterfactual_score < threshold:
            pn_hits += 1
        if item.factual_score >= threshold:
            ps_hits += 1

    pn = pn_hits / len(positives)
    ps = ps_hits / len(positives)
    fns = 0.0 if (pn + ps) == 0 else 2 * pn * ps / (pn + ps)
    return {
        "count": len(results),
        "positive_count": len(positives),
        "pn": float(pn),
        "ps": float(ps),
        "fns": float(fns),
        "avg_selected_units": _safe_mean([float(len(r.selected_units)) for r in results]),
        "avg_base_score": _safe_mean([r.base_score for r in results]),
    }

