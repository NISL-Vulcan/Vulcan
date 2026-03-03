from __future__ import annotations

import copy
from typing import Any, Callable

import torch

from .base import BaseExplainer, ExplanationResult


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(v, device) for v in data)
    return data


def _extract_probability(raw_output: Any) -> float:
    if isinstance(raw_output, (list, tuple)):
        # Many Vulcan models return tuples like (features, logits).
        tensor_candidates = [x for x in raw_output if isinstance(x, torch.Tensor)]
        if not tensor_candidates:
            raise ValueError("Model output tuple/list does not contain tensor values.")
        raw_output = tensor_candidates[-1]

    if isinstance(raw_output, torch.Tensor):
        out = raw_output.detach().float().cpu()
        if out.ndim == 0:
            return float(torch.sigmoid(out).item())
        if out.ndim == 1:
            if out.numel() == 1:
                return float(torch.sigmoid(out[0]).item())
            probs = torch.softmax(out, dim=0)
            return float(probs[min(1, out.numel() - 1)].item())
        if out.ndim >= 2:
            out = out.reshape(out.shape[0], -1)
            if out.shape[1] == 1:
                return float(torch.sigmoid(out[0, 0]).item())
            probs = torch.softmax(out, dim=1)
            return float(probs[0, 1].item())

    if isinstance(raw_output, (float, int)):
        return float(raw_output)
    raise ValueError(f"Unsupported model output type for scoring: {type(raw_output).__name__}")


class CocaDualViewExplainer(BaseExplainer):
    """
    Coca-inspired dual-view explainer.

    It first scores unit importance by leave-one-out perturbation, then evaluates:
      - factual view: keep only selected units
      - counterfactual view: remove selected units
    """

    def __init__(
        self,
        model: Any | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        topk: int = 5,
        max_units: int = 128,
        score_fn: Callable[[Any], float] | None = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.topk = max(1, int(topk))
        self.max_units = max(1, int(max_units))
        self.score_fn = score_fn
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

    def _score(self, sample: Any) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(sample))
        if self.model is None:
            raise ValueError("Either `model` or `score_fn` must be provided to CocaDualViewExplainer.")

        moved = _move_to_device(sample, self.device)
        with torch.no_grad():
            try:
                raw_output = self.model(moved)
            except TypeError:
                if isinstance(moved, dict):
                    raw_output = self.model(**moved)
                else:
                    raise
        return _extract_probability(raw_output)

    @staticmethod
    def _extract_units(sample: Any) -> list[str]:
        if isinstance(sample, dict):
            statements = sample.get("statements")
            if isinstance(statements, list):
                return [str(x) for x in statements]
            graph = sample.get("graph")
            if isinstance(graph, dict):
                node_statements = graph.get("node_statements")
                if isinstance(node_statements, list):
                    return [str(x) for x in node_statements]
            code = sample.get("code")
            if isinstance(code, str):
                return [line for line in code.splitlines() if line.strip()]
        elif isinstance(sample, str):
            return [line for line in sample.splitlines() if line.strip()]
        return []

    @staticmethod
    def _remap_graph(graph: dict[str, Any], kept_indices: list[int]) -> dict[str, Any]:
        old_to_new = {old: new for new, old in enumerate(kept_indices)}
        old_set = set(kept_indices)
        old_edges = graph.get("edge_index") or []
        old_types = graph.get("edge_types") or []
        new_edges: list[list[int]] = []
        new_types: list[str] = []
        for idx, edge in enumerate(old_edges):
            if not isinstance(edge, list) or len(edge) < 2:
                continue
            src, dst = edge[0], edge[1]
            if src in old_set and dst in old_set:
                new_edges.append([old_to_new[src], old_to_new[dst]])
                if idx < len(old_types):
                    new_types.append(str(old_types[idx]))
        graph["edge_index"] = new_edges
        graph["edge_types"] = new_types
        graph["num_nodes"] = len(kept_indices)
        return graph

    def _mask_sample(self, sample: Any, kept_indices: list[int] | None = None, removed_indices: list[int] | None = None) -> Any:
        masked = copy.deepcopy(sample)
        units = self._extract_units(masked)
        if not units:
            return masked

        all_indices = list(range(len(units)))
        if kept_indices is None:
            removed = set(removed_indices or [])
            kept_indices = [i for i in all_indices if i not in removed]
        kept_indices = sorted(set(i for i in kept_indices if 0 <= i < len(units)))
        selected_units = [units[i] for i in kept_indices]

        if isinstance(masked, dict):
            if "statements" in masked and isinstance(masked["statements"], list):
                masked["statements"] = selected_units
            if "code" in masked and isinstance(masked["code"], str):
                masked["code"] = "\n".join(selected_units)
            if "graph" in masked and isinstance(masked["graph"], dict):
                graph = masked["graph"]
                if isinstance(graph.get("node_statements"), list):
                    graph["node_statements"] = selected_units
                masked["graph"] = self._remap_graph(graph, kept_indices=kept_indices)
        elif isinstance(masked, str):
            masked = "\n".join(selected_units)
        return masked

    def explain(self, sample: Any, label: int, sample_id: str) -> ExplanationResult:
        units = self._extract_units(sample)
        if len(units) > self.max_units:
            units = units[: self.max_units]
            sample = self._mask_sample(sample, kept_indices=list(range(len(units))))

        base_score = self._score(sample)
        if not units:
            return ExplanationResult(
                sample_id=str(sample_id),
                label=int(label),
                base_score=base_score,
                factual_score=base_score,
                counterfactual_score=base_score,
                selected_units=[],
                unit_scores=[],
                metadata={"note": "no units available"},
            )

        importance: list[tuple[int, float]] = []
        for idx in range(len(units)):
            perturbed = self._mask_sample(sample, removed_indices=[idx])
            score_without_unit = self._score(perturbed)
            importance.append((idx, base_score - score_without_unit))

        importance.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in importance[: self.topk]]
        selected_scores = [float(score) for _, score in importance[: self.topk]]

        factual_sample = self._mask_sample(sample, kept_indices=selected)
        counterfactual_sample = self._mask_sample(sample, removed_indices=selected)
        factual_score = self._score(factual_sample)
        counterfactual_score = self._score(counterfactual_sample)

        return ExplanationResult(
            sample_id=str(sample_id),
            label=int(label),
            base_score=float(base_score),
            factual_score=float(factual_score),
            counterfactual_score=float(counterfactual_score),
            selected_units=selected,
            unit_scores=selected_scores,
            metadata={
                "num_units": len(units),
                "threshold": self.threshold,
            },
        )

