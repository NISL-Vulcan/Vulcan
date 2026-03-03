from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExplanationResult:
    sample_id: str
    label: int
    base_score: float
    factual_score: float
    counterfactual_score: float
    selected_units: list[int] = field(default_factory=list)
    unit_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, sample: Any, label: int, sample_id: str) -> ExplanationResult:
        raise NotImplementedError

    def explain_many(self, samples: list[tuple[Any, int, str]]) -> list[ExplanationResult]:
        return [self.explain(sample=s, label=y, sample_id=sid) for s, y, sid in samples]

