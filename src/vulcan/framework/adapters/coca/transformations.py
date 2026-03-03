"""Vulcan-native Coca transformation adapters."""
# pylint: disable=too-few-public-methods

from __future__ import annotations

import keyword
import random
import re
from dataclasses import dataclass


_C_LIKE_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
}


@dataclass(frozen=True)
class CodeTransformationResult:
    """Result payload for a single code transformation."""

    code: str
    success: bool
    transformation: str


class TransformationBase:
    """Common transformation interface."""

    def transform_code(self, code: str) -> CodeTransformationResult:
        """Transform one code snippet."""
        raise NotImplementedError


class SyntacticNoisingTransformation(TransformationBase):
    """
    Lightweight token-level noise used as an in-framework replacement for Coca's
    old `src.data_preprocessors` implementation.
    """

    def __init__(self, noise_ratio: float = 0.15, seed: int | None = None):
        self.noise_ratio = max(0.0, min(1.0, noise_ratio))
        self.rng = random.Random(seed)

    def transform_code(self, code: str) -> CodeTransformationResult:
        tokens = code.split()
        if len(tokens) < 2:
            return CodeTransformationResult(
                code=code,
                success=False,
                transformation="SyntacticNoisingTransformation",
            )

        mode = self.rng.choice(["mask", "drop"])
        transformed = []
        for token in tokens:
            if self.rng.random() < self.noise_ratio:
                if mode == "mask":
                    transformed.append("<mask>")
                continue
            transformed.append(token)

        if not transformed:
            transformed = tokens[:]
        return CodeTransformationResult(
            code=" ".join(transformed),
            success=True,
            transformation="SyntacticNoisingTransformation",
        )


class VariableRenamingTransformation(TransformationBase):
    """
    Conservative regex-based variable renaming that avoids language keywords.
    """

    _identifier_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    def __init__(self, max_renames: int = 5):
        self.max_renames = max_renames

    def transform_code(self, code: str) -> CodeTransformationResult:
        identifiers = []
        for match in self._identifier_pattern.finditer(code):
            token = match.group(0)
            if token in _C_LIKE_KEYWORDS or token in keyword.kwlist:
                continue
            identifiers.append(token)
        if not identifiers:
            return CodeTransformationResult(
                code=code,
                success=False,
                transformation="VariableRenamingTransformation",
            )

        mapping = {}
        for idx, name in enumerate(sorted(set(identifiers))[: self.max_renames], start=1):
            mapping[name] = f"var_{idx}"

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            return mapping.get(token, token)

        transformed = self._identifier_pattern.sub(_replace, code)
        return CodeTransformationResult(
            code=transformed,
            success=True,
            transformation="VariableRenamingTransformation",
        )


class SemanticPreservingTransformation:
    """
    Vulcan-native transformation pipeline replacing Coca's external preprocessor.
    """

    def __init__(self, transforms: list[TransformationBase] | None = None, seed: int | None = None):
        self.rng = random.Random(seed)
        self.transforms = transforms or [
            VariableRenamingTransformation(),
            SyntacticNoisingTransformation(seed=seed),
        ]

    def transform_code(self, code: str) -> tuple[str, str | None]:
        """Apply a random successful transformation, otherwise return original code."""
        candidates = self.transforms[:]
        self.rng.shuffle(candidates)
        for transform in candidates:
            result = transform.transform_code(code)
            if result.success:
                return result.code, result.transformation
        return code, None
