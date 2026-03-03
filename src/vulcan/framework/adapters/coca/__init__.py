"""
Utilities to integrate Coca-style datasets and preprocessing into Vulcan.
"""

from .compat import (
    CocaDependencyError,
    CocaRuntimeConfig,
    check_legacy_coca_dependencies,
    ensure_dependency,
    ensure_legacy_coca_preprocess_or_raise,
    get_word2vec_vocab,
    make_word2vec_compatible,
)
from .convert import convert_coca_directory
from .transformations import SemanticPreservingTransformation

__all__ = [
    "CocaDependencyError",
    "CocaRuntimeConfig",
    "check_legacy_coca_dependencies",
    "ensure_dependency",
    "ensure_legacy_coca_preprocess_or_raise",
    "get_word2vec_vocab",
    "make_word2vec_compatible",
    "convert_coca_directory",
    "SemanticPreservingTransformation",
]

