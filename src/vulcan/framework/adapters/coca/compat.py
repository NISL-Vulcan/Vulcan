from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Iterable, Mapping


class CocaDependencyError(RuntimeError):
    """Raised when a Coca integration dependency is missing."""


def ensure_dependency(package: str, purpose: str) -> None:
    """
    Validate whether an optional package is installed.

    Args:
        package: Package import name.
        purpose: Human-readable usage description shown in the error.
    """
    if find_spec(package) is None:
        raise CocaDependencyError(
            f"Missing optional dependency '{package}' required for {purpose}. "
            f"Please install it in the current environment."
        )


def get_word2vec_vocab(model: Any) -> Iterable[str]:
    """
    Return Word2Vec vocabulary keys for both gensim 3.x and 4.x.
    """
    # gensim 4.x: model.wv.key_to_index
    key_to_index = getattr(getattr(model, "wv", None), "key_to_index", None)
    if key_to_index is not None:
        return key_to_index.keys()

    # gensim 3.x: model.wv.vocab
    vocab = getattr(getattr(model, "wv", None), "vocab", None)
    if vocab is not None:
        return vocab.keys()

    raise ValueError("Unsupported Word2Vec object: cannot find vocabulary mapping.")


def make_word2vec_compatible(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize Word2Vec constructor kwargs to gensim 4.x names.

    Coca scripts commonly use `size` (gensim 3.x). This adapter maps it to
    `vector_size` to keep one code path in Vulcan.
    """
    adapted = dict(params)
    if "size" in adapted and "vector_size" not in adapted:
        adapted["vector_size"] = adapted.pop("size")
    return adapted


def check_legacy_coca_dependencies() -> dict[str, bool]:
    """
    Check legacy Coca-only modules and report availability.

    Notes:
      - `CppCodeAnalyzer` is needed only for Coca's original preprocessing path.
      - `src.data_preprocessors` was used in the old repository layout and is
        intentionally replaced in Vulcan by local transformations.
    """
    return {
        "CppCodeAnalyzer": find_spec("CppCodeAnalyzer") is not None,
        "src.data_preprocessors": find_spec("src.data_preprocessors") is not None,
    }


def ensure_legacy_coca_preprocess_or_raise() -> None:
    checks = check_legacy_coca_dependencies()
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise CocaDependencyError(
            "Legacy Coca preprocessing dependencies are missing: "
            f"{missing}. Use Vulcan adapter transformations in "
            "`vulcan.framework.adapters.coca.transformations` as the replacement path."
        )


@dataclass(frozen=True)
class CocaRuntimeConfig:
    """
    Replacement for Coca's `global_defines` globals.

    This object is intentionally simple and can be created from Vulcan config.
    """

    cur_dir: str
    vul_types: tuple[str, ...]
    cur_vul_type_idx: int
    device: str = "cpu"
    num_classes: int = 2

    @property
    def current_vul_type(self) -> str:
        if not self.vul_types:
            return "default"
        if self.cur_vul_type_idx < 0 or self.cur_vul_type_idx >= len(self.vul_types):
            return self.vul_types[0]
        return self.vul_types[self.cur_vul_type_idx]

    @classmethod
    def from_vulcan_cfg(cls, cfg: Mapping[str, Any]) -> "CocaRuntimeConfig":
        dataset = cfg.get("DATASET", {})
        dataset_root = dataset.get("ROOT", "")
        device = cfg.get("DEVICE", "cpu")
        vul_types = dataset.get("COCA_VUL_TYPES") or ["default"]
        current_idx = int(dataset.get("COCA_VUL_TYPE_IDX", 0))
        num_classes = int(dataset.get("COCA_NUM_CLASSES", 2))
        return cls(
            cur_dir=str(dataset_root),
            vul_types=tuple(str(x) for x in vul_types),
            cur_vul_type_idx=current_idx,
            device=str(device),
            num_classes=num_classes,
        )

    def to_global_defines_dict(self) -> dict[str, Any]:
        """
        Export a dictionary shape compatible with Coca's former global settings.
        """
        return {
            "cur_dir": self.cur_dir,
            "vul_types": list(self.vul_types),
            "cur_vul_type_idx": self.cur_vul_type_idx,
            "device": self.device,
            "num_classes": self.num_classes,
        }

