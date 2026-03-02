"""Unit tests for vulcan.framework.model: get_model and loading behavior."""
import types

import pytest
import torch

import vulcan.framework.models as models_pkg
from vulcan.framework.model import get_model


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        get_model({"NAME": "UnknownModel"})


def test_get_model_vuldeepecker():
    """VulDeepecker can be constructed with default parameters."""
    config = {"NAME": "VulDeepecker", "PARAMS": {}}
    model = get_model(config)
    assert model is not None
    assert hasattr(model, "forward")
    x = torch.randn(2, 10, 50)
    out = model(x)
    assert out.shape == (2, 2)


def test_get_model_russell():
    """Russell requires WORDS_SIZE and INPUT_SIZE."""
    config = {
        "NAME": "Russell",
        "PARAMS": {"WORDS_SIZE": 1000, "INPUT_SIZE": 100},
    }
    model = get_model(config)
    assert model is not None
    x = torch.randint(0, 1000, (2, 100))
    out = model(x)
    assert out.dim() == 2


def test_get_model_returns_module():
    config = {"NAME": "VulDeepecker"}
    model = get_model(config)
    assert isinstance(model, torch.nn.Module)


def test_models_package_lazy_getattr(monkeypatch):
    sentinel = object()
    fake_module = types.SimpleNamespace(VulDeepecker=sentinel)
    monkeypatch.setattr(models_pkg, "import_module", lambda name: fake_module)
    assert models_pkg.VulDeepecker is sentinel


def test_models_package_unknown_attr_raises():
    with pytest.raises(AttributeError):
        _ = models_pkg.NOT_A_MODEL
