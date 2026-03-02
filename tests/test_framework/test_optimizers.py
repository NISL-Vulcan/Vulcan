"""Unit tests for vulcan.framework.optimizers."""
import torch
from torch import nn

from vulcan.framework.optimizers import get_optimizer


def _tiny_model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_get_optimizer_sgd():
    model = _tiny_model()
    opt = get_optimizer(model, "sgd", lr=0.01, weight_decay=0.01)
    assert opt is not None
    assert "param_groups" in dir(opt)


def test_get_optimizer_adamw():
    model = _tiny_model()
    opt = get_optimizer(model, "adamw", lr=1e-4, weight_decay=0.01)
    assert opt is not None


def test_get_optimizer_adamax():
    model = _tiny_model()
    opt = get_optimizer(model, "adamax", lr=1e-4, weight_decay=0.01)
    assert opt is not None


def test_get_optimizer_unknown_falls_back_to_sgd():
    model = _tiny_model()
    opt = get_optimizer(model, "unknown_opt", lr=0.01)
    assert opt is not None


def test_optimizer_step():
    model = _tiny_model()
    opt = get_optimizer(model, "adamw", lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
