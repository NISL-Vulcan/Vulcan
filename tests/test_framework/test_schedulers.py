"""Unit tests for vulcan.framework.schedulers."""
import pytest
import torch
from torch import nn

from vulcan.framework.schedulers import (
    PolyLR,
    WarmupLR,
    WarmupPolyLR,
    WarmupExpLR,
    WarmupCosineLR,
    get_scheduler,
)


def test_polylr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = PolyLR(opt, max_iter=100, decay_iter=1, power=0.9)
    lr0 = sched.get_lr()[0]
    sched.step()
    lr1 = sched.get_lr()[0]
    assert isinstance(lr0, float)
    assert isinstance(lr1, float)


def test_warmup_polylr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = WarmupPolyLR(
        opt, power=0.9, max_iter=200,
        warmup_iter=10, warmup_ratio=0.1, warmup="linear"
    )
    lrs = []
    for _ in range(15):
        lrs.append(sched.get_lr()[0])
        sched.step()
    assert lrs[0] < lrs[5]


def test_warmup_cosine_lr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = WarmupCosineLR(opt, max_iter=100, warmup_iter=5)
    sched.step()
    lr = sched.get_lr()[0]
    assert 0 <= lr <= 0.02


def test_warmup_exp_lr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = WarmupExpLR(opt, gamma=0.9, interval=1, warmup_iter=3)
    for _ in range(5):
        sched.step()
    lr = sched.get_lr()[0]
    assert isinstance(lr, float)


def test_get_scheduler_polylr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = get_scheduler("polylr", opt, max_iter=100, power=1, warmup_iter=0, warmup_ratio=0.0)
    assert isinstance(sched, PolyLR)


def test_get_scheduler_warmuppolylr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = get_scheduler("warmuppolylr", opt, max_iter=100, power=0.9, warmup_iter=10, warmup_ratio=0.1)
    assert isinstance(sched, WarmupPolyLR)


def test_get_scheduler_warmupcosinelr():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = get_scheduler("warmupcosinelr", opt, max_iter=100, power=0, warmup_iter=5, warmup_ratio=0.1)
    assert isinstance(sched, WarmupCosineLR)


def test_get_scheduler_unknown_raises():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2))], lr=0.01)
    with pytest.raises(AssertionError, match="Unavailable scheduler"):
        get_scheduler("unknown", opt, 100, 1, 0, 0.0)
