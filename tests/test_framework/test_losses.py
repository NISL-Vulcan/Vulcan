"""Unit tests for vulcan.framework.losses."""
import torch
import pytest

from vulcan.framework.losses import (
    BCEWithLogitsLoss,
    BinaryCrossEntropy,
    CrossEntropy,
    MSELoss,
    BCELoss,
    NLLLoss,
    get_loss,
)


def test_bce_with_logits_loss():
    loss_fn = BCEWithLogitsLoss()
    logits = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)
    target = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    out = loss_fn(logits, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_cross_entropy_loss():
    loss_fn = CrossEntropy()
    logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32)
    target = torch.tensor([1, 0], dtype=torch.long)
    out = loss_fn(logits, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_mse_loss():
    loss_fn = MSELoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.1, 1.9, 3.0])
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_nll_loss():
    loss_fn = NLLLoss()
    log_probs = torch.log_softmax(torch.randn(2, 3), dim=1)
    target = torch.tensor([0, 1], dtype=torch.long)
    out = loss_fn(log_probs, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_get_loss_known():
    for name in ["CrossEntropy", "MSELoss", "BCEWithLogitsLoss"]:
        fn = get_loss(name)
        assert fn is not None
        assert callable(fn)


def test_get_loss_unknown_raises():
    with pytest.raises(AssertionError):
        get_loss("NotALoss")
