"""Unit tests for vulcan.framework.metrics."""
import pytest
import torch

from vulcan.framework.metrics import Metrics


def test_metrics_init(device):
    m = Metrics(num_classes=2, device=device)
    assert m.hist.shape == (2, 2)
    assert m.all_preds == []
    assert m.all_targets == []


def test_metrics_update_and_compute_acc(device, sample_logits_2class, sample_float_label):
    m = Metrics(num_classes=2, device=device)
    m.update(sample_logits_2class.to(device), sample_float_label.to(device))
    acc = m.compute_acc()
    assert 0 <= acc <= 1
    assert isinstance(acc, (int, float))


def test_metrics_compute_f1(device, sample_logits_2class, sample_float_label):
    m = Metrics(num_classes=2, device=device)
    m.update(sample_logits_2class.to(device), sample_float_label.to(device))
    f1 = m.compute_f1()
    assert 0 <= f1 <= 1
    assert isinstance(f1, (int, float))


def test_metrics_compute_rec_prec(device, sample_logits_2class, sample_float_label):
    m = Metrics(num_classes=2, device=device)
    m.update(sample_logits_2class.to(device), sample_float_label.to(device))
    assert 0 <= m.compute_rec() <= 1
    assert 0 <= m.compute_prec() <= 1


def test_metrics_compute_roc_auc(device, sample_logits_2class, sample_float_label):
    m = Metrics(num_classes=2, device=device)
    m.update(sample_logits_2class.to(device), sample_float_label.to(device))
    roc = m.compute_roc_auc()
    assert isinstance(roc, (int, float))
    assert 0 <= roc <= 1 or roc == 0.0


def test_metrics_compute_pr_auc(device, sample_logits_2class, sample_float_label):
    m = Metrics(num_classes=2, device=device)
    m.update(sample_logits_2class.to(device), sample_float_label.to(device))
    pr = m.compute_pr_auc()
    assert isinstance(pr, (int, float))
    assert 0 <= pr <= 1 or pr == 0.0


def test_metrics_single_class_roc_auc_returns_zero(device):
    """When validation has one class only, compute_roc_auc should return 0.0."""
    m = Metrics(num_classes=2, device=device)
    single_label = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
    single_logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], device=device)
    m.update(single_logits, single_label)
    roc = m.compute_roc_auc()
    assert roc == 0.0


def test_metrics_multiple_updates(device):
    m = Metrics(num_classes=2, device=device)
    for _ in range(3):
        logits = torch.randn(4, 2, device=device)
        labels = torch.randint(0, 2, (4,), device=device)
        m.update(logits, labels)
    assert len(m.all_preds) == 12
    assert len(m.all_targets) == 12
    m.compute_acc()
    m.compute_f1()
