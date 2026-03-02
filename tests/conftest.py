# conftest.py: shared fixtures and pytest config
import pytest
import torch


@pytest.fixture
def device():
    """Prefer CPU to keep CI / no-GPU environments stable."""
    return torch.device("cpu")


@pytest.fixture
def sample_float_label():
    """Binary labels (0/1) for metrics/loss tests."""
    return torch.tensor([0, 1, 1, 0], dtype=torch.long)


@pytest.fixture
def sample_logits_2class():
    """(N, 2) logits for Metrics.update and selected loss tests."""
    return torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
        dtype=torch.float32,
    )
