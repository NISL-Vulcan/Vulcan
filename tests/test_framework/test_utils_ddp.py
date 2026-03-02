"""Unit tests for vulcan.framework.utils.ddp."""
import os
import pytest
import torch

from vulcan.framework.utils.ddp import setup_ddp, cleanup_ddp, reduce_tensor


def test_setup_ddp_without_env():
    """Return gpu=0 when distributed env vars are absent."""
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(k, None)
    gpu = setup_ddp()
    assert gpu == 0


def test_cleanup_ddp_no_init():
    """cleanup should not raise when dist is not initialized."""
    cleanup_ddp()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed not available")
def test_reduce_tensor_requires_init():
    """reduce_tensor should fail when dist is not initialized."""
    if torch.distributed.is_initialized():
        return
    with pytest.raises(Exception):
        reduce_tensor(torch.tensor(1.0))


def test_setup_ddp_with_env(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    calls = {}
    monkeypatch.setattr("vulcan.framework.utils.ddp.torch.cuda.set_device", lambda gpu: calls.setdefault("gpu", gpu))
    monkeypatch.setattr(
        "vulcan.framework.utils.ddp.dist.init_process_group",
        lambda backend, init_method, world_size, rank: calls.setdefault(
            "init", (backend, init_method, world_size, rank)
        ),
    )
    monkeypatch.setattr("vulcan.framework.utils.ddp.dist.barrier", lambda: calls.setdefault("barrier", True))

    gpu = setup_ddp()
    assert gpu == 1
    assert calls["gpu"] == 1
    assert calls["init"] == ("nccl", "env://", 2, 0)
    assert calls["barrier"] is True


def test_cleanup_ddp_with_init(monkeypatch):
    state = {"destroyed": False}
    monkeypatch.setattr("vulcan.framework.utils.ddp.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "vulcan.framework.utils.ddp.dist.destroy_process_group",
        lambda: state.__setitem__("destroyed", True),
    )
    cleanup_ddp()
    assert state["destroyed"] is True


def test_reduce_tensor_happy_path(monkeypatch):
    monkeypatch.setattr("vulcan.framework.utils.ddp.dist.get_world_size", lambda: 2)

    def _fake_all_reduce(tensor, op):
        tensor.mul_(2)

    monkeypatch.setattr("vulcan.framework.utils.ddp.dist.all_reduce", _fake_all_reduce)
    x = torch.tensor(3.0)
    y = reduce_tensor(x)
    assert torch.is_tensor(y)
    assert y.item() == pytest.approx(3.0)
    assert x.item() == pytest.approx(3.0)
