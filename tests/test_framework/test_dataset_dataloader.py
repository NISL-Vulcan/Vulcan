"""Unit tests for vulcan.framework.dataset.get_dataloader."""
import torch
from torch.utils.data import Dataset, RandomSampler

from vulcan.framework.dataset import get_dataloader


class _MinimalDataset(Dataset):
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(4), torch.tensor(i % 2, dtype=torch.long)


def test_get_dataloader_sequence_train():
    ds = _MinimalDataset(20)
    cfg = {}
    dl = get_dataloader(cfg, "train", ds, batch_size=4, num_workers=0, sampler=RandomSampler(ds))
    batch = next(iter(dl))
    assert len(batch) == 2
    assert batch[0].shape[0] == 4
    assert batch[1].shape[0] == 4


def test_get_dataloader_sequence_val():
    ds = _MinimalDataset(5)
    cfg = {}
    dl = get_dataloader(cfg, "val", ds, batch_size=1, num_workers=0)
    batch = next(iter(dl))
    assert batch[0].shape[0] == 1


def test_get_dataloader_geometric_train(monkeypatch):
    ds = _MinimalDataset(10)
    cfg = {"dataloader": "geometric"}
    monkeypatch.setattr("vulcan.framework.dataset.graph_collate_fn", lambda b: ("graph", "labels"))

    dl = get_dataloader(cfg, "train", ds, batch_size=4, num_workers=0, sampler=RandomSampler(ds))
    batch = next(iter(dl))
    assert batch == ("graph", "labels")


def test_get_dataloader_geometric_val(monkeypatch):
    ds = _MinimalDataset(3)
    cfg = {"dataloader": "geometric"}
    monkeypatch.setattr("vulcan.framework.dataset.graph_collate_fn", lambda b: ("graph", "labels"))

    dl = get_dataloader(cfg, "val", ds, batch_size=1, num_workers=0)
    batch = next(iter(dl))
    assert batch == ("graph", "labels")
