"""Unit tests for vulcan.framework.dataset.get_dataloader."""
import sys
import types

import torch
from torch.utils.data import Dataset, RandomSampler

from vulcan.framework.dataset import get_dataloader, graph_collate_fn


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


def test_graph_collate_fn_builds_data_list_and_labels(monkeypatch):
    class _FakeData:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.edge_index = kwargs.get("edge_index")
            self.my_data = kwargs.get("my_data")

    fake_tg_data = types.SimpleNamespace(Data=_FakeData)
    monkeypatch.setitem(sys.modules, "torch_geometric", types.SimpleNamespace(data=fake_tg_data))
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_tg_data)

    edge_index = ("unused", torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    batch = [
        ((edge_index, "payload-a"), torch.tensor(1, dtype=torch.long)),
        ((edge_index, "payload-b"), torch.tensor(0, dtype=torch.long)),
    ]
    data_list, labels = graph_collate_fn(batch)
    assert len(data_list) == 2
    assert isinstance(data_list[0], _FakeData)
    assert tuple(labels.shape) == (2,)
