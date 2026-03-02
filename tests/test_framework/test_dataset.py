"""Unit tests for vulcan.framework.dataset: get_dataset and config validation."""
import json
import types

import numpy as np
import pandas as pd
import pytest
import torch

import vulcan.framework.datasets as datasets_pkg
from vulcan.framework.dataset import get_dataset
from vulcan.framework.datasets.test_source import test_source
from vulcan.framework.datasets import vddata as vddata_mod
from vulcan.framework.datasets import vdet_data as vdet_mod
from vulcan.framework.datasets.vddata_utils.clean_gadget import clean_gadget as ds_clean_gadget
from vulcan.framework.datasets.vddata_utils.vectorize_gadget import GadgetVectorizer


def test_get_dataset_unknown_name_raises():
    config = {
        "DATASET": {
            "NAME": "UnknownDataset",
            "ROOT": "/tmp",
            "PREPROCESS": {"ENABLE": False},
        },
        "TRAIN": {"INPUT_SIZE": 128},
        "EVAL": {"INPUT_SIZE": 128},
    }
    with pytest.raises(ValueError, match="does not exist"):
        get_dataset(config, "train")


class _DummyDataset:
    def __init__(self, split, root, preprocess_format=None, **kwargs):
        self.split = split
        self.root = root
        self.preprocess_format = preprocess_format
        self.kwargs = kwargs


def _make_valid_config():
    return {
        "DATASET": {
            "NAME": "ReGVD",
            "ROOT": "/tmp/data",
            "PARAMS": {"alpha": 1},
            "PREPROCESS": {"ENABLE": True, "COMPOSE": ["resize"]},
        },
        "TRAIN": {"INPUT_SIZE": 64},
        "EVAL": {"INPUT_SIZE": 32},
    }


def test_get_dataset_train_uses_train_preprocess(monkeypatch):
    config = _make_valid_config()

    def _fake_import_module(name):
        class _M:
            ReGVD = _DummyDataset
        return _M

    monkeypatch.setattr("vulcan.framework.dataset.import_module", _fake_import_module)
    monkeypatch.setattr(
        "vulcan.framework.dataset.get_preprocess",
        lambda input_size, compose: {"size": input_size, "compose": compose},
    )

    ds = get_dataset(config, "train")
    assert isinstance(ds, _DummyDataset)
    assert ds.split == "train"
    assert ds.root == "/tmp/data"
    assert ds.kwargs["alpha"] == 1
    assert ds.preprocess_format == {"size": 64, "compose": ["resize"]}


def test_get_dataset_val_uses_eval_preprocess(monkeypatch):
    config = _make_valid_config()

    def _fake_import_module(name):
        class _M:
            ReGVD = _DummyDataset
        return _M

    monkeypatch.setattr("vulcan.framework.dataset.import_module", _fake_import_module)
    monkeypatch.setattr(
        "vulcan.framework.dataset.get_preprocess",
        lambda input_size, compose: {"size": input_size, "compose": compose},
    )

    ds = get_dataset(config, "val")
    assert isinstance(ds, _DummyDataset)
    assert ds.preprocess_format == {"size": 32, "compose": ["resize"]}


def test_get_dataset_invalid_split_keeps_preprocess_none(monkeypatch):
    config = _make_valid_config()

    def _fake_import_module(name):
        class _M:
            ReGVD = _DummyDataset
        return _M

    monkeypatch.setattr("vulcan.framework.dataset.import_module", _fake_import_module)
    monkeypatch.setattr(
        "vulcan.framework.dataset.get_preprocess",
        lambda input_size, compose: {"size": input_size, "compose": compose},
    )

    ds = get_dataset(config, "test")
    assert isinstance(ds, _DummyDataset)
    assert ds.preprocess_format is None


def test_datasets_package_lazy_getattr(monkeypatch):
    sentinel = object()
    fake_module = types.SimpleNamespace(ReGVD=sentinel)
    monkeypatch.setattr(datasets_pkg, "import_module", lambda name: fake_module)
    assert datasets_pkg.ReGVD is sentinel


def test_datasets_package_unknown_attr_raises():
    with pytest.raises(AttributeError):
        _ = datasets_pkg.NOT_A_DATASET


def test_test_source_dataset_basic_flow(tmp_path):
    data_file = tmp_path / "d.jsonl"
    rows = [
        {"func": "int a = 0;", "idx": 1, "target": 0},
        {"func": "int b = 1;", "idx": 2, "target": 1},
    ]
    data_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    args = {
        "train_data_file": str(data_file),
        "eval_data_file": str(data_file),
        "test_data_file": str(data_file),
        "training_percent": 1.0,
    }

    ds = test_source(root=str(tmp_path), split="train", preprocess_format=None, args=args)
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert isinstance(x0, str)
    assert torch.is_tensor(y0)


def test_test_source_dataset_with_preprocess(tmp_path):
    data_file = tmp_path / "d.jsonl"
    rows = [{"func": "int c = 2;", "idx": 3, "target": 0}]
    data_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    args = {
        "train_data_file": str(data_file),
        "eval_data_file": str(data_file),
        "test_data_file": str(data_file),
        "training_percent": 1.0,
    }

    def _preprocess(x, y):
        return x + " //p", y + 1

    ds = test_source(root=str(tmp_path), split="val", preprocess_format=_preprocess, args=args)
    x, y = ds[0]
    assert x.endswith("//p")
    assert y.item() == 1


def test_vddata_parse_file(monkeypatch, tmp_path):
    p = tmp_path / "gadgets.txt"
    p.write_text(
        "\n".join(
            [
                "line_a",
                "line_b",
                "1",
                "---------------------------------",
                "line_c",
                "0",
                "---------------------------------",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vddata_mod, "clean_gadget", lambda g: g)
    got = list(vddata_mod.parse_file(str(p)))
    assert len(got) == 2
    assert got[0][0] == ["line_a", "line_b"]
    assert got[0][1] == 1
    assert got[1][0] == ["line_c"]
    assert got[1][1] == 0


def test_vddata_get_vectors_df(monkeypatch):
    monkeypatch.setattr(vddata_mod, "parse_file", lambda _: [(["a"], 0), (["b"], 1)])

    class _FakeVectorizer:
        def __init__(self, vector_length):
            self.vector_length = vector_length
            self.forward_slices = 1
            self.backward_slices = 1
            self.seen = []

        def add_gadget(self, gadget):
            self.seen.append(gadget)

        def train_model(self):
            return None

        def vectorize(self, gadget):
            return [len(gadget), self.vector_length]

    monkeypatch.setattr(vddata_mod, "GadgetVectorizer", _FakeVectorizer)
    df = vddata_mod.get_vectors_df("dummy.txt", vector_length=7)
    assert list(df.columns) == ["vector", "val"]
    assert len(df) == 2
    assert df.iloc[0]["vector"] == [1, 7]


def test_vddata_dataset_single_class_shortcut(monkeypatch):
    df = pd.DataFrame({"vector": [[1.0, 2.0], [3.0, 4.0]], "val": [1, 1]})
    monkeypatch.setattr(vddata_mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(vddata_mod.pd, "read_pickle", lambda _: df)

    args = {"file_path": "/tmp/demo.txt", "training_percent": 1.0}
    ds = vddata_mod.VDdata(root="/tmp", split="train", tokenizer=None, preprocess_format=None, args=args)
    assert len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (2,)
    assert y.item() == 1


def test_vddata_dataset_train_val_split(monkeypatch):
    vectors = [[float(i), float(i + 1)] for i in range(12)]
    labels = [1] * 6 + [0] * 6
    df = pd.DataFrame({"vector": vectors, "val": labels})

    monkeypatch.setattr(vddata_mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(vddata_mod.pd, "read_pickle", lambda _: df)
    monkeypatch.setattr(vddata_mod.np.random, "choice", lambda arr, size, replace=False: np.array(arr[:size]))
    monkeypatch.setattr(
        vddata_mod,
        "train_test_split",
        lambda idxs, test_size, stratify: (np.array(idxs[:8]), np.array(idxs[8:])),
    )

    args = {"file_path": "/tmp/demo.txt", "training_percent": 1.0}
    train_ds = vddata_mod.VDdata(root="/tmp", split="train", tokenizer=None, preprocess_format=None, args=args)
    val_ds = vddata_mod.VDdata(root="/tmp", split="val", tokenizer=None, preprocess_format=None, args=args)
    assert len(train_ds) == 8
    assert len(val_ds) == 4


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens, max_length, truncation, padding):
        base = [len(text), 99, 100]
        return base[:max_length]

    def encode_plus(
        self,
        text,
        add_special_tokens,
        max_length,
        truncation,
        padding,
        return_attention_mask,
        return_tensors,
    ):
        seq = [1, 2, 3][:max_length]
        padded = seq + [0] * (max_length - len(seq))
        mask = [1] * len(seq) + [0] * (max_length - len(seq))
        return {
            "input_ids": torch.tensor([padded], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def test_vdet_tokenize_truncate():
    tok = _DummyTokenizer()
    out = vdet_mod.tokenize_truncate(tok, ["aa", "bbb"], max_length=2)
    assert len(out) == 2
    assert all(len(x) <= 2 for x in out)


def test_vdet_build_batches(monkeypatch):
    samples = [([1], 0), ([1, 2], 1), ([1, 2, 3], 0)]
    monkeypatch.setattr(vdet_mod.random, "randint", lambda a, b: 0)
    text_batches, label_batches = vdet_mod.build_batches(samples.copy(), batch_size=2)
    assert len(text_batches) == 2
    assert len(label_batches) == 2
    assert sum(len(x) for x in text_batches) == 3


def test_vdet_add_padding_per_batch():
    tok = _DummyTokenizer()
    batch_text = [[[1, 2], [1]], [[3, 4, 5]]]
    batch_labels = [[0, 1], [1]]
    ids, masks, labels = vdet_mod.add_padding_per_batch(tok, batch_text, batch_labels)
    assert len(ids) == 2 and len(masks) == 2 and len(labels) == 2
    assert tuple(ids[0].shape) == (2, 2)
    assert tuple(masks[0].shape) == (2, 2)


def test_vdet_smart_batching(monkeypatch):
    tok = _DummyTokenizer()
    monkeypatch.setattr(vdet_mod.random, "randint", lambda a, b: 0)
    ids, masks, labels = vdet_mod.smart_batching(
        tok,
        max_length=4,
        text_samples=["a", "bb", "ccc"],
        labels=[0, 1, 0],
        batch_size=2,
    )
    assert len(ids) == 2
    assert len(masks) == 2
    assert len(labels) == 2


def test_vdet_tokenize_and_pad():
    tok = _DummyTokenizer()
    ids, masks = vdet_mod.tokenize_and_pad(tok, ["x", "yy"], max_length=5)
    assert tuple(ids.shape) == (2, 5)
    assert tuple(masks.shape) == (2, 5)


def test_ds_clean_gadget_replaces_identifiers_and_literals():
    gadget = [
        'int main(int argc, char **argv) {',
        'foo("str", ch);',
        "char c = 'x';",
        'int x = 1;',
        'x = x + bar(x);',
        'this is a comment */',
        '}',
    ]
    cleaned = ds_clean_gadget(gadget)
    text = " ".join(cleaned)
    assert '""' in text
    assert "''" in text
    assert "FUN1" in text
    assert "VAR" in text
    assert "argc" in text
    assert "argv" in text


def test_gadget_vectorizer_tokenize_and_slice_direction():
    tokens = GadgetVectorizer.tokenize("a += b + foo(c);")
    assert "+=" in tokens
    assert "(" in tokens and ")" in tokens

    tokenized, backward = GadgetVectorizer.tokenize_gadget(["FUN1(VAR1);"])
    assert tokenized
    assert backward is True

    tokenized2, backward2 = GadgetVectorizer.tokenize_gadget(["x = y + 1;"])
    assert tokenized2
    assert backward2 is False


def test_gadget_vectorizer_add_vectorize_and_train(monkeypatch):
    gv = GadgetVectorizer(vector_length=4)
    gv.add_gadget(["FUN1(a);"])
    gv.add_gadget(["x = y + 1;"])
    assert gv.backward_slices == 1
    assert gv.forward_slices == 1

    class _FakeWv(dict):
        def __getitem__(self, token):
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    class _FakeWord2Vec:
        def __init__(self, gadgets, min_count, vector_size, sg):
            assert gadgets
            assert min_count == 1
            assert vector_size == 4
            assert sg == 1
            self.wv = _FakeWv()

    monkeypatch.setattr("vulcan.framework.datasets.vddata_utils.vectorize_gadget.Word2Vec", _FakeWord2Vec)
    gv.train_model()
    assert hasattr(gv, "embeddings")

    vec_backward = gv.vectorize(["FUN1(a);"])
    vec_forward = gv.vectorize(["x = y + 1;"])
    assert vec_backward.shape == (50, 4)
    assert vec_forward.shape == (50, 4)
    assert np.count_nonzero(vec_backward) > 0
    assert np.count_nonzero(vec_forward) > 0
