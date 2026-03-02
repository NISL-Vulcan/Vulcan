"""Unit tests for vulcan.framework.dataset: get_dataset and config validation."""
import importlib
import json
import sys
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
from vulcan.framework.datasets import linevul as linevul_mod
from vulcan.framework.datasets import regvd as regvd_mod
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


class _SimpleTokenizer:
    cls_token = "<cls>"
    sep_token = "<sep>"
    pad_token_id = 0

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        return [len(t) % 17 for t in tokens]


class _WordLevelEncoded:
    def __init__(self, ids):
        self.ids = ids


class _WordLevelTokenizer:
    def encode(self, text):
        return _WordLevelEncoded([3, 4, 5, 6])


def test_linevul_convert_examples_normal_path():
    tok = _SimpleTokenizer()
    args = types.SimpleNamespace(use_word_level_tokenizer=False, block_size=8)
    feat = linevul_mod.convert_examples_to_features("int a = 0 ;", 1, tok, args)
    assert feat.label == 1
    assert len(feat.input_ids) == 8
    assert feat.input_tokens[0] == "<cls>"
    assert feat.input_tokens[-1] == "<sep>"


def test_linevul_convert_examples_word_level():
    tok = _WordLevelTokenizer()
    args = types.SimpleNamespace(use_word_level_tokenizer=True, block_size=512)
    feat = linevul_mod.convert_examples_to_features("int a = 0 ;", 0, tok, args)
    assert feat.label == 0
    assert feat.input_ids[0] == 0
    assert feat.input_ids[-1] == 1
    assert len(feat.input_ids) == 512


@pytest.mark.parametrize(
    "fn_name,js,label_key",
    [
        ("convert_examples_to_features_draper", {"func": "int a = 0;", "idx": 1, "target": 1}, "target"),
        ("convert_examples_to_features_reveal", {"code": "int b = 1;", "idx": 2, "target": 0}, "target"),
        ("convert_examples_to_features_diverse", {"func": "int c = 2;", "idx": 3, "target": 1}, "target"),
        ("convert_examples_to_features_d2a", {"code": "int d = 3;", "id": 4, "label": 0}, "label"),
        ("convert_examples_to_features_MSR", {"processed_func": "int e = 4;", "index": 5, "target": 1}, "target"),
        ("convert_examples_to_features", {"code": "int f = 5;", "label": 0}, "label"),
        ("convert_examples_to_features_csv", {"Code": "int g = 6;", "id": 7, "isVulnerable": 1}, "isVulnerable"),
    ],
)
def test_regvd_convert_examples_variants(fn_name, js, label_key):
    tok = _SimpleTokenizer()
    args = types.SimpleNamespace(block_size=10)
    fn = getattr(regvd_mod, fn_name)
    feat = fn(js, tok, args)
    assert feat is not None
    assert len(feat.input_ids) == 10
    if label_key == "isVulnerable":
        assert feat.label == int(js[label_key])
    else:
        assert feat.label == js[label_key]


def _import_devign_partial_with_stubs(monkeypatch):
    fake_tg = types.ModuleType("torch_geometric")
    fake_tg_data = types.ModuleType("torch_geometric.data")
    fake_tg_data.DataLoader = object
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_tg_data)
    module = importlib.import_module("vulcan.framework.datasets.devign_partial")
    return importlib.reload(module)


def test_devign_partial_get_ratio():
    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        devign_partial_mod = _import_devign_partial_with_stubs(mp)

        data = list(range(10))
        got = devign_partial_mod.get_ratio(data, 0.3)
        assert got == [0, 1, 2]


def test_devign_partial_load_applies_ratio(monkeypatch):
    devign_partial_mod = _import_devign_partial_with_stubs(monkeypatch)

    class _DF:
        def __init__(self):
            self.memory_called = False

        def info(self, memory_usage=None):
            self.memory_called = memory_usage == "deep"

        def __len__(self):
            return 1

        def __getitem__(self, item):
            return ["ok"][:item.stop]

    df = _DF()
    monkeypatch.setattr(devign_partial_mod.pd, "read_pickle", lambda _: df)
    got = devign_partial_mod.load("/tmp", "demo.pkl", ratio=0.5)
    assert got == []
    assert df.memory_called is True


def test_devign_partial_loads_concat(monkeypatch):
    devign_partial_mod = _import_devign_partial_with_stubs(monkeypatch)

    df1 = pd.DataFrame({"input": ["a"], "target": [0]})
    df2 = pd.DataFrame({"input": ["b"], "target": [1]})
    monkeypatch.setattr(devign_partial_mod, "listdir", lambda _: ["b.pkl", "a.pkl"])
    monkeypatch.setattr(devign_partial_mod, "isfile", lambda _: True)
    monkeypatch.setattr(
        devign_partial_mod,
        "load",
        lambda _path, name: df1 if name == "a.pkl" else df2,
    )
    got = devign_partial_mod.loads("/tmp/demo", ratio=1)
    assert len(got) == 2
    assert set(got["input"]) == {"a", "b"}


def test_devign_partial_train_val_test_split_balanced():
    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        devign_partial_mod = _import_devign_partial_with_stubs(mp)

        df = pd.DataFrame(
            {
                "input": [f"code-{i}" for i in range(20)],
                "target": [0] * 10 + [1] * 10,
            }
        )
        train, test, val = devign_partial_mod.train_val_test_split(df, shuffle=False)
        assert len(train) == 16
        assert len(test) == 2
        assert len(val) == 2
        assert set(train["target"].unique()) == {0, 1}
        assert set(test["target"].unique()) == {0, 1}
        assert set(val["target"].unique()) == {0, 1}


def test_codexglue_convert_examples_and_set_seed():
    import vulcan.framework.datasets.CodeXGLUE as codexglue_mod

    tok = _SimpleTokenizer()
    args = types.SimpleNamespace(block_size=9)
    js = {"func": "int main ( ) { return 0 ; }", "idx": 9, "target": 1}
    feat = codexglue_mod.convert_examples_to_features(js, tok, args)
    assert feat.idx == "9"
    assert feat.label == 1
    assert len(feat.input_ids) == 9

    codexglue_mod.set_seed(123)
    assert codexglue_mod.os.environ["PYHTONHASHSEED"] == "123"


def test_codexglue_dataset_len_and_getitem(monkeypatch, tmp_path):
    import vulcan.framework.datasets.CodeXGLUE as codexglue_mod

    # Stub MODEL_CLASSES to avoid real transformers dependency and downloads
    class _Cfg:
        pass

    class _StubTokenizer:
        cls_token = "<cls>"
        sep_token = "<sep>"
        pad_token_id = 0

        def tokenize(self, text):
            return str(text).split()

        def convert_tokens_to_ids(self, tokens):
            return [len(t) % 13 for t in tokens]

    class _TokClass:
        @staticmethod
        def from_pretrained(name, do_lower_case=None):
            return _StubTokenizer()

    monkeypatch.setattr(
        codexglue_mod,
        "MODEL_CLASSES",
        {"dummy": (_Cfg, object, _TokClass)},
    )

    # Build a simple jsonl data file
    data_file = tmp_path / "codex.jsonl"
    rows = [
        {"func": "int a() { return 0; }", "idx": 1, "target": 0},
        {"func": "int b() { return 1; }", "idx": 2, "target": 1},
    ]
    data_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    args = {
        "train_data_file": str(data_file),
        "eval_data_file": str(data_file),
        "test_data_file": str(data_file),
        "training_percent": 1.0,
        "block_size": 16,
    }

    ds = codexglue_mod.CodeXGLUE(
        root=str(tmp_path),
        split="train",
        tokenizer="dummy",
        preprocess_format=None,
        args=args,
    )
    assert len(ds) == 2

    x0, y0 = ds[0]
    assert torch.is_tensor(x0) and torch.is_tensor(y0)
    assert tuple(x0.shape) == (16,)
    assert y0.item() in (0, 1)

def test_d2alb_trace_dataset_getitem(tmp_path):
    from vulcan.framework.datasets.D2ALB import D2ALB_TraceDataset

    csv_file = tmp_path / "trace.csv"
    pd.DataFrame(
        [{"id": 1, "trace": "t1", "label": 1}, {"id": 2, "trace": "t2", "label": 0}]
    ).to_csv(csv_file, index=False)
    args = types.SimpleNamespace(csv_file=str(csv_file))
    ds = D2ALB_TraceDataset(root=str(tmp_path), split="train", tokenizer=None, preprocess_format=None, args=args)
    assert len(ds) == 2
    x, y = ds[0]
    assert x["id"] == 1
    assert x["trace"] == "t1"
    assert y.item() == 1.0


def test_d2alb_code_dataset_and_download(monkeypatch, tmp_path):
    from vulcan.framework.datasets.D2ALB import D2ALB_CodeDataset

    csv_file = tmp_path / "code.csv"
    pd.DataFrame(
        [
            {
                "id": 9,
                "bug_url": "http://example.com",
                "bug_function": "foo",
                "functions": "bar",
                "label": 1,
            }
        ]
    ).to_csv(csv_file, index=False)
    args = types.SimpleNamespace(csv_file=str(csv_file))
    ds = D2ALB_CodeDataset(root=str(tmp_path), split="train", tokenizer=None, preprocess_format=None, args=args)
    x, y = ds[0]
    assert x["bug_function"] == "foo"
    assert x["functions"] == "bar"
    assert y.item() == 1.0

    class _Resp:
        text = "page-content"

    monkeypatch.setattr("vulcan.framework.datasets.D2ALB.requests.get", lambda _: _Resp())
    assert ds.download_bug_url("http://example.com") == "page-content"


def test_d2alb_trace_code_and_function_dataset(tmp_path):
    from vulcan.framework.datasets.D2ALB import D2ALB_TraceCodeDataset, D2ALB_FunctionDataset

    common = [
        {
            "id": 11,
            "bug_url": "u",
            "bug_function": "bf",
            "functions": "f1",
            "trace": "tr",
            "code": "int a=0;",
            "label": 0,
        }
    ]
    csv_file = tmp_path / "trace_code.csv"
    pd.DataFrame(common).to_csv(csv_file, index=False)
    args = types.SimpleNamespace(csv_file=str(csv_file))

    ds_tc = D2ALB_TraceCodeDataset(root=str(tmp_path), split="train", tokenizer=None, preprocess_format=None, args=args)
    x_tc, y_tc = ds_tc[0]
    assert x_tc["trace"] == "tr"
    assert x_tc["bug_url"] == "u"
    assert y_tc.item() == 0.0

    ds_fn = D2ALB_FunctionDataset(root=str(tmp_path), split="train", tokenizer=None, preprocess_format=None, args=args)
    x_fn, y_fn = ds_fn[0]
    assert x_fn["code"] == "int a=0;"
    assert y_fn.item() == 0.0


def _import_xfg_build_with_stubs(monkeypatch):
    fake_tg = types.ModuleType("torch_geometric")
    fake_tg_data = types.ModuleType("torch_geometric.data")

    class _FakeData:
        def __init__(self, x=None, edge_index=None):
            self.x = x
            self.edge_index = edge_index

        def pin_memory(self):
            return self

        def to(self, _device):
            return self

    class _FakeBatch:
        @staticmethod
        def from_data_list(graphs):
            class _Obj:
                def __init__(self, gs):
                    self.graphs = gs

                def pin_memory(self):
                    return self

                def to(self, _device):
                    return self

            return _Obj(graphs)

    fake_tg_data.Data = _FakeData
    fake_tg_data.Batch = _FakeBatch
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_tg_data)

    module = importlib.import_module("vulcan.framework.datasets.XFGDataset_build")
    return importlib.reload(module)


def test_xfg_build_and_to_torch(monkeypatch):
    xfg_mod = _import_xfg_build_with_stubs(monkeypatch)

    import networkx as nx

    g = nx.DiGraph()
    g.add_node(1, code_sym_token=["a", "b"])
    g.add_node(2, code_sym_token=["c"])
    g.add_edge(1, 2, **{"c/d": "c"})
    g.graph["label"] = 1

    xfg = xfg_mod.XFG(xfg=g)
    assert len(xfg.nodes) == 2
    assert len(xfg.edges) == 1
    assert xfg.label == 1

    class _Vocab:
        @staticmethod
        def get_pad_id():
            return 0

        @staticmethod
        def convert_tokens_to_ids(tokens):
            return [len(t) for t in tokens]

    data = xfg.to_torch(_Vocab(), max_len=4)
    assert tuple(data.x.shape) == (2, 4)
    assert tuple(data.edge_index.shape) == (2, 1)


def test_xfgbatch_len_and_move(monkeypatch):
    xfg_mod = _import_xfg_build_with_stubs(monkeypatch)

    class _Graph:
        def pin_memory(self):
            return self

        def to(self, _device):
            return self

    samples = [
        types.SimpleNamespace(graph=_Graph(), label=1),
        types.SimpleNamespace(graph=_Graph(), label=0),
    ]
    batch = xfg_mod.XFGBatch(samples)
    assert len(batch) == 2
    try:
        batch.pin_memory()
    except NotImplementedError:
        # Some CPU-only/runtime combinations don't support tensor pinning.
        pass
    batch.move_to_device(torch.device("cpu"))
