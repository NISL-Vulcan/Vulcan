"""Unit tests for vulcan.framework.utils.utils."""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from vulcan.framework.config_templates import ConfigTemplate, ConfigTemplateManager
from vulcan.framework.models.modules.GNN import utils as gnn_utils
from vulcan.framework.utils.utils import (
    count_parameters,
    fix_seeds,
    get_model_size,
    setup_cudnn,
    test_model_latency as measure_model_latency,
    time_sync,
    timer,
)


def test_fix_seeds():
    fix_seeds(42)
    a = torch.rand(3)
    fix_seeds(42)
    b = torch.rand(3)
    torch.testing.assert_close(a, b)


def test_count_parameters():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    n = count_parameters(model)
    assert n > 0
    assert isinstance(n, float)
    expected = (10 * 20 + 20) + (20 * 2 + 2)
    assert abs(n * 1e6 - expected) < 1.0


def test_time_sync_returns_float():
    t = time_sync()
    assert isinstance(t, float)
    assert t > 0


def test_setup_cudnn_changes_flags():
    setup_cudnn()
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False


def test_get_model_size_creates_and_cleans_temp_file(tmp_path, monkeypatch):
    model = nn.Linear(4, 2)
    monkeypatch.chdir(tmp_path)
    size = get_model_size(model)
    assert isinstance(size, float)
    assert size > 0
    assert not (tmp_path / "temp.p").exists()


def test_test_model_latency_returns_non_negative_float():
    model = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    latency = measure_model_latency(model, x, use_cuda=False)
    assert isinstance(latency, float)
    assert latency >= 0


def test_timer_decorator_prints_elapsed_and_returns_value(capsys):
    @timer
    def _add(a, b):
        return a + b

    assert _add(1, 2) == 3
    captured = capsys.readouterr()
    assert "Elapsed time:" in captured.out


def test_config_template_generate_config_deep_copy_and_nested_override():
    raw = {"A": {"B": [{"C": 1}]}, "X": 1}
    tmpl = ConfigTemplate("demo", raw, "desc")
    cfg = tmpl.generate_config(**{"A.D": 3, "Y": 4})

    assert cfg["A"]["D"] == 3
    assert cfg["Y"] == 4
    assert raw == {"A": {"B": [{"C": 1}]}, "X": 1}


def test_config_template_manager_queries_and_generate():
    mgr = ConfigTemplateManager()
    templates = mgr.list_templates()
    models = mgr.list_models()

    assert "deepwukong" in templates
    assert "DeepWuKong" in models

    vulberta = mgr.get_template_by_model("vulberta_cnn")
    assert vulberta is not None
    assert vulberta.name == "VulBERTa"

    cfg = mgr.generate_config("devign", **{"TRAIN.BATCH_SIZE": 16, "NEW.FLAG": True})
    assert cfg is not None
    assert cfg["TRAIN"]["BATCH_SIZE"] == 16
    assert cfg["NEW"]["FLAG"] is True

    assert mgr.get_template("not_exists") is None
    assert mgr.generate_config("unknown_model") is None


def test_gnn_parse_index_file_and_sample_mask(tmp_path):
    idx_file = tmp_path / "idx.txt"
    idx_file.write_text("1\n3\n5\n", encoding="utf-8")
    idx = gnn_utils.parse_index_file(str(idx_file))
    mask = gnn_utils.sample_mask(idx, 7)
    assert idx == [1, 3, 5]
    assert mask.dtype == np.bool_
    assert mask.tolist() == [False, True, False, True, False, True, False]


def test_gnn_preprocess_features_and_adj():
    features = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([[1.0, 1.0]], dtype=float),
    ]
    f = gnn_utils.preprocess_features(features)
    assert f.shape == (2, 2, 2)

    adj = [np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0]])]
    a, m = gnn_utils.preprocess_adj(adj)
    assert a.shape == (2, 2, 2)
    assert m.shape == (2, 2, 1)
    assert m[0, 0, 0] == 1.0 and m[1, 1, 0] == 0.0


def test_gnn_sparse_to_tuple_and_normalize_adj():
    mx = sp.coo_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    coords, values, shape = gnn_utils.sparse_to_tuple(mx)
    assert shape == (2, 2)
    assert coords.shape[1] == 2
    assert len(values) == 2

    norm = gnn_utils.normalize_adj(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    assert norm.shape == (2, 2)
    assert np.isfinite(norm).all()


def test_gnn_clean_and_remove_comments():
    s = gnn_utils.clean_str("Hello, World!  ")
    assert s == "hello , world !"
    s2 = gnn_utils.clean_str_sst("A  B\tC")
    assert s2 == "a b c"

    py_src = '"""doc"""\n# cmt\nx = 1\n'
    out_py = gnn_utils.remove_comments_and_docstrings(py_src, "python")
    assert "doc" not in out_py
    assert "# cmt" not in out_py
    assert "x = 1" in out_py

    c_src = "int a = 1; // cmt\nchar* s = \"//not-comment\";\n/*blk*/int b=2;"
    out_c = gnn_utils.remove_comments_and_docstrings(c_src, "c")
    assert "// cmt" not in out_c
    assert "blk" not in out_c
    assert '"//not-comment"' in out_c


def test_gnn_tree_and_index_helpers():
    class _Node:
        def __init__(self, type_, children=None, start=(0, 0), end=(0, 1)):
            self.type = type_
            self.children = children or []
            self.start_point = start
            self.end_point = end

    leaf_a = _Node("identifier", [], (0, 0), (0, 1))
    leaf_comment = _Node("comment", [], (0, 2), (0, 3))
    root = _Node("root", [leaf_a, leaf_comment], (0, 0), (0, 3))

    idx = gnn_utils.tree_to_token_index(root)
    assert idx == [((0, 0), (0, 1))]

    idx_ved = gnn_utils.tree_to_token_index_ved(root)
    assert idx_ved == [((0, 0), (0, 1), "identifier")]

    mapping = {((0, 0), (0, 1)): ("identifier", "x")}
    var_idx = gnn_utils.tree_to_variable_index(leaf_a, mapping)
    assert var_idx == [((0, 0), (0, 1))]

    token = gnn_utils.index_to_code_token(((0, 1), (1, 2)), ["abcd", "WXYZ"])
    assert token == "bcdWX"


def test_gnn_construct_feed_dict_and_chebyshev(monkeypatch):
    placeholders = {
        "labels": "L",
        "features": "F",
        "support": "S",
        "mask": "M",
        "num_features_nonzero": "N",
    }
    features = (None, np.zeros((3, 2)))
    support = np.eye(3)
    mask = np.ones((1, 3, 1))
    labels = np.array([1, 0, 1])
    feed = gnn_utils.construct_feed_dict(features, support, mask, labels, placeholders)
    assert feed["L"] is labels
    assert feed["F"] is features
    assert feed["S"] is support
    assert feed["M"] is mask
    assert feed["N"] == (3, 2)

    monkeypatch.setattr(gnn_utils, "normalize_adj", lambda adj: sp.eye(3))
    monkeypatch.setattr(gnn_utils, "eigsh", lambda lap, k, which: (np.array([2.0]), None))
    cheb = gnn_utils.chebyshev_polynomials(sp.eye(3), k=2)
    assert isinstance(cheb, list)
    assert len(cheb) == 3


def test_gnn_load_word2vec(tmp_path):
    w2v = tmp_path / "w2v.txt"
    w2v.write_text("tok1 0.1 0.2\nbadline\ntok2 1 2 3\n", encoding="utf-8")
    vocab, embd, mapping = gnn_utils.loadWord2Vec(str(w2v))
    assert vocab == ["tok1", "tok2"]
    assert len(embd) == 2
    assert mapping["tok1"] == [0.1, 0.2]
