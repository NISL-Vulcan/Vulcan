"""Unit tests for multiple vulcan.framework.models."""
import runpy
import importlib
import sys
import types

import numpy as np
import torch
import pytest
from transformers import RobertaConfig

from vulcan.framework.models.classifier import MLP as mlp_module
from vulcan.framework.models.classifier.MLP import MLP
from vulcan.framework.models import CodeXGLUE_baseline as codex_mod
from vulcan.framework.models.Concoction import Concoction
from vulcan.framework.models.VulBERTa_CNN import VulBERTa_CNN
from vulcan.framework.models import LineVul as linevul_mod
from vulcan.framework.models import VDET as vdet_mod
from vulcan.framework.models import VulBG as vulbg_mod


def test_mlp_forward():
    model = MLP(input_size=64, hidden_layers=[32, 16], output_size=2, dropout_rate=0.3)
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 2)


def test_mlp_single_hidden():
    model = MLP(input_size=10, hidden_layers=[20], output_size=1)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 1)


def test_mlp_training_mode():
    model = MLP(input_size=8, hidden_layers=[16], output_size=2)
    model.train()
    x = torch.randn(2, 8)
    out = model(x)
    assert out.requires_grad or not torch.is_grad_enabled()


def test_mlp_no_hidden_layers():
    model = MLP(input_size=5, hidden_layers=[], output_size=3)
    x = torch.randn(7, 5)
    out = model(x)
    assert out.shape == (7, 3)


def _load_mlp_training_wrapper():
    globs = runpy.run_path(mlp_module.__file__, run_name="__main__")
    return globs["MLPForVulnerabilityDetection"]


def test_mlp_training_wrapper_evaluate_and_predict():
    wrapper_cls = _load_mlp_training_wrapper()
    wrapper = wrapper_cls(input_size=4, hidden_layers=[8], output_size=1, dropout_rate=0.0)

    data = torch.randn(4, 4)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = [(data, labels)]

    loss = wrapper.evaluate(loader)
    assert isinstance(loss, float)

    probs = wrapper.predict([[0.1, 0.2, 0.3, 0.4]])
    assert torch.is_tensor(probs)


def test_mlp_training_wrapper_train_one_epoch():
    wrapper_cls = _load_mlp_training_wrapper()
    wrapper = wrapper_cls(input_size=4, hidden_layers=[8], output_size=1, dropout_rate=0.0)

    train_data = torch.randn(6, 4)
    train_labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    valid_data = torch.randn(2, 4)
    valid_labels = torch.tensor([0.0, 1.0])
    train_loader = [(train_data, train_labels)]
    valid_loader = [(valid_data, valid_labels)]

    wrapper.train(train_loader, valid_loader, epochs=1, early_stopping_rounds=1)


def test_codexglue_baseline_forward_with_mocked_pretrained(monkeypatch):
    class _FakeEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            bs, seqlen = input_ids.shape
            return (torch.ones(bs, seqlen, 2),)

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(name, config):
            return _FakeEncoder()

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(name, do_lower_case=None):
            return object()

    class _FakeConfigObj:
        hidden_size = 2
        hidden_dropout_prob = 0.1

    class _FakeConfigClass:
        @staticmethod
        def from_pretrained(name):
            return _FakeConfigObj()

    monkeypatch.setattr(
        codex_mod,
        "MODEL_CLASSES",
        {"dummy": (_FakeConfigClass, _FakeModelClass, _FakeTokenizerClass)},
    )

    model = codex_mod.CodeXGLUE_baseline(
        encoder=None,
        config=_FakeConfigObj(),
        tokenizer="dummy",
        args={"config_name": "", "model_name_or_path": "dummy"},
    )
    x = torch.tensor([[2, 3, 1], [4, 1, 1]], dtype=torch.long)
    prob = model(x)
    assert prob.shape == (2, 3, 2)
    assert torch.all(prob >= 0) and torch.all(prob <= 1)


def test_concoction_forward_with_and_without_labels():
    class _DummyEmb:
        def word_embeddings(self, input_ids):
            bs, l = input_ids.shape
            return torch.randn(bs, l, 8)

    class _DummyRoberta:
        def __init__(self):
            self.embeddings = _DummyEmb()

        def __call__(self, inputs_embeds, attention_mask, position_ids, token_type_ids):
            bs, l, h = inputs_embeds.shape
            return (torch.randn(bs, l, h),)

    class _DummyEncoder:
        def __init__(self):
            self.roberta = _DummyRoberta()

    class _Cfg:
        hidden_size = 8
        hidden_dropout_prob = 0.0

    model = Concoction(encoder=_DummyEncoder(), config=_Cfg(), tokenizer=None, args={})
    bs, l = 2, 4
    ids1 = torch.randint(0, 10, (bs, l))
    ids2 = torch.randint(0, 10, (bs, l))
    pos1 = torch.tensor([[0, 2, 2, 2], [0, 2, 2, 2]])
    pos2 = torch.tensor([[0, 2, 2, 2], [0, 2, 2, 2]])
    mask1 = torch.ones(bs, l, l, dtype=torch.bool)
    mask2 = torch.ones(bs, l, l, dtype=torch.bool)

    prob = model(ids1, pos1, mask1, ids2, pos2, mask2, labels=None)
    assert prob.shape == (bs, 2)

    labels = torch.tensor([0, 1], dtype=torch.long)
    loss, prob2 = model(ids1, pos1, mask1, ids2, pos2, mask2, labels=labels)
    assert torch.is_tensor(loss)
    assert prob2.shape == (bs, 2)


def test_vulberta_cnn_forward_classification_and_regression():
    class _DummyBaseModel:
        def __call__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            bs, seq_len = input_ids.shape
            return (torch.randn(bs, seq_len, 768), "h2", "h3")

    base = _DummyBaseModel()

    cls_model = VulBERTa_CNN(base_model=base, n_classes=2)
    x = torch.randint(0, 100, (2, 8))
    labels = torch.tensor([0, 1], dtype=torch.long)
    out = cls_model(input_ids=x, labels=labels, return_dict=False)
    assert torch.is_tensor(out[0])  # loss
    assert out[1].shape == (2, 2)   # logits

    reg_model = VulBERTa_CNN(base_model=base, n_classes=1)
    reg_labels = torch.tensor([0.1, -0.2], dtype=torch.float32)
    out2 = reg_model(input_ids=x, labels=reg_labels, return_dict=False)
    assert torch.is_tensor(out2[0])  # loss
    assert out2[1].shape == (2, 1)   # logits


def test_linevul_forward_branches_with_mocked_model_classes(monkeypatch):
    class _DummyOutput:
        def __init__(self, last_hidden_state, attentions):
            self.last_hidden_state = last_hidden_state
            self.attentions = attentions

    class _DummyRoberta:
        def __call__(self, *args, **kwargs):
            output_attentions = kwargs.get("output_attentions", False)
            if output_attentions:
                hidden = torch.randn(2, 6, 8)
                attn = (torch.randn(2, 2, 6, 6),)
                return _DummyOutput(last_hidden_state=hidden, attentions=attn)
            return (torch.randn(2, 6, 8),)

    class _DummyEncoder:
        def __init__(self):
            self.roberta = _DummyRoberta()

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(name, config):
            return _DummyEncoder()

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(name, do_lower_case=None):
            return object()

    class _FakeConfigClass:
        @staticmethod
        def from_pretrained(name):
            return RobertaConfig(
                vocab_size=100,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )

    monkeypatch.setattr(
        linevul_mod,
        "MODEL_CLASSES",
        {"dummy": (_FakeConfigClass, _FakeModelClass, _FakeTokenizerClass)},
    )

    model = linevul_mod.LineVul(
        encoder=None,
        config=None,
        tokenizer="dummy",
        args={"config_name": "", "model_name_or_path": "dummy"},
    )
    input_ids = torch.randint(0, 100, (2, 6))
    labels = torch.tensor([0, 1], dtype=torch.long)

    prob = model(input_ids=input_ids, output_attentions=False)
    assert prob.shape == (2, 2)

    loss, prob2, attentions = model(
        input_ids=input_ids,
        labels=labels,
        output_attentions=True,
    )
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)
    assert isinstance(attentions, tuple)


def _import_mvdetection_module(monkeypatch):
    try:
        return importlib.import_module("vulcan.framework.models.mvdetection")
    except ImportError:
        fake_tg = types.ModuleType("torch_geometric")
        fake_tg_nn = types.ModuleType("torch_geometric.nn")

        class _FallbackRGCNConv:
            def __init__(self, in_channels, out_channels, num_relations=None, num_bases=None):
                self.out_channels = out_channels

            def __call__(self, x, edge_index, edge_type):
                if x.shape[1] == self.out_channels:
                    return x
                if x.shape[1] > self.out_channels:
                    return x[:, : self.out_channels]
                pad = torch.zeros(
                    x.shape[0],
                    self.out_channels - x.shape[1],
                    dtype=x.dtype,
                    device=x.device,
                )
                return torch.cat([x, pad], dim=1)

        def _global_mean_pool(x, batch):
            if batch.numel() == 0:
                return torch.zeros(0, x.shape[1], dtype=x.dtype, device=x.device)
            batch_size = int(batch.max().item()) + 1
            out = []
            for b in range(batch_size):
                mask = batch == b
                if mask.any():
                    out.append(x[mask].mean(dim=0))
            return torch.stack(out, dim=0)

        fake_tg_nn.RGCNConv = _FallbackRGCNConv
        fake_tg_nn.global_mean_pool = _global_mean_pool
        fake_tg.nn = fake_tg_nn

        monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg)
        monkeypatch.setitem(sys.modules, "torch_geometric.nn", fake_tg_nn)
        sys.modules.pop("vulcan.framework.models.mvdetection", None)
        return importlib.import_module("vulcan.framework.models.mvdetection")


def test_custom_graph_conv_layer_edge_update_and_forward(monkeypatch):
    mvd_mod = _import_mvdetection_module(monkeypatch)

    class _DummyRGCN:
        def __call__(self, x, edge_index, edge_type):
            return x + 1.0

    layer = mvd_mod.CustomGraphConvolutionLayer(
        in_features=4, out_features=4, num_relations=2, num_bases=1
    )
    monkeypatch.setattr(layer, "rgcn", _DummyRGCN())
    x = torch.zeros(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    edge_attr = torch.randn(2, 5)

    out_x, out_e = layer(x, edge_index, edge_type, edge_attr)
    assert torch.allclose(out_x, x + 1.0)
    assert torch.equal(out_e, edge_attr)


def test_fsgnn_forward_with_mocked_pool(monkeypatch):
    mvd_mod = _import_mvdetection_module(monkeypatch)

    class _DummyGraphLayer(torch.nn.Module):
        def forward(self, x, edge_index, edge_type, edge_attr):
            return x + 2.0, edge_attr

    monkeypatch.setattr(
        mvd_mod,
        "CustomGraphConvolutionLayer",
        lambda *args, **kwargs: _DummyGraphLayer(),
    )
    monkeypatch.setattr(
        mvd_mod,
        "global_mean_pool",
        lambda x, batch: torch.stack([x[batch == 0].mean(0), x[batch == 1].mean(0)], dim=0),
    )

    model = mvd_mod.FSGNN(
        num_node_features=4,
        num_edge_features=3,
        num_relations=2,
        num_bases=1,
        hidden_dim=4,
        num_classes=2,
    )

    class _Data:
        pass

    data = _Data()
    data.x = torch.randn(6, 4)
    data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data.edge_type = torch.tensor([0, 1], dtype=torch.long)
    data.edge_attr = torch.randn(2, 3)
    data.batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    out = model(data)
    assert out.shape == (2, 2)


def test_custom_graph_layer_node_and_edge_update(monkeypatch):
    mvd_mod = _import_mvdetection_module(monkeypatch)

    class _DummyRGCN:
        def __call__(self, x, edge_index, edge_type):
            return x + 3.0

    layer = mvd_mod.CustomGraphConvolutionLayer(
        in_features=4, out_features=4, num_relations=2, num_bases=1
    )
    monkeypatch.setattr(layer, "rgcn", _DummyRGCN())

    x = torch.zeros(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    edge_attr = torch.randn(2, 5)

    updated_x = layer.node_update(x, edge_index, edge_type, edge_attr)
    updated_e = layer.edge_update(edge_attr)
    assert torch.allclose(updated_x, x + 3.0)
    assert torch.equal(updated_e, edge_attr)


def test_custom_graph_layer_double_underscore_forward(monkeypatch):
    mvd_mod = _import_mvdetection_module(monkeypatch)

    class _DummyRGCN:
        def __call__(self, x, edge_index, edge_type):
            return x + 1.0

    layer = mvd_mod.CustomGraphConvolutionLayer__(
        in_features=4, out_features=4, num_relations=2
    )
    monkeypatch.setattr(layer, "rgcn", _DummyRGCN())

    x = torch.zeros(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    edge_attr = torch.randn(2, 3)
    out = layer(x, edge_index, edge_type, edge_attr)
    assert torch.allclose(out, x + 1.0)


def test_vdet_for_java_forward_with_mocked_automodel(monkeypatch):
    class _DummyAutoModel:
        @staticmethod
        def from_pretrained(name, output_hidden_states=True):
            class _Model:
                def __call__(self, ids, attention_mask=None):
                    bs, seq_len = ids.shape
                    hs = tuple(torch.randn(bs, seq_len, 768) for _ in range(6))
                    return {"hidden_states": hs}

            return _Model()

    monkeypatch.setattr(vdet_mod.transformers, "AutoModel", _DummyAutoModel)
    model = vdet_mod.vdet_for_java(encoder=None, config=None, tokenizer=None, args={})
    ids = torch.randint(0, 100, (2, 5), dtype=torch.long)
    mask = torch.ones_like(ids)
    out = model((ids, mask))
    assert out.shape == (2, model.N_CLASSES)


def test_devign_model_and_prediction_head():
    devign_mod = importlib.import_module("vulcan.framework.models.Devign")

    class _DummyEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            bs = input_ids.shape[0]
            return (torch.full((bs, 2), 0.5),)

    class _Args:
        hidden_size = 4
        num_classes = 2

    class _Cfg:
        hidden_dropout_prob = 0.0

    mdl = devign_mod.Model(encoder=_DummyEncoder(), config=None, tokenizer=None, args={})
    x = torch.tensor([[2, 3], [4, 1]], dtype=torch.long)
    prob = mdl(x)
    assert prob.shape == (2, 2)
    loss, prob2 = mdl(x, labels=torch.tensor([0.0, 1.0]))
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)

    head = devign_mod.PredictionClassification(config=_Cfg(), args=_Args())
    out = head(torch.randn(3, 4))
    assert out.shape == (3, 2)


def _run_devign_forward_with_mocks(module_obj, monkeypatch, graph_format):
    class _DummyWordEmb:
        def __init__(self):
            self.weight = torch.randn(10, 4)

    class _DummyEmb:
        def __init__(self):
            self.word_embeddings = _DummyWordEmb()

    class _DummyRoberta:
        def __init__(self):
            self.embeddings = _DummyEmb()

    class _DummyEncoder:
        def __init__(self):
            self.roberta = _DummyRoberta()

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(name, config):
            return _DummyEncoder()

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(name, do_lower_case=None):
            return object()

    class _FakeConfig:
        hidden_dropout_prob = 0.0

    class _FakeConfigClass:
        @staticmethod
        def from_pretrained(name):
            return _FakeConfig()

    monkeypatch.setattr(
        module_obj,
        "MODEL_CLASSES",
        {"dummy": (_FakeConfigClass, _FakeModelClass, _FakeTokenizerClass)},
    )

    class _DummyGGGNN:
        def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout):
            self.hidden_size = hidden_size

        def __call__(self, features, adj, adj_mask):
            return features

    monkeypatch.setattr(module_obj, "GGGNN", _DummyGGGNN)
    monkeypatch.setattr(
        module_obj,
        "build_graph",
        lambda arr, emb: (np.zeros((2, 20, 20), dtype=float), np.random.randn(2, 20, 4)),
    )
    monkeypatch.setattr(
        module_obj,
        "build_graph_text",
        lambda arr, emb: (np.zeros((2, 20, 20), dtype=float), np.random.randn(2, 20, 4)),
    )
    monkeypatch.setattr(
        module_obj,
        "preprocess_adj",
        lambda adj: (adj, np.ones_like(adj)),
    )
    monkeypatch.setattr(module_obj, "preprocess_features", lambda x: x)

    model = module_obj.Devign(
        encoder=None,
        config=None,
        tokenizer="dummy",
        args={
            "config_name": "",
            "model_name_or_path": "dummy",
            "feature_dim_size": 4,
            "hidden_size": 4,
            "num_GNN_layers": 1,
            "num_classes": 2,
            "format": graph_format,
        },
    )

    if module_obj.__name__.endswith("Devign_origin"):
        class _Input:
            x = torch.randn(2, 5)
        input_value = _Input()
    else:
        input_value = torch.randn(2, 5)

    prob = model(input_value, labels=None)
    assert prob.shape == (2, 2)
    loss, prob2 = model(input_value, labels=torch.tensor([0.0, 1.0]))
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)


def test_devign_forward_with_mocks(monkeypatch):
    devign_mod = importlib.import_module("vulcan.framework.models.Devign")
    _run_devign_forward_with_mocks(devign_mod, monkeypatch, graph_format="uni")


def test_devign_origin_forward_with_mocks(monkeypatch):
    devign_origin_mod = importlib.import_module("vulcan.framework.models.Devign_origin")
    _run_devign_forward_with_mocks(devign_origin_mod, monkeypatch, graph_format="bi")


def test_devign_origin_model_wrapper_and_prediction_head():
    devign_origin_mod = importlib.import_module("vulcan.framework.models.Devign_origin")

    class _DummyEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            bs = input_ids.shape[0]
            # match caller expectation: encoder(...) returns tuple and we take [0]
            return (torch.full((bs, 2), 0.25),)

    wrapper = devign_origin_mod.Model(encoder=_DummyEncoder(), config=None, tokenizer=None, args={})
    x = torch.tensor([[2, 3], [4, 1]], dtype=torch.long)
    prob = wrapper(x)
    assert prob.shape == (2, 2)
    loss, prob2 = wrapper(x, labels=torch.tensor([0.0, 1.0]))
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)

    class _Args:
        hidden_size = 4
        num_classes = 2

    class _Cfg:
        hidden_dropout_prob = 0.0

    # cover default input_size path + explicit input_size path
    head_default = devign_origin_mod.PredictionClassification(config=_Cfg(), args=_Args())
    out = head_default(torch.randn(3, 4))
    assert out.shape == (3, 2)

    head_custom = devign_origin_mod.PredictionClassification(config=_Cfg(), args=_Args(), input_size=6)
    out2 = head_custom(torch.randn(3, 6))
    assert out2.shape == (3, 2)


def test_gnnregvd_model_and_prediction_head():
    gnnregvd_mod = importlib.import_module("vulcan.framework.models.GNNReGVD")

    class _DummyEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            bs = input_ids.shape[0]
            return (torch.full((bs, 2), 0.5),)

    class _Args:
        hidden_size = 4
        num_classes = 2

    class _Cfg:
        hidden_dropout_prob = 0.0

    mdl = gnnregvd_mod.Model(encoder=_DummyEncoder(), config=None, tokenizer=None, args={})
    x = torch.tensor([[2, 3], [4, 1]], dtype=torch.long)
    prob = mdl(x)
    assert prob.shape == (2, 2)
    loss, prob2 = mdl(x, labels=torch.tensor([0.0, 1.0]))
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)

    head = gnnregvd_mod.PredictionClassification(config=_Cfg(), args=_Args())
    out = head(torch.randn(3, 4))
    assert out.shape == (3, 2)


def _run_gnnregvd_forward_with_mocks(module_obj, monkeypatch, graph_format, gnn_name):
    class _DummyWordEmb:
        def __init__(self):
            self.weight = torch.randn(10, 4)

    class _DummyEmb:
        def __init__(self):
            self.word_embeddings = _DummyWordEmb()

    class _DummyRoberta:
        def __init__(self):
            self.embeddings = _DummyEmb()

    class _DummyEncoder:
        def __init__(self):
            self.roberta = _DummyRoberta()

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(name, config):
            return _DummyEncoder()

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(name, do_lower_case=None):
            return object()

    class _FakeConfig:
        hidden_dropout_prob = 0.0
        num_labels = 1

    class _FakeConfigClass:
        @staticmethod
        def from_pretrained(name):
            return _FakeConfig()

    monkeypatch.setattr(
        module_obj,
        "MODEL_CLASSES",
        {"dummy": (_FakeConfigClass, _FakeModelClass, _FakeTokenizerClass)},
    )

    class _DummyGNN:
        def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, residual, att_op):
            self.out_dim = hidden_size

        def __call__(self, features, adj, adj_mask):
            bs = features.shape[0]
            return torch.randn(bs, self.out_dim, dtype=torch.double)

    monkeypatch.setattr(module_obj, "ReGGNN", _DummyGNN)
    monkeypatch.setattr(module_obj, "ReGCN", _DummyGNN)
    monkeypatch.setattr(
        module_obj,
        "build_graph",
        lambda arr, emb, window_size=5: (np.zeros((2, 6, 6), dtype=float), np.random.randn(2, 6, 4)),
    )
    monkeypatch.setattr(
        module_obj,
        "build_graph_text",
        lambda arr, emb, window_size=5: (np.zeros((2, 6, 6), dtype=float), np.random.randn(2, 6, 4)),
    )
    monkeypatch.setattr(module_obj, "preprocess_adj", lambda adj: (adj, np.ones_like(adj)))
    monkeypatch.setattr(module_obj, "preprocess_features", lambda x: x)

    model = module_obj.GNNReGVD(
        encoder=None,
        config=_FakeConfig(),
        tokenizer="dummy",
        args={
            "config_name": "",
            "model_name_or_path": "dummy",
            "feature_dim_size": 4,
            "hidden_size": 4,
            "num_GNN_layers": 1,
            "num_classes": 2,
            "format": graph_format,
            "gnn": gnn_name,
            "remove_residual": False,
            "att_op": "mul",
            "window_size": 5,
        },
    )

    inp = torch.randn(2, 5)
    prob = model(inp, labels=None)
    assert prob.shape == (2, 2)
    loss, prob2 = model(inp, labels=torch.tensor([0.0, 1.0]))
    assert torch.is_tensor(loss)
    assert prob2.shape == (2, 2)


def test_gnnregvd_forward_with_reggnn(monkeypatch):
    gnnregvd_mod = importlib.import_module("vulcan.framework.models.GNNReGVD")
    _run_gnnregvd_forward_with_mocks(
        gnnregvd_mod, monkeypatch, graph_format="uni", gnn_name="ReGGNN"
    )


def test_gnnregvd_forward_with_regcn(monkeypatch):
    gnnregvd_mod = importlib.import_module("vulcan.framework.models.GNNReGVD")
    _run_gnnregvd_forward_with_mocks(
        gnnregvd_mod, monkeypatch, graph_format="bi", gnn_name="ReGCN"
    )


def test_vulbg_codebert_mixin_baseline_and_fusion_forward():
    baseline_input = torch.randn(3, 768)
    bg_input = torch.randn(3, 128)

    baseline_model = vulbg_mod.CodebertMixin(fusion=False)
    baseline_out = baseline_model(baseline_input, bg_input)
    assert baseline_out.shape == (3, 2)

    fusion_model = vulbg_mod.CodebertMixin(fusion=True)
    fusion_out = fusion_model(baseline_input, bg_input)
    assert fusion_out.shape == (3, 2)


def test_vulbg_textcnn_baseline_and_fusion_forward():
    baseline_input = torch.randn(2, 6, vulbg_mod.embedding_dimension)
    bg_input = torch.randn(2, 128)

    baseline_model = vulbg_mod.TextCNN(fusion=False)
    baseline_out = baseline_model(baseline_input, bg_input)
    assert baseline_out.shape == (2, 2)

    fusion_model = vulbg_mod.TextCNN(fusion=True)
    fusion_out = fusion_model(baseline_input, bg_input)
    assert fusion_out.shape == (2, 2)


def _mock_contraflow_graph_ops(contraflow_mod, monkeypatch):
    class _DummyGCNConv:
        def __init__(self, in_channels, out_channels):
            self.out_channels = out_channels

        def __call__(self, x, edge_index):
            return torch.ones(x.shape[0], self.out_channels, device=x.device, dtype=x.dtype)

    def _mean_pool(x, batch):
        return x.mean(dim=0, keepdim=True)

    def _max_pool(x, batch):
        return x.max(dim=0, keepdim=True).values

    monkeypatch.setattr(contraflow_mod, "GCNConv", _DummyGCNConv)
    monkeypatch.setattr(contraflow_mod, "global_mean_pool", _mean_pool)
    monkeypatch.setattr(contraflow_mod, "global_max_pool", _max_pool)


def _import_contraflow_module(monkeypatch):
    try:
        return importlib.import_module("vulcan.framework.models.contraflow")
    except ImportError:
        fake_tg = types.ModuleType("torch_geometric")
        fake_tg_nn = types.ModuleType("torch_geometric.nn")

        class _FallbackGCNConv:
            def __init__(self, in_channels, out_channels):
                self.out_channels = out_channels

            def __call__(self, x, edge_index):
                return torch.ones(
                    x.shape[0], self.out_channels, device=x.device, dtype=x.dtype
                )

        fake_tg_nn.GCNConv = _FallbackGCNConv
        fake_tg_nn.global_mean_pool = lambda x, batch: x.mean(dim=0, keepdim=True)
        fake_tg_nn.global_max_pool = (
            lambda x, batch: x.max(dim=0, keepdim=True).values
        )
        fake_tg.nn = fake_tg_nn

        monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg)
        monkeypatch.setitem(sys.modules, "torch_geometric.nn", fake_tg_nn)
        sys.modules.pop("vulcan.framework.models.contraflow", None)
        return importlib.import_module("vulcan.framework.models.contraflow")


def test_contraflow_statement_encoder_forward(monkeypatch):
    contraflow_mod = _import_contraflow_module(monkeypatch)
    _mock_contraflow_graph_ops(contraflow_mod, monkeypatch)

    model = contraflow_mod.StatementEncoder(num_node_features=4, hidden_dim=6, output_dim=5)
    node_features = torch.randn(7, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model(node_features, edge_index)
    assert out.shape == (1, 5)


def test_contraflow_valueflow_encoder_and_contrastive_loss(monkeypatch):
    contraflow_mod = _import_contraflow_module(monkeypatch)
    _mock_contraflow_graph_ops(contraflow_mod, monkeypatch)

    model = contraflow_mod.ContrastiveLearningModel(input_dim=4, hidden_dim=5, output_dim=4)

    x1 = [torch.randn(6, 4), torch.randn(5, 4), torch.randn(4, 4)]
    e1 = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    ]
    x2 = [torch.randn(6, 4), torch.randn(5, 4), torch.randn(4, 4)]
    e2 = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    ]
    lengths = torch.tensor([3], dtype=torch.long)

    rep1, rep2 = model((x1, e1, lengths), (x2, e2, lengths))
    assert rep1.shape == (1, 10)
    assert rep2.shape == (1, 10)

    loss = model.compute_contrastive_loss(rep1, rep2, temperature=0.5)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def _import_devign_re_module(monkeypatch):
    try:
        return importlib.import_module("vulcan.framework.models.devign_re")
    except ImportError:
        fake_tg = types.ModuleType("torch_geometric")
        fake_tg_nn = types.ModuleType("torch_geometric.nn")
        fake_tg_conv = types.ModuleType("torch_geometric.nn.conv")

        class _FallbackGatedGraphConv:
            def __init__(self, out_channels, num_layers, aggr="add", bias=True):
                self.out_channels = out_channels

            def __call__(self, x, edge_index):
                if x.shape[1] == self.out_channels:
                    return x
                if x.shape[1] > self.out_channels:
                    return x[:, : self.out_channels]
                pad = torch.zeros(
                    x.shape[0],
                    self.out_channels - x.shape[1],
                    dtype=x.dtype,
                    device=x.device,
                )
                return torch.cat([x, pad], dim=1)

        fake_tg_conv.GatedGraphConv = _FallbackGatedGraphConv
        fake_tg_nn.conv = fake_tg_conv
        fake_tg.nn = fake_tg_nn

        monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg)
        monkeypatch.setitem(sys.modules, "torch_geometric.nn", fake_tg_nn)
        monkeypatch.setitem(sys.modules, "torch_geometric.nn.conv", fake_tg_conv)
        sys.modules.pop("vulcan.framework.models.devign_re", None)
        return importlib.import_module("vulcan.framework.models.devign_re")


def test_devign_re_helpers_and_conv_forward(monkeypatch):
    devign_re_mod = _import_devign_re_module(monkeypatch)

    size = devign_re_mod.get_conv_mp_out_size(
        in_size=16,
        last_layer={"out_channels": 3},
        mps=[{"kernel_size": 2, "stride": 2}, {"kernel_size": 2, "stride": 2}],
    )
    assert isinstance(size, int)
    assert size > 0

    linear = torch.nn.Linear(4, 2)
    devign_re_mod.init_weights(linear)
    assert linear.weight.shape == (2, 4)

    conv = devign_re_mod.Conv(
        conv1d_1={"in_channels": 1, "out_channels": 2, "kernel_size": 1},
        conv1d_2={"in_channels": 2, "out_channels": 2, "kernel_size": 1},
        maxpool1d_1={"kernel_size": 1, "stride": 1},
        maxpool1d_2={"kernel_size": 1, "stride": 1},
        fc_1_size=16,
        fc_2_size=8,
    )
    hidden = torch.randn(1, 8)
    x = torch.randn(1, 8)
    out = conv(hidden, x)
    assert out.shape == (1,)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_devign_re_net_and_wrapper(monkeypatch, tmp_path):
    devign_re_mod = _import_devign_re_module(monkeypatch)

    class _DummyGatedGraphConv:
        def __init__(self, out_channels, num_layers, aggr="add", bias=True):
            self.out_channels = out_channels

        def __call__(self, x, edge_index):
            if x.shape[1] == self.out_channels:
                return x
            if x.shape[1] > self.out_channels:
                return x[:, : self.out_channels]
            pad = torch.zeros(
                x.shape[0],
                self.out_channels - x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
            return torch.cat([x, pad], dim=1)

    monkeypatch.setattr(devign_re_mod, "GatedGraphConv", _DummyGatedGraphConv)

    net = devign_re_mod.Net(
        gated_graph_conv_args={"out_channels": 4, "num_layers": 1},
        conv_args={
            "conv1d_1": {"in_channels": 1, "out_channels": 2, "kernel_size": 1},
            "conv1d_2": {"in_channels": 2, "out_channels": 2, "kernel_size": 1},
            "maxpool1d_1": {"kernel_size": 1, "stride": 1},
            "maxpool1d_2": {"kernel_size": 1, "stride": 1},
        },
        emb_size=2,
    )

    class _FakeData:
        def __init__(self):
            self._x = torch.randn(5, 2)
            self.my_data = {"x": self._x}
            self.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        def to(self, _device):
            return self

    out = net([_FakeData(), _FakeData()])
    assert out.ndim == 1
    assert out.shape[0] == 10

    ckpt = tmp_path / "devign_re_net.pt"
    net.save(str(ckpt))
    net.load(str(ckpt))

    wrapper = devign_re_mod.Devign(
        encoder=None,
        config=None,
        tokenizer=None,
        args={
            "gated_graph_conv_args": {"out_channels": 4, "num_layers": 1},
            "conv_args": {
                "conv1d_1": {"in_channels": 1, "out_channels": 2, "kernel_size": 1},
                "conv1d_2": {"in_channels": 2, "out_channels": 2, "kernel_size": 1},
                "maxpool1d_1": {"kernel_size": 1, "stride": 1},
                "maxpool1d_2": {"kernel_size": 1, "stride": 1},
            },
            "emb_size": 2,
        },
    )
    pred = wrapper([_FakeData()])
    assert pred.ndim == 1
    assert pred.shape[0] == 5
    target = torch.ones_like(pred)
    loss = wrapper.loss(pred, target)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_modulesgnn_graph_convolution_forward_and_bias():
    from vulcan.framework.models.modules.GNN import modulesGNN as gnn_mod

    layer = gnn_mod.GraphConvolution(in_features=3, out_features=2, dropout=0.0, bias=True, act=torch.relu)
    x = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
    adj = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    out = layer(x, adj)
    assert out.shape == (1, 2, 2)
    assert torch.isfinite(out).all()


def test_modulesgnn_reggnn_forward_att_ops():
    from vulcan.framework.models.modules.GNN import modulesGNN as gnn_mod

    inputs = torch.randn(2, 4, 3, dtype=torch.float64)
    adj = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.ones(2, 4, 1, dtype=torch.float64)

    m_sum = gnn_mod.ReGGNN(3, 5, num_GNN_layers=1, dropout=0.0, residual=True, att_op="sum")
    out_sum = m_sum(inputs, adj, mask)
    assert out_sum.shape == (2, 5)

    m_concat = gnn_mod.ReGGNN(3, 5, num_GNN_layers=1, dropout=0.0, residual=False, att_op="concat")
    out_concat = m_concat(inputs, adj, mask)
    assert out_concat.shape == (2, 10)

    m_mul = gnn_mod.ReGGNN(3, 5, num_GNN_layers=1, dropout=0.0, residual=True, att_op="mul")
    out_mul = m_mul(inputs, adj, mask)
    assert out_mul.shape == (2, 5)


def test_modulesgnn_regcn_forward_residual_branches():
    from vulcan.framework.models.modules.GNN import modulesGNN as gnn_mod

    inputs = torch.randn(2, 4, 3, dtype=torch.float32)
    adj = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.ones(2, 4, 1, dtype=torch.float32)

    m_res = gnn_mod.ReGCN(3, 4, num_GNN_layers=2, dropout=0.0, residual=True, att_op="sum")
    out_res = m_res(inputs, adj, mask)
    assert out_res.shape == (2, 4)

    m_nores = gnn_mod.ReGCN(3, 4, num_GNN_layers=2, dropout=0.0, residual=False, att_op="concat")
    out_nores = m_nores(inputs, adj, mask)
    assert out_nores.shape == (2, 8)


def test_modulesgnn_gggnn_and_build_graph_functions():
    from vulcan.framework.models.modules.GNN import modulesGNN as gnn_mod

    inputs = torch.randn(1, 3, 2, dtype=torch.float64)
    adj = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, 3, 1, dtype=torch.float64)
    g = gnn_mod.GGGNN(2, 3, num_GNN_layers=1, dropout=0.0)
    out = g(inputs, adj, mask)
    assert out.shape == (1, 3, 3)

    word_embeddings = {1: [0.1, 0.2], 2: [0.3, 0.4], 3: [0.5, 0.6]}
    x_adj, x_feat = gnn_mod.build_graph([[1, 2, 3], [2]], word_embeddings, window_size=2)
    assert len(x_adj) == 2
    assert len(x_feat) == 2
    assert x_adj[0].shape[0] == len(x_feat[0])

    x_adj_t, x_feat_t = gnn_mod.build_graph_text([[1, 2, 3], [3]], word_embeddings, window_size=2)
    assert len(x_adj_t) == 2
    assert len(x_feat_t) == 2
