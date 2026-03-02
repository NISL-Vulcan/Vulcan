"""Unit tests for vulcan.framework.utils.utils."""
import torch
from torch import nn

from vulcan.framework.config_templates import ConfigTemplate, ConfigTemplateManager
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
