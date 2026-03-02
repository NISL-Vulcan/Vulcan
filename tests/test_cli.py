"""Unit tests for vulcan.cli."""
from argparse import Namespace
import importlib
import sys
import types

import pytest
import torch
import vulcan.cli.train as train_module
import vulcan.cli.val as val_module


def test_ordered_load():
    yaml_content = "a: 1\nb: 2"
    result = train_module.ordered_load(yaml_content)
    assert "a" in result
    assert result["a"] == 1
    assert result["b"] == 2


def test_ordered_load_preserves_order():
    yaml_content = "z: 1\na: 2\nm: 3"
    result = train_module.ordered_load(yaml_content)
    keys = list(result.keys())
    assert keys == ["z", "a", "m"] or len(keys) == 3


def test_convert_output():
    pred = torch.tensor([0.2, 0.8, 0.5])
    out = val_module.convert_output(pred)
    assert out.shape == (3, 2)
    torch.testing.assert_close(out[:, 0] + out[:, 1], torch.ones(3))


def test_train_cli_main_calls_pipeline(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    save_dir = tmp_path / "save"
    cfg_path.write_text(
        "SAVE_DIR: '{}'\n".format(save_dir.as_posix()),
        encoding="utf-8",
    )

    called = {}

    monkeypatch.setattr(
        train_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(cfg=str(cfg_path)),
    )
    monkeypatch.setattr(train_module, "fix_seeds", lambda seed: called.setdefault("seed", seed))
    monkeypatch.setattr(train_module, "setup_cudnn", lambda: called.setdefault("cudnn", True))
    monkeypatch.setattr(train_module, "setup_ddp", lambda: 0)
    monkeypatch.setattr(
        train_module,
        "main",
        lambda cfg, gpu, out_dir: called.setdefault("main", (cfg, gpu, out_dir)),
    )
    monkeypatch.setattr(train_module, "cleanup_ddp", lambda: called.setdefault("cleanup", True))

    train_module.cli_main()

    assert called["seed"] == 123456
    assert called["cudnn"] is True
    assert called["cleanup"] is True
    cfg, gpu, out_dir = called["main"]
    assert gpu == 0
    assert cfg["SAVE_DIR"] == save_dir.as_posix()
    assert out_dir == save_dir


def test_val_cli_main_calls_main(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("DEVICE: cpu\n", encoding="utf-8")

    called = {}
    monkeypatch.setattr(
        val_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(cfg=str(cfg_path)),
    )
    monkeypatch.setattr(val_module, "setup_cudnn", lambda: called.setdefault("cudnn", True))
    monkeypatch.setattr(val_module, "main", lambda cfg: called.setdefault("cfg", cfg))

    val_module.cli_main()

    assert called["cudnn"] is True
    assert called["cfg"]["DEVICE"] == "cpu"


def _import_cli_benchmark_with_stubs(monkeypatch):
    fake_models_pkg = types.ModuleType("vulcan.framework.models")
    fake_models_pkg.__all__ = []
    fake_datasets_pkg = types.ModuleType("vulcan.framework.datasets")
    fake_datasets_pkg.__all__ = []

    fake_sklearnex = types.SimpleNamespace(
        patch_sklearn=lambda: None,
        unpatch_sklearn=lambda: None,
    )
    fake_tg_data = types.SimpleNamespace(Data=object)
    fake_tg_module = types.SimpleNamespace(data=fake_tg_data)

    monkeypatch.setitem(sys.modules, "vulcan.framework.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "vulcan.framework.datasets", fake_datasets_pkg)
    monkeypatch.setitem(sys.modules, "sklearnex", fake_sklearnex)
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_tg_module)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_tg_data)

    module = importlib.import_module("vulcan.cli.benchmark")
    return importlib.reload(module)


def _import_cli_export_with_stubs(monkeypatch):
    fake_models_pkg = types.ModuleType("vulcan.framework.models")
    fake_models_pkg.__all__ = []
    fake_datasets_pkg = types.ModuleType("vulcan.framework.datasets")
    fake_datasets_pkg.__all__ = []

    fake_onnx_checker = types.SimpleNamespace(check_model=lambda model: None)
    fake_onnx = types.SimpleNamespace(
        load=lambda path: {"path": path},
        checker=fake_onnx_checker,
        save=lambda model, path: None,
    )
    fake_onnxsim = types.SimpleNamespace(simplify=lambda model: (model, True))

    monkeypatch.setitem(sys.modules, "vulcan.framework.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "vulcan.framework.datasets", fake_datasets_pkg)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxsim", fake_onnxsim)
    module = importlib.import_module("vulcan.cli.export")
    return importlib.reload(module)


def test_benchmark_ordered_load_with_stubbed_import(monkeypatch):
    benchmark_module = _import_cli_benchmark_with_stubs(monkeypatch)
    ordered = benchmark_module.ordered_load("a: 1\nb: 2")
    assert list(ordered.keys()) == ["a", "b"]


def test_benchmark_cli_main_calls_pipeline(tmp_path, monkeypatch):
    benchmark_module = _import_cli_benchmark_with_stubs(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    save_dir = tmp_path / "bench_save"
    cfg_path.write_text("SAVE_DIR: '{}'\n".format(save_dir.as_posix()), encoding="utf-8")

    called = {}
    monkeypatch.setattr(
        benchmark_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(cfg=str(cfg_path)),
    )
    monkeypatch.setattr(benchmark_module, "fix_seeds", lambda seed: called.setdefault("seed", seed))
    monkeypatch.setattr(benchmark_module, "setup_cudnn", lambda: called.setdefault("cudnn", True))
    monkeypatch.setattr(benchmark_module, "setup_ddp", lambda: 1)
    monkeypatch.setattr(
        benchmark_module,
        "main",
        lambda cfg, gpu, out_dir: called.setdefault("main", (cfg, gpu, out_dir)),
    )
    monkeypatch.setattr(benchmark_module, "cleanup_ddp", lambda: called.setdefault("cleanup", True))

    benchmark_module.cli_main()

    assert called["seed"] == 3407
    assert called["cudnn"] is True
    assert called["cleanup"] is True
    cfg, gpu, out_dir = called["main"]
    assert cfg["SAVE_DIR"] == save_dir.as_posix()
    assert gpu == 1
    assert out_dir == save_dir


def test_export_onnx_calls_onnx_pipeline(monkeypatch):
    export_module = _import_cli_export_with_stubs(monkeypatch)
    called = {"export": None, "load": None, "check": None, "save": None}

    fake_onnx_model = {"m": 1}

    monkeypatch.setattr(
        export_module.torch.onnx,
        "export",
        lambda model, inputs, path, **kwargs: called.__setitem__("export", (path, kwargs)),
    )
    monkeypatch.setattr(export_module.onnx, "load", lambda path: called.__setitem__("load", path) or fake_onnx_model)
    monkeypatch.setattr(
        export_module.onnx.checker,
        "check_model",
        lambda model: called.__setitem__("check", model),
    )
    monkeypatch.setattr(export_module, "simplify", lambda model: (model, True))
    monkeypatch.setattr(
        export_module.onnx,
        "save",
        lambda model, path: called.__setitem__("save", (model, path)),
    )

    export_module.export_onnx(model=object(), inputs=torch.randn(1, 3, 8, 8), file="demo_model")

    assert called["export"][0] == "demo_model.onnx"
    assert called["load"] == "demo_model.onnx"
    assert called["check"] == fake_onnx_model
    assert called["save"][1] == "demo_model.onnx"


def test_export_onnx_raises_when_simplify_invalid(monkeypatch):
    export_module = _import_cli_export_with_stubs(monkeypatch)
    monkeypatch.setattr(export_module.torch.onnx, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_module.onnx, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(export_module.onnx.checker, "check_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_module, "simplify", lambda model: (model, False))
    monkeypatch.setattr(export_module.onnx, "save", lambda *args, **kwargs: None)

    with pytest.raises(AssertionError, match="Simplified ONNX model could not be validated"):
        export_module.export_onnx(model=object(), inputs=torch.randn(1, 3, 8, 8), file="demo_model")


def test_export_coreml_missing_dependency_prints_hint(monkeypatch, capsys):
    export_module = _import_cli_export_with_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "coremltools", raising=False)

    export_module.export_coreml(model=object(), inputs=torch.randn(1, 3, 8, 8), file="demo_model")
    out = capsys.readouterr().out
    assert "Please install coremltools" in out


def test_export_cli_main_calls_main(tmp_path, monkeypatch):
    export_module = _import_cli_export_with_stubs(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    save_dir = tmp_path / "export_save"
    cfg_path.write_text("SAVE_DIR: '{}'\n".format(save_dir.as_posix()), encoding="utf-8")

    called = {}
    monkeypatch.setattr(
        export_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(cfg=str(cfg_path)),
    )
    monkeypatch.setattr(export_module, "main", lambda cfg: called.setdefault("cfg", cfg))

    export_module.cli_main()

    assert called["cfg"]["SAVE_DIR"] == save_dir.as_posix()
