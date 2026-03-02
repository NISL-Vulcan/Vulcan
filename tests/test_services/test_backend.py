"""Unit tests for vulcan.services."""
import pytest


def test_services_import():
    import vulcan.services
    assert hasattr(vulcan.services, "__name__")


def test_backend_app_legacy_path():
    from vulcan.services.backend_app import legacy_backend_script_path, _repo_root_from_src_layout
    root = _repo_root_from_src_layout()
    assert root.exists()
    path = legacy_backend_script_path(root)
    assert path.suffix == ".py" or "backend" in str(path)


def test_load_legacy_namespace_missing_file(tmp_path):
    from vulcan.services.backend_app import load_legacy_namespace

    with pytest.raises(FileNotFoundError):
        load_legacy_namespace(repo_root=tmp_path)


def test_load_legacy_flask_app_returns_app(monkeypatch):
    from vulcan.services import backend_app

    fake_app = object()
    monkeypatch.setattr(
        backend_app,
        "load_legacy_namespace",
        lambda repo_root=None: {"app": fake_app},
    )
    assert backend_app.load_legacy_flask_app() is fake_app


def test_run_legacy_backend_invokes_app_run(monkeypatch):
    from vulcan.services import backend_app

    called = {}

    class _FakeApp:
        def run(self, **kwargs):
            called["kwargs"] = kwargs

    monkeypatch.setattr(backend_app, "load_legacy_flask_app", lambda repo_root=None: _FakeApp())
    backend_app.run_legacy_backend(host="127.0.0.1", port=5050, debug=False, use_reloader=False)

    assert called["kwargs"]["host"] == "127.0.0.1"
    assert called["kwargs"]["port"] == 5050
    assert called["kwargs"]["debug"] is False
    assert called["kwargs"]["use_reloader"] is False


@pytest.mark.parametrize(
    "status,is_validation,expected",
    [
        ("pending", False, "Pending"),
        ("running", False, "Training in progress"),
        ("running", True, "Validation in progress"),
        ("completed", False, "Completed"),
        ("failed", False, "Failed"),
        ("unknown", False, "Unknown status"),
    ],
)
def test_get_status_description(status, is_validation, expected):
    from vulcan.services import backend_server_app as app_module

    assert app_module.get_status_description(status, is_validation) == expected


@pytest.mark.parametrize(
    "status,expected",
    [
        ("pending", "Validation pending"),
        ("running", "Validation in progress"),
        ("completed", "Validation completed"),
        ("failed", "Validation failed"),
        ("unknown", "Unknown status"),
    ],
)
def test_get_validation_status_description(status, expected):
    from vulcan.services import backend_server_app as app_module

    assert app_module.get_validation_status_description(status) == expected


def test_generate_performance_summary_contains_comparison():
    from vulcan.services import backend_server_app as app_module

    train_metrics = {"accuracy": 0.9, "loss": 0.2, "f1": 0.88}
    val_metrics = {"validation_accuracy": 0.82, "validation_loss": 0.3, "validation_f1": 0.79}
    summary = app_module.generate_performance_summary(train_metrics, val_metrics)

    assert "training" in summary and "validation" in summary and "comparison" in summary
    assert summary["training"]["accuracy"]["value"] == 0.9
    assert summary["validation"]["accuracy"]["value"] == 0.82
    assert summary["comparison"]["accuracy_diff"]["interpretation"] in {"better", "overfitting", "normal"}


def test_generate_final_results_summary_recommendations():
    from vulcan.services import backend_server_app as app_module

    class _Job:
        end_time = "2026-03-01T12:00:00"
        model_name = "DemoModel"
        config_path = "configs/demo.yaml"

        @staticmethod
        def get_duration():
            return "00:10:00"

    train_metrics = {"accuracy": 0.95, "f1": 0.9}
    val_metrics = {"validation_accuracy": 0.75, "validation_f1": 0.6}
    results = app_module.generate_final_results_summary(train_metrics, val_metrics, _Job())

    assert results["training_completed"] is True
    assert results["validation_completed"] is True
    assert "final_accuracy" in results["key_metrics"]
    assert isinstance(results["recommendations"], list)
    assert len(results["recommendations"]) >= 1


def test_generate_validation_summary_and_final_results():
    from vulcan.services import backend_server_app as app_module

    validation_metrics = {
        "validation_accuracy": 0.86,
        "validation_f1": 0.82,
        "validation_precision": 0.84,
        "validation_recall": 0.8,
        "validation_auc": 0.91,
        "validation_overall_score": 0.83,
    }
    summary = app_module.generate_validation_summary(validation_metrics)
    assert summary["accuracy"]["value"] == 0.86
    assert summary["overall_score"]["value"] == 0.83

    class _Job:
        end_time = "2026-03-01T12:00:00"
        model_name = "DemoModel"
        config_path = "configs/demo.yaml"

        @staticmethod
        def get_duration():
            return "00:05:00"

    final_results = app_module.generate_final_validation_results(validation_metrics, _Job())
    assert final_results["validation_completed"] is True
    assert final_results["key_metrics"]["accuracy"] == 0.86
    assert isinstance(final_results["performance_assessment"], list)
    assert isinstance(final_results["recommendations"], list)


def test_health_check_route():
    from vulcan.services import backend_server_app as app_module

    with app_module.app.app_context():
        resp = app_module.health_check()
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload["success"] is True
    assert "timestamp" in payload


def test_models_and_datasets_routes(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.setattr(app_module.config_generator, "get_available_models", lambda: ["A", "B"])
    monkeypatch.setattr(app_module.config_generator, "get_available_datasets", lambda: ["D1"])
    with app_module.app.app_context():
        models_resp = app_module.get_models()
        datasets_resp = app_module.get_datasets()

    assert models_resp.status_code == 200
    assert models_resp.get_json()["models"] == ["A", "B"]
    assert datasets_resp.status_code == 200
    assert datasets_resp.get_json()["datasets"] == ["D1"]


def test_get_training_logs_not_found():
    from vulcan.services import backend_server_app as app_module

    with app_module.app.app_context():
        resp, status = app_module.get_training_logs("not-exists")
    payload = resp.get_json()
    assert status == 404
    assert payload["success"] is False


def test_get_training_status_running(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    class _Job:
        status = "running"
        progress = 55
        current_epoch = 3
        total_epochs = 10
        current_iteration = 12
        total_iterations = 100
        start_time = "2026-03-01T10:00:00"
        end_time = None
        metrics = {"accuracy": 0.8, "loss": 0.3}
        model_name = "DemoModel"
        config_path = "configs/demo.yaml"
        current_phase = "training"
        auto_validation = True

        @staticmethod
        def get_recent_logs(count):
            assert count == 100
            return ["log-a", "log-b"]

        @staticmethod
        def get_full_logs():
            return ["full-log"]

        @staticmethod
        def get_log_file_path():
            return "logs/demo.log"

        @staticmethod
        def get_duration():
            return "00:01:00"

    monkeypatch.setitem(app_module.training_jobs, "job-running", _Job())
    with app_module.app.app_context():
        resp = app_module.get_training_status("job-running")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["status"] == "running"
    assert payload["log_count"] == 2
    assert payload["status_description"] == "Training in progress"
    assert payload["training_metrics"]["accuracy"] == 0.8
    assert payload["validation_metrics"] == {}


def test_get_training_status_completed_includes_final_results(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    class _Job:
        status = "completed"
        progress = 100
        current_epoch = 10
        total_epochs = 10
        current_iteration = 100
        total_iterations = 100
        start_time = "2026-03-01T10:00:00"
        end_time = "2026-03-01T11:00:00"
        metrics = {"accuracy": 0.9, "validation_accuracy": 0.85, "validation_f1": 0.8}
        model_name = "DemoModel"
        config_path = "configs/demo.yaml"
        current_phase = "completed"
        auto_validation = True

        @staticmethod
        def get_recent_logs(count):
            return ["log-a"]

        @staticmethod
        def get_full_logs():
            return ["log-a", "log-b", "log-c"]

        @staticmethod
        def get_log_file_path():
            return "logs/not-exist.log"

        @staticmethod
        def get_duration():
            return "01:00:00"

    monkeypatch.setitem(app_module.training_jobs, "job-completed", _Job())
    with app_module.app.app_context():
        resp = app_module.get_training_status("job-completed")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["status"] == "completed"
    assert payload["log_count"] == 3
    assert "final_results" in payload
    assert "completion_summary" in payload


def test_get_training_status_error_path(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    class _BrokenJob:
        status = "running"
        progress = 0
        current_epoch = 0
        total_epochs = 1
        current_iteration = 0
        total_iterations = 1
        start_time = None
        end_time = None
        metrics = None
        model_name = "X"
        config_path = "Y"

        @staticmethod
        def get_recent_logs(count):
            return []

        @staticmethod
        def get_full_logs():
            return []

        @staticmethod
        def get_log_file_path():
            return "logs/broken.log"

        @staticmethod
        def get_duration():
            return "N/A"

    monkeypatch.setitem(app_module.training_jobs, "job-broken", _BrokenJob())
    with app_module.app.app_context():
        resp, status = app_module.get_training_status("job-broken")
    payload = resp.get_json()

    assert status == 500
    assert payload["success"] is False


def test_get_validation_status_failed_contains_error_details(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    class _ValidationJob:
        is_validation_task = True
        status = "failed"
        progress = 70
        current_epoch = 1
        total_epochs = 1
        current_iteration = 7
        total_iterations = 10
        start_time = "2026-03-01T10:00:00"
        end_time = "2026-03-01T10:05:00"
        metrics = {"validation_accuracy": 0.6, "validation_f1": 0.55}
        model_name = "DemoModel"
        config_path = "configs/demo.yaml"

        @staticmethod
        def get_recent_logs(count):
            return ["ok log", "validation failed due to error", "traceback line"]

        @staticmethod
        def get_log_file_path():
            return "logs/val.log"

    monkeypatch.setitem(app_module.training_jobs, "job-val-failed", _ValidationJob())
    with app_module.app.app_context():
        resp = app_module.get_validation_status("job-val-failed")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["status"] == "failed"
    assert "error_details" in payload
    assert isinstance(payload["error_details"]["error_logs"], list)


def test_get_validation_status_not_found():
    from vulcan.services import backend_server_app as app_module

    with app_module.app.app_context():
        resp, status = app_module.get_validation_status("not-exists")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_get_optimization_status_and_logs(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    class _OptJob:
        status = "completed"
        status_description = "completed"
        progress = 100
        current_iteration = 15
        total_iterations = 15
        start_time = "2026-03-01T10:00:00"
        end_time = "2026-03-01T10:10:00"
        metrics = {"x": 1}
        best_ratio = 0.2
        best_f1_score = 0.88
        current_ratio = 0.2
        current_f1 = 0.88

        @staticmethod
        def get_duration():
            return "10m0s"

        @staticmethod
        def get_full_logs():
            return ["log1", "log2"]

        @staticmethod
        def get_logs(limit=None):
            return ["log1", "log2"] if limit is None else ["log2"]

    monkeypatch.setitem(app_module.optimization_jobs, "job-opt", _OptJob())
    with app_module.app.app_context():
        status_resp = app_module.get_optimization_status("job-opt")
        logs_resp = app_module.get_optimization_logs("job-opt")

    status_payload = status_resp.get_json()
    logs_payload = logs_resp.get_json()

    assert status_payload["success"] is True
    assert status_payload["status"] == "completed"
    assert status_payload["log_count"] == 2
    assert logs_payload["success"] is True
    assert logs_payload["log_count"] == 2


def test_get_optimization_status_not_found():
    from vulcan.services import backend_server_app as app_module

    with app_module.app.app_context():
        resp, status = app_module.get_optimization_status("not-exists")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_get_existing_config_not_found(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.chdir(tmp_path)
    with app_module.app.app_context():
        resp, status = app_module.get_existing_config("missing")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_get_existing_config_success(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "demo.yaml"
    cfg_path.write_text("MODEL:\n  NAME: DemoModel\nDATASET:\n  NAME: DemoDataset\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with app_module.app.app_context():
        resp = app_module.get_existing_config("demo")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_name"] == "demo"
    assert payload["config_data"]["MODEL"]["NAME"] == "DemoModel"


def test_get_existing_config_yaml_error_fallback(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "broken.yaml"
    cfg_path.write_text(":\n  - invalid", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with app_module.app.app_context():
        resp = app_module.get_existing_config("broken")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_data"] == {}


def test_start_training_with_config_id_not_found():
    from vulcan.services import backend_server_app as app_module

    with app_module.app.test_request_context(json={"auto_validation": True}):
        resp, status = app_module.start_training_with_config_id("missing-id")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_start_training_with_config_id_success(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text("MODEL:\n  NAME: DemoModel\nTRAIN:\n  EPOCHS: 7\n", encoding="utf-8")
    monkeypatch.setitem(
        app_module.training_configs,
        "cfg-1",
        {
            "config_path": str(cfg_path),
            "model_name": "DemoModel",
            "config": {"TRAIN": {"EPOCHS": 7}},
            "filename": "demo.yaml",
        },
    )

    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(app_module.threading, "Thread", _FakeThread)
    with app_module.app.test_request_context(json={"auto_validation": True}):
        resp = app_module.start_training_with_config_id("cfg-1")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_id"] == "cfg-1"
    assert payload["auto_validation"] is True
    assert started["started"] is True


def test_start_validation_with_config_id_not_found(monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.setattr(app_module, "ensure_validation_script_format", lambda: True)
    with app_module.app.app_context():
        resp, status = app_module.start_validation_with_config_id("missing-id")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_start_validation_with_config_id_success(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text("MODEL:\n  NAME: DemoModel\n", encoding="utf-8")
    monkeypatch.setitem(
        app_module.training_configs,
        "cfg-val-1",
        {"config_path": str(cfg_path), "model_name": "DemoModel", "filename": "demo.yaml"},
    )
    monkeypatch.setattr(app_module, "ensure_validation_script_format", lambda: True)

    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(app_module.threading, "Thread", _FakeThread)
    with app_module.app.app_context():
        resp = app_module.start_validation_with_config_id("cfg-val-1")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_id"] == "cfg-val-1"
    assert payload["task_type"] == "validation"
    assert started["started"] is True


def test_list_existing_configs_when_dir_missing(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.chdir(tmp_path)
    with app_module.app.app_context():
        resp = app_module.list_existing_configs()
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["configs"] == []


def test_list_existing_configs_with_mixed_files(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "good.yaml").write_text("MODEL:\n  NAME: M1\nDATASET:\n  NAME: D1\n", encoding="utf-8")
    (cfg_dir / "bad.yaml").write_text(":\n- broken", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with app_module.app.app_context():
        resp = app_module.list_existing_configs()
    payload = resp.get_json()

    assert payload["success"] is True
    assert len(payload["configs"]) == 1
    assert payload["configs"][0]["name"] == "good"


def test_start_training_with_existing_config_not_found(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.chdir(tmp_path)
    with app_module.app.test_request_context(json={"auto_validation": False}):
        resp, status = app_module.start_training_with_existing_config("missing")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_start_training_with_existing_config_success(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "demo.yaml").write_text("MODEL:\n  NAME: M1\nTRAIN:\n  EPOCHS: 9\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(app_module.threading, "Thread", _FakeThread)
    with app_module.app.test_request_context(json={"auto_validation": True}):
        resp = app_module.start_training_with_existing_config("demo")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_name"] == "demo"
    assert payload["auto_validation"] is True
    assert started["started"] is True


def test_start_validation_with_config_not_found(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "ensure_validation_script_format", lambda: True)
    with app_module.app.app_context():
        resp, status = app_module.start_validation_with_config("missing")
    payload = resp.get_json()

    assert status == 404
    assert payload["success"] is False


def test_start_validation_with_config_success(tmp_path, monkeypatch):
    from vulcan.services import backend_server_app as app_module

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "demo.yaml").write_text("MODEL:\n  NAME: M1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "ensure_validation_script_format", lambda: False)
    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(app_module.threading, "Thread", _FakeThread)
    with app_module.app.app_context():
        resp = app_module.start_validation_with_config("demo")
    payload = resp.get_json()

    assert payload["success"] is True
    assert payload["config_name"] == "demo"
    assert payload["task_type"] == "validation"
    assert started["started"] is True
