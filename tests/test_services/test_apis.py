"""Unit tests for vulcan.services.apis."""

import sys
import types

import pytest

import vulcan.services as services_pkg
import vulcan.services.apis as apis


def test_dataset_optimization_wrappers_delegate(monkeypatch):
    fake_module = types.SimpleNamespace(
        start_dataset_optimization_api=lambda max_iterations: ("start", max_iterations),
        get_optimization_status_api=lambda job_id: ("status", job_id),
        get_optimization_logs_api=lambda job_id: ("logs", job_id),
    )
    monkeypatch.setitem(
        sys.modules,
        "vulcan.services.dataset_optimization_api_app",
        fake_module,
    )
    monkeypatch.setattr(services_pkg, "dataset_optimization_api_app", fake_module, raising=False)

    assert apis.start_dataset_optimization(max_iterations=7) == ("start", 7)
    assert apis.get_dataset_optimization_status("job-1") == ("status", "job-1")
    assert apis.get_dataset_optimization_logs("job-1") == ("logs", "job-1")


def test_getattr_data_collection_app(monkeypatch):
    fake_app = object()
    fake_module = types.SimpleNamespace(app=fake_app)
    monkeypatch.setitem(sys.modules, "vulcan.services.data_collection_api_app", fake_module)
    monkeypatch.setattr(services_pkg, "data_collection_api_app", fake_module, raising=False)

    assert apis.__getattr__("DataCollectionApp") is fake_app


def test_getattr_dataset_optimization_jobs(monkeypatch):
    fake_jobs = {"job-1": {"status": "running"}}
    fake_module = types.SimpleNamespace(optimization_jobs=fake_jobs)
    monkeypatch.setitem(
        sys.modules,
        "vulcan.services.dataset_optimization_api_app",
        fake_module,
    )
    monkeypatch.setattr(services_pkg, "dataset_optimization_api_app", fake_module, raising=False)

    assert apis.__getattr__("DatasetOptimizationJobs") is fake_jobs


def test_getattr_unknown_raises():
    with pytest.raises(AttributeError):
        apis.__getattr__("UnknownApiAttr")


def test_dataset_optimization_api_start_success(monkeypatch):
    import vulcan.services.dataset_optimization_api_app as opt_api

    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(opt_api.threading, "Thread", _FakeThread)
    result = opt_api.start_dataset_optimization_api(max_iterations=9)

    assert result["success"] is True
    assert result["job_id"] in opt_api.optimization_jobs
    assert opt_api.optimization_jobs[result["job_id"]].total_iterations == 9
    assert started["started"] is True


def test_dataset_optimization_api_status_and_logs(monkeypatch):
    import vulcan.services.dataset_optimization_api_app as opt_api

    class _Job:
        status = "completed"
        progress = 100
        current_iteration = 15
        total_iterations = 15
        start_time = "2026-03-01T10:00:00"
        end_time = "2026-03-01T10:10:00"
        metrics = {"best_ratio": 0.2}
        best_ratio = 0.2
        best_f1_score = 0.88

        @staticmethod
        def get_duration():
            return "10m0s"

        @staticmethod
        def get_full_logs():
            return ["l1", "l2"]

        @staticmethod
        def get_recent_logs(count):
            return ["l2"]

    monkeypatch.setitem(opt_api.optimization_jobs, "job-opt-api", _Job())
    status = opt_api.get_optimization_status_api("job-opt-api")
    logs = opt_api.get_optimization_logs_api("job-opt-api")

    assert status["success"] is True
    assert status["status"] == "completed"
    assert "best ratio" in status["status_description"].lower()
    assert logs["success"] is True
    assert logs["log_count"] == 2


def test_dataset_optimization_api_not_found():
    import vulcan.services.dataset_optimization_api_app as opt_api

    status = opt_api.get_optimization_status_api("missing")
    logs = opt_api.get_optimization_logs_api("missing")

    assert status["success"] is False
    assert logs["success"] is False


def test_data_collection_api_start_status_logs(monkeypatch):
    import vulcan.services.data_collection_api_app as dc_api

    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=()):
            started["target"] = target
            started["args"] = args
            self.daemon = False

        def start(self):
            started["started"] = True

    monkeypatch.setattr(dc_api.threading, "Thread", _FakeThread)
    with dc_api.app.test_request_context(json={"collection_type": "unit"}):
        start_resp = dc_api.start_data_collection()
    start_payload = start_resp.get_json()
    job_id = start_payload["job_id"]

    assert start_payload["success"] is True
    assert start_payload["collection_type"] == "unit"
    assert started["started"] is True

    class _Job:
        status = "completed"
        status_description = "completed"
        start_time = None
        end_time = None
        logs = ["a", "b"]
        collection_type = "unit"
        collection_results = {"success": True}
        error = None

        @staticmethod
        def get_duration():
            return "not started"

        @staticmethod
        def get_recent_logs(count):
            return ["b"]

        @staticmethod
        def get_logs(limit=None):
            return ["a", "b"] if limit is None else ["b"]

    monkeypatch.setitem(dc_api.jobs, job_id, _Job())
    with dc_api.app.app_context():
        status_resp = dc_api.get_data_collection_status(job_id)
    with dc_api.app.test_request_context("/api/data-collection-logs?limit=1"):
        logs_resp = dc_api.get_data_collection_logs(job_id)

    status_payload = status_resp.get_json()
    logs_payload = logs_resp.get_json()
    assert status_payload["success"] is True
    assert status_payload["status"] == "completed"
    assert "collection_results" in status_payload
    assert logs_payload["success"] is True
    assert logs_payload["logs"] == ["b"]


def test_data_collection_api_not_found_and_health():
    import vulcan.services.data_collection_api_app as dc_api

    with dc_api.app.app_context():
        missing_status_resp, missing_status_code = dc_api.get_data_collection_status("missing")
        missing_logs_resp, missing_logs_code = dc_api.get_data_collection_logs("missing")
        health_resp = dc_api.health_check()

    assert missing_status_code == 404
    assert missing_status_resp.get_json()["success"] is False
    assert missing_logs_code == 404
    assert missing_logs_resp.get_json()["success"] is False
    assert health_resp.get_json()["status"] == "healthy"


def test_dataset_optimization_server_start_status_logs_health(monkeypatch):
    import vulcan.services.dataset_optimization_server_app as opt_srv

    started = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(opt_srv.threading, "Thread", _FakeThread)
    with opt_srv.app.test_request_context(json={"max_iterations": 6}):
        start_resp = opt_srv.start_dataset_optimization()
    start_payload = start_resp.get_json()
    job_id = start_payload["job_id"]

    assert start_payload["success"] is True
    assert started["started"] is True
    assert opt_srv.optimization_jobs[job_id].total_iterations == 6

    class _Job:
        status = "completed"
        progress = 100
        current_iteration = 6
        total_iterations = 6
        start_time = "2026-03-01T10:00:00"
        end_time = "2026-03-01T10:01:00"
        metrics = {"best_ratio": 0.2}
        best_ratio = 0.2
        best_f1_score = 0.88

        @staticmethod
        def get_duration():
            return "1m0s"

        @staticmethod
        def get_full_logs():
            return ["log1", "log2"]

        @staticmethod
        def get_recent_logs(count):
            return ["log2"]

    monkeypatch.setitem(opt_srv.optimization_jobs, job_id, _Job())
    with opt_srv.app.app_context():
        status_resp = opt_srv.get_optimization_status(job_id)
        logs_resp = opt_srv.get_optimization_logs(job_id)
        health_resp = opt_srv.health_check()

    status_payload = status_resp.get_json()
    logs_payload = logs_resp.get_json()
    health_payload = health_resp.get_json()
    assert status_payload["success"] is True
    assert status_payload["status"] == "completed"
    assert logs_payload["success"] is True
    assert logs_payload["log_count"] == 2
    assert health_payload["success"] is True


def test_dataset_optimization_server_not_found():
    import vulcan.services.dataset_optimization_server_app as opt_srv

    with opt_srv.app.app_context():
        status_resp, status_code = opt_srv.get_optimization_status("missing")
        logs_resp, logs_code = opt_srv.get_optimization_logs("missing")

    assert status_code == 404
    assert status_resp.get_json()["success"] is False
    assert logs_code == 404
    assert logs_resp.get_json()["success"] is False


def test_resource_updater_semantic_search_success(monkeypatch):
    from vulcan.services.resource_updater import ResourceUpdater

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [
                    {"title": "P1", "abstract": "A1", "year": 2024, "authors": [{"name": "Alice"}], "url": "u1"},
                    {"title": "P2", "abstract": "A2", "year": 2022, "authors": [{"name": "Bob"}], "url": "u2"},
                ]
            }

    monkeypatch.setattr("vulcan.services.resource_updater.requests.get", lambda *args, **kwargs: _Resp())
    updater = ResourceUpdater()
    papers = updater.semantic_search_papers("demo", year_from=2023, top_k=5)

    assert len(papers) == 1
    assert papers[0]["title"] == "P1"
    assert papers[0]["authors"] == ["Alice"]


def test_resource_updater_semantic_search_failure(monkeypatch):
    from vulcan.services.resource_updater import ResourceUpdater

    class _Resp:
        status_code = 500
        text = "server error"

    monkeypatch.setattr("vulcan.services.resource_updater.requests.get", lambda *args, **kwargs: _Resp())
    updater = ResourceUpdater()
    assert updater.semantic_search_papers("demo") == []


def test_resource_updater_update_resources_deduplicate(monkeypatch):
    from vulcan.services.resource_updater import ResourceUpdater

    updater = ResourceUpdater()
    monkeypatch.setattr(
        updater,
        "semantic_search_papers",
        lambda query: [
            {"title": "Same", "abstract": "A", "url": "u", "year": 2024, "authors": ["X"]},
            {"title": "Only-" + query, "abstract": "B", "url": "u2", "year": 2024, "authors": ["Y"]},
        ],
    )
    extracted = []
    monkeypatch.setattr(updater, "extract_resources", lambda paper: {"dataset_url": "d", "model_url": "m"})
    monkeypatch.setattr(
        updater,
        "validate_and_standardize",
        lambda resource: extracted.append(resource) or True,
    )

    updater.update_resources()
    # Two queries each have one unique title plus one duplicate title: 3 unique items.
    assert len(extracted) == 3
