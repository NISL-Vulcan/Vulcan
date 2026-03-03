"""Unit tests for vulcan.services.apis."""

import os
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
            return "10 minutes 0 seconds"

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
    assert "best ratio" in status["status_description"]
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
            return "1 minute 0 seconds"

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
    # Two queries each have one unique title plus one shared title, yielding 3 unique entries in total
    assert len(extracted) == 3


def test_dynamic_ratio_find_dirs_and_files(tmp_path, monkeypatch):
    from vulcan.services import auto_update_and_dynamic_ratio as ratio_mod

    dataset_dir = tmp_path / "dataset"
    configs_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    configs_dir.mkdir()
    (dataset_dir / "vuln_samples.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (dataset_dir / "non-vulnerables.jsonl").write_text('{"a":0}\n', encoding="utf-8")
    (configs_dir / "regvd.yaml").write_text("x: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    tuner = ratio_mod.DynamicRatioTuner()
    assert tuner.dataset_dir.endswith("dataset")
    assert tuner.configs_dir.endswith("configs")

    vuln_path, nonvuln_path = tuner._find_data_files()
    assert os.path.exists(vuln_path)
    assert os.path.exists(nonvuln_path)
    assert tuner._find_config_file().endswith("regvd.yaml")


def test_dynamic_ratio_sample_and_generate_trainset(tmp_path, monkeypatch):
    from vulcan.services import auto_update_and_dynamic_ratio as ratio_mod

    dataset_dir = tmp_path / "dataset"
    configs_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    configs_dir.mkdir()
    vuln_path = dataset_dir / "vuln.jsonl"
    nonvuln_path = dataset_dir / "non-vulnerables.jsonl"
    vuln_path.write_text("".join(f'{{"i":{i}}}\n' for i in range(12)), encoding="utf-8")
    nonvuln_path.write_text("".join(f'{{"j":{i}}}\n' for i in range(24)), encoding="utf-8")

    tuner = ratio_mod.DynamicRatioTuner.__new__(ratio_mod.DynamicRatioTuner)
    tuner.dataset_dir = str(dataset_dir)
    tuner.configs_dir = str(configs_dir)
    tuner._used_vuln_indices = set()
    tuner._vuln_shuffle_order = list(range(12))
    tuner._last_sample_content = None
    tuner._last_sample_hash = None

    monkeypatch.setattr(ratio_mod.random, "randint", lambda a, b: 6)
    monkeypatch.setattr(ratio_mod.random, "shuffle", lambda x: None)
    out_path = dataset_dir / "train_dynamic.jsonl"
    tuner.sample_and_generate_trainset(str(vuln_path), str(nonvuln_path), 0.5, str(out_path))
    lines = out_path.read_text(encoding="utf-8").splitlines()
    # vuln=6, nonvuln=12 (ratio=0.5)
    assert len(lines) == 18


def test_dynamic_ratio_evaluate_ratio_with_mocked_process(tmp_path, monkeypatch):
    from vulcan.services import auto_update_and_dynamic_ratio as ratio_mod

    dataset_dir = tmp_path / "dataset"
    configs_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    configs_dir.mkdir()
    vuln_path = dataset_dir / "vuln.jsonl"
    nonvuln_path = dataset_dir / "non-vulnerables.jsonl"
    vuln_path.write_text('{"a":1}\n', encoding="utf-8")
    nonvuln_path.write_text('{"a":0}\n', encoding="utf-8")
    cfg_path = configs_dir / "regvd.yaml"
    cfg_path.write_text("DATASET:\n  PARAMS:\n    args:\n      train_data_file: x\n", encoding="utf-8")
    train_py = tmp_path / "train.py"
    train_py.write_text("print('noop')\n", encoding="utf-8")

    tuner = ratio_mod.DynamicRatioTuner.__new__(ratio_mod.DynamicRatioTuner)
    tuner.dataset_dir = str(dataset_dir)
    tuner.configs_dir = str(configs_dir)
    tuner._used_vuln_indices = set()
    tuner._vuln_shuffle_order = []
    tuner._last_sample_content = None
    tuner._last_sample_hash = None

    monkeypatch.setattr(tuner, "_find_data_files", lambda: (str(vuln_path), str(nonvuln_path)))
    monkeypatch.setattr(tuner, "_find_config_file", lambda: str(cfg_path))
    monkeypatch.setattr(tuner, "_find_train_script", lambda: str(train_py))

    def _fake_sample(vp, nvp, ratio, outp):
        with open(outp, "w", encoding="utf-8") as f:
            f.write('{"x":1}\n')

    monkeypatch.setattr(tuner, "sample_and_generate_trainset", _fake_sample)

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            self._lines = ["F1-score: 0.77\n", "Accuracy: 0.81\n"]
            self.stdout = self

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def poll(self):
            return None if self._lines else 0

        def wait(self):
            return 0

    monkeypatch.setattr(ratio_mod.subprocess, "Popen", _FakePopen)
    f1, metrics = tuner.evaluate_ratio(0.3)
    assert f1 == 0.77
    assert metrics["Accuracy"] == 0.81
    assert not any(p.name.startswith("tmp_dynamic_") for p in configs_dir.iterdir())


def test_dynamic_ratio_get_best_ratio(monkeypatch):
    from vulcan.services import auto_update_and_dynamic_ratio as ratio_mod

    tuner = ratio_mod.DynamicRatioTuner.__new__(ratio_mod.DynamicRatioTuner)
    score_fn = lambda r: 1.0 - abs(r - 0.42)
    monkeypatch.setattr(tuner, "evaluate_ratio", lambda r: (score_fn(r), {"F1": score_fn(r)}))
    best = tuner.get_best_ratio(max_iter=4)
    assert 0.1 <= best <= 0.9


def test_dataset_optimization_api_run_dataset_optimization_success(monkeypatch, tmp_path):
    import vulcan.services.dataset_optimization_api_app as opt_api

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(opt_api.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(opt_api.os.path, "exists", lambda p: True)
    monkeypatch.setattr(opt_api.os.path, "getsize", lambda p: 123)

    job = opt_api.OptimizationJob(job_id="job-success")
    job.log_file_path = str(tmp_path / "opt_success.log")

    opt_api.run_dataset_optimization(job)
    assert job.status == "completed"
    assert job.progress == 100
    # metrics 中应包含关键字段
    assert "best_ratio" in job.metrics
    assert "best_f1_score" in job.metrics
    assert "total_iterations" in job.metrics
    assert "duration" in job.metrics


def test_dataset_optimization_api_run_dataset_optimization_missing_script(monkeypatch, tmp_path):
    import vulcan.services.dataset_optimization_api_app as opt_api

    def _fake_exists(path):
        if path.endswith("auto_update_and_dynamic_ratio.py"):
            return False
        return True

    monkeypatch.setattr(opt_api.os.path, "exists", _fake_exists)

    job = opt_api.OptimizationJob(job_id="job-missing")
    job.log_file_path = str(tmp_path / "opt_missing.log")

    opt_api.run_dataset_optimization(job)
    assert job.status == "failed"
    assert job.progress == 0


def test_dataset_optimization_server_run_dataset_optimization_success(monkeypatch, tmp_path):
    import vulcan.services.dataset_optimization_server_app as opt_srv

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(opt_srv.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(opt_srv.os.path, "exists", lambda p: True)
    monkeypatch.setattr(opt_srv.os.path, "getsize", lambda p: 123)

    job = opt_srv.OptimizationJob(job_id="srv-job")
    job.log_file_path = str(tmp_path / "srv_success.log")

    opt_srv.run_dataset_optimization(job)
    assert job.status == "completed"
    assert job.progress == 100
