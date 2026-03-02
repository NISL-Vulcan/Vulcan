"""Unit tests for vulcan.datacollection.data_collector."""

from pathlib import Path

from vulcan.datacollection.data_collector import DataCollector


def test_collect_sample_data_creates_output(tmp_path, monkeypatch):
    monkeypatch.setattr("vulcan.datacollection.data_collector.time.sleep", lambda *_: None)
    collector = DataCollector("unit-test")
    collector.data_dir = tmp_path
    collector.data_dir.mkdir(parents=True, exist_ok=True)

    result = collector.collect_sample_data()

    assert result["success"] is True
    assert result["stats"]["total_items"] == 380
    assert Path(result["output_file"]).exists()


def test_collect_web_data_success(monkeypatch):
    monkeypatch.setattr("vulcan.datacollection.data_collector.time.sleep", lambda *_: None)
    collector = DataCollector("web")
    result = collector.collect_web_data(["https://a.example", "https://b.example"])

    assert result["success"] is True
    assert result["total_urls"] == 2
    assert result["successful_urls"] == 2
    assert len(result["collected_data"]) == 2


def test_collect_api_data_success(monkeypatch):
    monkeypatch.setattr("vulcan.datacollection.data_collector.time.sleep", lambda *_: None)
    collector = DataCollector("api")
    result = collector.collect_api_data(["/endpoint-1"])

    assert result["success"] is True
    assert result["total_apis"] == 1
    assert result["successful_apis"] == 1
    assert result["collected_data"][0]["endpoint"] == "/endpoint-1"


def test_process_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr("vulcan.datacollection.data_collector.time.sleep", lambda *_: None)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")

    collector = DataCollector("files")
    collector.project_root = tmp_path
    result = collector.process_existing_files("*.txt")

    assert result["success"] is True
    assert result["total_files"] == 2
    assert result["successful_files"] == 2
