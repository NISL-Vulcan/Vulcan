from argparse import Namespace

import torch

import vulcan.cli.explain as explain_module


class _DummyDataset:
    def __init__(self):
        self.items = [
            (
                {
                    "sample_id": "a-1",
                    "statements": ["safe();", "sink(x);"],
                    "code": "safe();\nsink(x);",
                    "graph": {
                        "num_nodes": 2,
                        "node_statements": ["safe();", "sink(x);"],
                        "edge_index": [[0, 1]],
                        "edge_types": ["cfg"],
                    },
                },
                torch.tensor(1),
            ),
            (
                {
                    "sample_id": "a-2",
                    "statements": ["safe();", "ret();"],
                    "code": "safe();\nret();",
                    "graph": {
                        "num_nodes": 2,
                        "node_statements": ["safe();", "ret();"],
                        "edge_index": [[0, 1]],
                        "edge_types": ["cfg"],
                    },
                },
                torch.tensor(0),
            ),
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_explain_main_writes_output(tmp_path, monkeypatch):
    cfg = {
        "DEVICE": "cpu",
        "SAVE_DIR": str(tmp_path),
        "MODEL": {"NAME": "Dummy", "BACKBONE": "", "PARAMS": {}},
        "DATASET": {"NAME": "Dummy", "ROOT": str(tmp_path), "PREPROCESS": {"ENABLE": False, "COMPOSE": []}},
        "EVAL": {"MODEL_PATH": ""},
        "EXPLAIN": {"TOPK": 1, "THRESHOLD": 0.5, "SCORE_MODE": "keyword"},
    }
    monkeypatch.setattr(explain_module, "get_dataset", lambda cfg_, split: _DummyDataset())

    out = explain_module.main(
        cfg=cfg,
        split="val",
        requested_sample_ids=[],
        max_samples=2,
        ckpt_path=None,
        output_path=str(tmp_path / "explain.json"),
        skip_load_ckpt=True,
        score_mode="keyword",
    )
    assert out["metrics"]["count"] == 2
    assert out["score_mode"] == "keyword"
    assert (tmp_path / "explain.json").exists()


def test_explain_cli_main_calls_main(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "DEVICE: cpu",
                f"SAVE_DIR: '{tmp_path.as_posix()}'",
                "MODEL:",
                "  NAME: Dummy",
                "  BACKBONE: ''",
                "  PARAMS: {}",
                "DATASET:",
                "  NAME: Dummy",
                f"  ROOT: '{tmp_path.as_posix()}'",
                "  PREPROCESS:",
                "    ENABLE: false",
                "    COMPOSE: []",
                "EVAL:",
                "  MODEL_PATH: ''",
            ]
        ),
        encoding="utf-8",
    )
    called = {}
    monkeypatch.setattr(
        explain_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            cfg=str(cfg_path),
            split="val",
            sample_ids="",
            max_samples=2,
            ckpt="",
            output="",
            skip_load_ckpt=True,
            score_mode="keyword",
        ),
    )
    monkeypatch.setattr(explain_module, "setup_cudnn", lambda: called.setdefault("cudnn", True))
    monkeypatch.setattr(explain_module, "main", lambda **kwargs: called.setdefault("kwargs", kwargs))

    explain_module.cli_main()
    assert called["cudnn"] is True
    assert called["kwargs"]["split"] == "val"
    assert called["kwargs"]["skip_load_ckpt"] is True
    assert called["kwargs"]["score_mode"] == "keyword"

