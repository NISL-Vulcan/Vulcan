import json

import torch

from vulcan.framework.adapters.coca.compat import get_word2vec_vocab, make_word2vec_compatible
from vulcan.framework.adapters.coca.convert import convert_coca_directory
from vulcan.framework.adapters.coca.transformations import SemanticPreservingTransformation
from vulcan.framework.config_templates import ConfigTemplateManager
from vulcan.framework.dataset import get_dataset
from vulcan.framework.explainers.base import ExplanationResult
from vulcan.framework.explainers.coca_dual_view import CocaDualViewExplainer
from vulcan.framework.explainers.metrics import aggregate_dual_view_metrics


def test_word2vec_compat_helpers():
    assert make_word2vec_compatible({"size": 128, "window": 10})["vector_size"] == 128

    class _Wv4:
        key_to_index = {"a": 0, "b": 1}

    class _M4:
        wv = _Wv4()

    class _Wv3:
        vocab = {"x": object(), "y": object()}

    class _M3:
        wv = _Wv3()

    assert set(get_word2vec_vocab(_M4())) == {"a", "b"}
    assert set(get_word2vec_vocab(_M3())) == {"x", "y"}


def test_convert_coca_directory_and_dataset_loading(tmp_path):
    input_root = tmp_path / "coca_raw"
    function_dir = input_root / "function"
    slice_dir = input_root / "slice"
    function_dir.mkdir(parents=True)
    slice_dir.mkdir(parents=True)

    # Minimal function-level Coca sample
    function_train_vul = [
        {
            "id": "f-1",
            "target": 1,
            "func": "int foo(){\nreturn 1;\n}",
            "nodes": [],
            "cfgEdges": ["[0,1]"],
            "ddgEdges": [],
        }
    ]
    function_train_nor = [
        {
            "id": "f-0",
            "target": 0,
            "func": "int bar(){\nreturn 0;\n}",
            "nodes": [],
            "cfgEdges": ["[0,1]"],
            "ddgEdges": [],
        }
    ]
    (function_dir / "train_vuls.json").write_text(json.dumps(function_train_vul), encoding="utf-8")
    (function_dir / "train_nors.json").write_text(json.dumps(function_train_nor), encoding="utf-8")
    (function_dir / "val_vuls.json").write_text("[]", encoding="utf-8")
    (function_dir / "val_nors.json").write_text("[]", encoding="utf-8")
    (function_dir / "test_vuls.json").write_text("[]", encoding="utf-8")
    (function_dir / "test_nors.json").write_text("[]", encoding="utf-8")

    # Minimal slice-level Coca sample
    slice_train_vul = [
        {
            "id": "s-1",
            "target": 1,
            "line-contents": ["a = b + 1;", "if (a > 0) return a;"],
            "data-dependences": ["[0,1]"],
            "control-dependences": [],
        }
    ]
    (slice_dir / "train_vuls.json").write_text(json.dumps(slice_train_vul), encoding="utf-8")
    (slice_dir / "train_nors.json").write_text("[]", encoding="utf-8")
    (slice_dir / "val_vuls.json").write_text("[]", encoding="utf-8")
    (slice_dir / "val_nors.json").write_text("[]", encoding="utf-8")
    (slice_dir / "test_vuls.json").write_text("[]", encoding="utf-8")
    (slice_dir / "test_nors.json").write_text("[]", encoding="utf-8")

    out_dir = tmp_path / "converted"
    manifest = convert_coca_directory(input_root, out_dir, detector_hint="devign", include_raw=False)
    assert manifest["splits"]["train"]["total"] == 3
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    config = {
        "DATASET": {
            "NAME": "CocaJSONL",
            "ROOT": str(out_dir),
            "dataloader": "coca",
            "PARAMS": {
                "args": {
                    "train_data_file": str(out_dir / "train.jsonl"),
                    "eval_data_file": str(out_dir / "val.jsonl"),
                    "test_data_file": str(out_dir / "test.jsonl"),
                }
            },
            "PREPROCESS": {"ENABLE": False, "COMPOSE": []},
        },
        "TRAIN": {"INPUT_SIZE": 1},
        "EVAL": {"INPUT_SIZE": 1},
    }
    dataset = get_dataset(config, "train")
    assert len(dataset) == 3
    input_x, label = dataset[0]
    assert isinstance(input_x, dict)
    assert "graph" in input_x
    assert torch.is_tensor(label)


def test_semantic_preserving_transformation_runs():
    transformer = SemanticPreservingTransformation(seed=7)
    code = "int sum(int a, int b) { int c = a + b; return c; }"
    transformed, name = transformer.transform_code(code)
    assert isinstance(transformed, str)
    assert name in {"VariableRenamingTransformation", "SyntacticNoisingTransformation", None}


def test_config_templates_include_explain_section():
    manager = ConfigTemplateManager()
    for template_name in manager.list_templates():
        template = manager.get_template(template_name)
        assert template is not None
        assert "EXPLAIN" in template.template
        explain = template.template["EXPLAIN"]
        assert explain["METHOD"] == "CocaDualView"
        assert "TOPK" in explain


def test_dual_view_explainer_and_metrics():
    sample = {
        "sample_id": "x-1",
        "statements": ["safe_call();", "sink_call(user_input);", "return 0;"],
        "code": "safe_call();\nsink_call(user_input);\nreturn 0;",
        "graph": {
            "num_nodes": 3,
            "node_statements": ["safe_call();", "sink_call(user_input);", "return 0;"],
            "edge_index": [[0, 1], [1, 2]],
            "edge_types": ["cfg", "cfg"],
        },
    }

    def _score_fn(inp):
        statements = inp.get("statements", [])
        score = 0.1
        for stmt in statements:
            if "sink_call" in stmt:
                score += 0.7
            if "user_input" in stmt:
                score += 0.2
        return min(score, 0.99)

    explainer = CocaDualViewExplainer(model=None, score_fn=_score_fn, topk=1, threshold=0.5)
    result = explainer.explain(sample=sample, label=1, sample_id="x-1")
    assert result.sample_id == "x-1"
    assert len(result.selected_units) == 1
    assert result.base_score >= 0.5

    metrics = aggregate_dual_view_metrics([result], threshold=0.5)
    assert metrics["count"] == 1
    assert "pn" in metrics and "ps" in metrics and "fns" in metrics

    # Ensure metric aggregation handles multiple records.
    metrics2 = aggregate_dual_view_metrics(
        [
            result,
            ExplanationResult(
                sample_id="x-2",
                label=0,
                base_score=0.2,
                factual_score=0.1,
                counterfactual_score=0.3,
                selected_units=[0],
                unit_scores=[0.1],
            ),
        ],
        threshold=0.5,
    )
    assert metrics2["count"] == 2

