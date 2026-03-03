from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _as_json_obj(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_sample_id(sample: dict[str, Any], fallback: str) -> str:
    for key in ("sample_id", "id", "idx", "index", "hash", "hash_id"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return fallback


def _extract_statements_from_line_nodes(line_nodes: Iterable[Any]) -> list[str]:
    statements: list[str] = []
    for node_info in line_nodes:
        node_obj = _as_json_obj(node_info)
        if not isinstance(node_obj, dict):
            continue
        # Coca line node format usually stores token text in:
        # {"contents": [[..., "<statement text>"], ...]}
        contents = node_obj.get("contents")
        if isinstance(contents, list) and contents:
            first = contents[0]
            if isinstance(first, list) and len(first) >= 2:
                statements.append(str(first[1]))
                continue
        # Fallback for other possible variants.
        text = node_obj.get("code") or node_obj.get("text") or node_obj.get("statement")
        if text:
            statements.append(str(text))
    return statements


def _extract_edges(sample: dict[str, Any], edge_keys: list[str]) -> tuple[list[list[int]], list[str]]:
    edge_index: list[list[int]] = []
    edge_types: list[str] = []
    for edge_key in edge_keys:
        raw_edges = sample.get(edge_key, []) or []
        for edge in raw_edges:
            parsed = _as_json_obj(edge)
            if isinstance(parsed, list) and len(parsed) >= 2:
                src, dst = parsed[0], parsed[1]
                try:
                    edge_index.append([int(src), int(dst)])
                    edge_types.append(edge_key)
                except (TypeError, ValueError):
                    continue
    return edge_index, edge_types


def _convert_function_sample(
    sample: dict[str, Any],
    label: int,
    sample_id: str,
    detector_hint: str,
    include_raw: bool,
) -> dict[str, Any]:
    statements: list[str] = []
    if isinstance(sample.get("code"), str):
        statements = [line for line in sample["code"].splitlines() if line.strip()]
    elif isinstance(sample.get("func"), str):
        statements = [line for line in sample["func"].splitlines() if line.strip()]
    if not statements:
        statements = _extract_statements_from_line_nodes(sample.get("nodes", []))

    edge_index, edge_types = _extract_edges(sample, edge_keys=["cfgEdges", "ddgEdges"])
    target = int(sample.get("target", label))
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "target": target,
        "sample_type": "function",
        "detector_hint": detector_hint,
        "code": "\n".join(statements),
        "statements": statements,
        "graph": {
            "num_nodes": len(statements),
            "node_statements": statements,
            "edge_index": edge_index,
            "edge_types": edge_types,
        },
        "meta": {
            "source_keys": sorted(sample.keys()),
            "source_label": label,
        },
    }
    if include_raw:
        record["raw"] = sample
    return record


def _convert_slice_sample(
    sample: dict[str, Any],
    label: int,
    sample_id: str,
    detector_hint: str,
    include_raw: bool,
) -> dict[str, Any]:
    statements = [str(x) for x in sample.get("line-contents", []) if str(x).strip()]
    if not statements:
        statements = _extract_statements_from_line_nodes(sample.get("line-nodes", []))

    edge_index, edge_types = _extract_edges(
        sample,
        edge_keys=["data-dependences", "control-dependences"],
    )
    target = int(sample.get("target", label))
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "target": target,
        "sample_type": "slice",
        "detector_hint": detector_hint,
        "code": "\n".join(statements),
        "statements": statements,
        "graph": {
            "num_nodes": len(statements),
            "node_statements": statements,
            "edge_index": edge_index,
            "edge_types": edge_types,
        },
        "meta": {
            "source_keys": sorted(sample.keys()),
            "source_label": label,
        },
    }
    if include_raw:
        record["raw"] = sample
    return record


def _convert_sample(
    sample: dict[str, Any],
    sample_type: str,
    label: int,
    fallback_id: str,
    detector_hint: str,
    include_raw: bool,
) -> dict[str, Any]:
    sample_id = _extract_sample_id(sample, fallback=fallback_id)
    if sample_type == "function":
        return _convert_function_sample(
            sample=sample,
            label=label,
            sample_id=sample_id,
            detector_hint=detector_hint,
            include_raw=include_raw,
        )
    return _convert_slice_sample(
        sample=sample,
        label=label,
        sample_id=sample_id,
        detector_hint=detector_hint,
        include_raw=include_raw,
    )


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def convert_coca_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    detector_hint: str = "auto",
    include_raw: bool = False,
) -> dict[str, Any]:
    """
    Convert Coca train/val/test json files into Vulcan-friendly JSONL files.

    Input layout (either or both):
      - <input_dir>/function/{train,val,test}_{vuls,nors}.json
      - <input_dir>/slice/{train,val,test}_{vuls,nors}.json
    """
    src_root = Path(input_dir)
    dst_root = Path(output_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "val", "test")
    label_files = (("vuls", 1), ("nors", 0))
    split_records: dict[str, list[dict[str, Any]]] = {k: [] for k in split_names}
    counts: dict[str, dict[str, int]] = {
        split: {"total": 0, "positive": 0, "negative": 0} for split in split_names
    }

    for sample_type in ("function", "slice"):
        sample_dir = src_root / sample_type
        if not sample_dir.exists():
            continue

        for split in split_names:
            for suffix, label in label_files:
                file_path = sample_dir / f"{split}_{suffix}.json"
                if not file_path.exists():
                    continue
                samples = _load_samples(file_path)
                for idx, sample in enumerate(samples):
                    fallback_id = f"{sample_type}-{split}-{suffix}-{idx}"
                    record = _convert_sample(
                        sample=sample,
                        sample_type=sample_type,
                        label=label,
                        fallback_id=fallback_id,
                        detector_hint=detector_hint,
                        include_raw=include_raw,
                    )
                    split_records[split].append(record)
                    counts[split]["total"] += 1
                    if int(record["target"]) == 1:
                        counts[split]["positive"] += 1
                    else:
                        counts[split]["negative"] += 1

    for split in split_names:
        out_file = dst_root / f"{split}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for record in split_records[split]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "input_dir": str(src_root),
        "output_dir": str(dst_root),
        "detector_hint": detector_hint,
        "include_raw": include_raw,
        "splits": counts,
        "schema_version": "coca-vulcan-jsonl-v1",
    }
    with (dst_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest

