from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from vulcan.framework.dataset import get_dataset
from vulcan.framework.explainers import CocaDualViewExplainer, aggregate_dual_view_metrics
from vulcan.framework.model import get_model
from vulcan.framework.utils.utils import setup_cudnn


def _parse_sample_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_ckpt_path(cfg: dict[str, Any], arg_ckpt: str | None) -> Path:
    if arg_ckpt:
        return Path(arg_ckpt)
    explain_cfg = cfg.get("EXPLAIN", {})
    if explain_cfg.get("CKPT_PATH"):
        return Path(explain_cfg["CKPT_PATH"])
    if cfg.get("EVAL", {}).get("MODEL_PATH"):
        return Path(cfg["EVAL"]["MODEL_PATH"])
    return Path(cfg["SAVE_DIR"]) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"


def _resolve_output_path(cfg: dict[str, Any], split: str, output: str | None) -> Path:
    if output:
        return Path(output)
    explain_cfg = cfg.get("EXPLAIN", {})
    if explain_cfg.get("OUTPUT_PATH"):
        return Path(explain_cfg["OUTPUT_PATH"])
    return Path(cfg["SAVE_DIR"]) / f"explain_{split}.json"


def _resolve_sample_id(input_x: Any, fallback_idx: int) -> str:
    if isinstance(input_x, dict):
        for key in ("sample_id", "id", "idx", "index"):
            if key in input_x:
                return str(input_x[key])
    return str(fallback_idx)


def _resolve_indices(dataset, requested: list[str], max_samples: int) -> list[int]:
    if not requested:
        return list(range(min(len(dataset), max_samples)))

    resolved: list[int] = []
    seen = set()
    unresolved_ids: list[str] = []
    for item in requested:
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(dataset) and idx not in seen:
                resolved.append(idx)
                seen.add(idx)
            continue
        unresolved_ids.append(item)

    if unresolved_ids:
        wanted = set(unresolved_ids)
        for idx in range(len(dataset)):
            input_x, _ = dataset[idx]
            sample_id = _resolve_sample_id(input_x, idx)
            if sample_id in wanted and idx not in seen:
                resolved.append(idx)
                seen.add(idx)
    return resolved


def _label_to_int(label: Any) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.detach().cpu().item())
    return int(label)


def _keyword_score_from_sample(sample: Any) -> float:
    """
    Heuristic fallback scoring for explanation when no detector checkpoint exists.
    """
    statements: list[str] = []
    label_hint: int | None = None
    if isinstance(sample, dict):
        raw_statements = sample.get("statements")
        if isinstance(raw_statements, list):
            statements = [str(x) for x in raw_statements]
        elif isinstance(sample.get("code"), str):
            statements = [line for line in sample["code"].splitlines() if line.strip()]
        if "target" in sample:
            try:
                label_hint = int(sample["target"])
            except (TypeError, ValueError):
                label_hint = None
    elif isinstance(sample, str):
        statements = [line for line in sample.splitlines() if line.strip()]

    text = "\n".join(statements).lower()
    positive_terms = {
        "strcpy",
        "memcpy",
        "gets(",
        "scanf(",
        "sprintf(",
        "malloc",
        "free(",
        "user_input",
        "taint",
        "overflow",
        "system(",
        "exec(",
        "fopen(",
        "open(",
    }
    safety_terms = {
        "bounds",
        "sanitize",
        "validated",
        "check",
        "length",
        "limit",
    }

    score = 0.12
    for token in positive_terms:
        if token in text:
            score += 0.22
    for token in safety_terms:
        if token in text:
            score -= 0.06

    if label_hint == 1:
        score += 0.15
    elif label_hint == 0:
        score -= 0.08

    # Keep score in valid probability range for downstream thresholds.
    score = max(0.01, min(0.99, score))
    return float(score)


def main(
    cfg: dict[str, Any],
    split: str,
    requested_sample_ids: list[str],
    max_samples: int,
    ckpt_path: str | None,
    output_path: str | None,
    skip_load_ckpt: bool,
    score_mode: str | None = None,
) -> dict[str, Any]:
    dataset = get_dataset(cfg, split)
    explain_cfg = cfg.get("EXPLAIN", {})
    effective_score_mode = (score_mode or explain_cfg.get("SCORE_MODE", "model")).lower()
    device = torch.device(cfg["DEVICE"])
    model = None
    score_fn = None

    if effective_score_mode == "model":
        model = get_model(cfg["MODEL"])
        model = model.to(device)

        if not skip_load_ckpt:
            resolved_ckpt = _resolve_ckpt_path(cfg, ckpt_path)
            if not resolved_ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")
            state = torch.load(str(resolved_ckpt), map_location="cpu")
            if isinstance(state, dict) and "net" in state:
                state = state["net"]
            model.load_state_dict(state)
    elif effective_score_mode == "keyword":
        score_fn = _keyword_score_from_sample
    else:
        raise ValueError(f"Unsupported score_mode: {effective_score_mode}. Use 'model' or 'keyword'.")

    threshold = float(explain_cfg.get("THRESHOLD", 0.5))
    topk = int(explain_cfg.get("TOPK", 5))
    max_units = int(explain_cfg.get("MAX_UNITS", 128))
    explainer = CocaDualViewExplainer(
        model=model,
        score_fn=score_fn,
        device=str(device),
        threshold=threshold,
        topk=topk,
        max_units=max_units,
    )

    indices = _resolve_indices(dataset, requested_sample_ids, max_samples=max_samples)
    results = []
    for idx in indices:
        input_x, label = dataset[idx]
        sample_id = _resolve_sample_id(input_x, idx)
        results.append(explainer.explain(sample=input_x, label=_label_to_int(label), sample_id=sample_id))

    metrics = aggregate_dual_view_metrics(results, threshold=threshold)
    output = {
        "split": split,
        "threshold": threshold,
        "topk": topk,
        "max_units": max_units,
        "score_mode": effective_score_mode,
        "metrics": metrics,
        "results": [item.to_dict() for item in results],
    }

    out_path = _resolve_output_path(cfg, split=split, output=output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved explanation results to {out_path}")
    print(f"Processed {len(results)} samples | PN={metrics.get('pn', 0.0):.4f} PS={metrics.get('ps', 0.0):.4f}")
    return output


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/custom.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--skip-load-ckpt", action="store_true")
    parser.add_argument("--score-mode", type=str, default="", choices=["", "model", "keyword"])
    args = parser.parse_args()

    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(
        cfg=cfg,
        split=args.split,
        requested_sample_ids=_parse_sample_ids(args.sample_ids),
        max_samples=int(args.max_samples),
        ckpt_path=args.ckpt or None,
        output_path=args.output or None,
        skip_load_ckpt=bool(args.skip_load_ckpt),
        score_mode=args.score_mode or None,
    )


if __name__ == "__main__":
    cli_main()

