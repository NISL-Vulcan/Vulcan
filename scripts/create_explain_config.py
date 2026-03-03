#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _default_ckpt_path(cfg: dict[str, Any]) -> str:
    save_dir = str(cfg.get("SAVE_DIR", "output"))
    model_cfg = cfg.get("MODEL", {})
    dataset_cfg = cfg.get("DATASET", {})
    model_name = str(model_cfg.get("NAME", "Model"))
    backbone = str(model_cfg.get("BACKBONE", ""))
    dataset_name = str(dataset_cfg.get("NAME", "Dataset"))
    return str(Path(save_dir) / f"{model_name}_{backbone}_{dataset_name}.pth")


def _inject_explain_config(
    cfg: dict[str, Any],
    ckpt_path: str | None,
    output_path: str | None,
    score_mode: str,
    threshold: float,
    topk: int,
    max_units: int,
) -> dict[str, Any]:
    copied = dict(cfg)
    effective_ckpt = ckpt_path or _default_ckpt_path(copied)
    effective_output = output_path or str(Path(copied.get("SAVE_DIR", "output")) / "explain_val.json")

    eval_cfg = dict(copied.get("EVAL", {}))
    if score_mode == "model":
        eval_cfg["MODEL_PATH"] = effective_ckpt
    copied["EVAL"] = eval_cfg

    copied["EXPLAIN"] = {
        "METHOD": "CocaDualView",
        "ENABLE": True,
        "SCORE_MODE": score_mode,
        "THRESHOLD": float(threshold),
        "TOPK": int(topk),
        "MAX_UNITS": int(max_units),
        "CKPT_PATH": effective_ckpt,
        "OUTPUT_PATH": effective_output,
    }
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explain-ready config from an existing training config.")
    parser.add_argument("--input-cfg", required=True, help="Path to base training config.")
    parser.add_argument("--output-cfg", required=True, help="Path to write explain config.")
    parser.add_argument("--ckpt", default="", help="Checkpoint path for model scoring mode.")
    parser.add_argument("--output", default="", help="Explain output json path.")
    parser.add_argument("--score-mode", default="model", choices=["model", "keyword"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-units", type=int, default=128)
    args = parser.parse_args()

    in_path = Path(args.input_cfg)
    out_path = Path(args.output_cfg)
    with in_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    new_cfg = _inject_explain_config(
        cfg=cfg,
        ckpt_path=args.ckpt or None,
        output_path=args.output or None,
        score_mode=args.score_mode,
        threshold=args.threshold,
        topk=args.topk,
        max_units=args.max_units,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=False)
    print(f"Generated explain config: {out_path}")


if __name__ == "__main__":
    main()

