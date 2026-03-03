#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vulcan.framework.adapters.coca import convert_coca_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Coca dataset files to Vulcan JSONL format.")
    parser.add_argument("--input-dir", required=True, help="Coca dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted JSONL files.")
    parser.add_argument(
        "--detector-hint",
        default="auto",
        help="Detector hint saved in converted metadata (e.g. reveal/devign/deepwukong).",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Embed original Coca sample JSON into each converted record.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = convert_coca_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        detector_hint=args.detector_hint,
        include_raw=bool(args.include_raw),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

