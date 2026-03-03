from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import Dataset


class CocaJSONL(Dataset):
    """
    Generic JSONL dataset for records converted from Coca.

    Expected record keys:
      - sample_id
      - target (or label)
      - code/statements/graph/meta (optional but recommended)
    """

    def __init__(self, split: str, root: str, preprocess_format, args: dict[str, Any] | None = None):
        assert split in ["train", "val", "test"]
        self.n_classes = 2
        self.preprocess = preprocess_format
        self.root = Path(root)
        self.split = split
        args = args or {}
        self.args = SimpleNamespace(**args)
        self.samples = self._load_records()

    def _resolve_split_file(self) -> Path:
        if self.split == "train":
            configured = getattr(self.args, "train_data_file", None)
        elif self.split == "val":
            configured = getattr(self.args, "eval_data_file", None)
        else:
            configured = getattr(self.args, "test_data_file", None)

        if configured:
            return Path(configured)

        data_dir = getattr(self.args, "data_dir", None)
        if data_dir:
            return Path(data_dir) / f"{self.split}.jsonl"

        return self.root / f"{self.split}.jsonl"

    def _load_records(self) -> list[dict[str, Any]]:
        split_file = self._resolve_split_file()
        if not split_file.exists():
            raise FileNotFoundError(f"CocaJSONL split file not found: {split_file}")

        samples: list[dict[str, Any]] = []
        with split_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {split_file}:{line_idx}") from e
                if not isinstance(obj, dict):
                    continue
                samples.append(obj)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        label_value = sample.get("target", sample.get("label", 0))
        label = torch.tensor(int(label_value), dtype=torch.long)
        if self.preprocess:
            sample, label = self.preprocess(sample, label)
        return sample, label

