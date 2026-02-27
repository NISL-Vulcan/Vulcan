#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄封装：数据收集任务入口。

核心实现位于 `src/vulcan/datacollection/data_collector.py`。
"""

from __future__ import annotations

try:
    from vulcan.datacollection.data_collector import main
except ModuleNotFoundError as e:
    raise SystemExit(
        "缺少数据收集依赖，请先安装完整依赖后再运行该脚本（例如：pip install -e .）。\n"
        f"原始错误: {e}"
    ) from e


if __name__ == "__main__":
    main()

