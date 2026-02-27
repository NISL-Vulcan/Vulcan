#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄封装：数据集优化 API 服务入口。

核心实现位于 `src/vulcan/services/dataset_optimization_api_app.py`。
"""

from __future__ import annotations

from vulcan.services.dataset_optimization_api_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)

