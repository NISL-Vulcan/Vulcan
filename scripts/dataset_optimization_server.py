#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄封装：数据集优化服务入口（legacy server wrapper）。

核心实现位于 `src/vulcan/services/dataset_optimization_server_app.py`。
"""

from __future__ import annotations

from vulcan.services.dataset_optimization_server_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

