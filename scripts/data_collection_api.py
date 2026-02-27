#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄封装：数据收集 API 服务入口。

核心实现位于 `src/vulcan/services/data_collection_api_app.py`。
"""

from __future__ import annotations

from vulcan.services.data_collection_api_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)

