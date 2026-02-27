#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄封装：后端服务启动入口。

核心实现位于 `src/vulcan/services/backend_server_app.py`，建议通过 console script `vulcan-backend` 启动。
"""

from __future__ import annotations

from vulcan.services.backend_server import run_backend


if __name__ == "__main__":
    run_backend()

