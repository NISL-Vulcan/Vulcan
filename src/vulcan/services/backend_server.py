"""
后端服务启动封装。

通过 `vulcan.services.backend_app` 以安装态方式加载/启动后端应用。
"""

from __future__ import annotations

from .backend_app import run_legacy_backend


def run_backend() -> None:
    run_legacy_backend()

