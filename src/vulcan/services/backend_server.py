"""
后端服务启动封装。

目前仍复用 `scripts/backend_server.py` 中的实现，通过函数封装为可导入的服务接口。
后续如有需要，可以逐步将真正的实现迁移到本模块中。
"""

from __future__ import annotations

from typing import Any

from scripts import backend_server as _backend_server


def run_backend(*args: Any, **kwargs: Any) -> Any:
    """
    启动后端服务。

    优先调用 `scripts/backend_server.py` 中的 `main` 或 `start_backend` 函数，
    以保持与现有脚本行为一致。
    """
    if hasattr(_backend_server, "main"):
        return _backend_server.main(*args, **kwargs)
    if hasattr(_backend_server, "start_backend"):
        return _backend_server.start_backend(*args, **kwargs)
    raise RuntimeError("backend_server script has no 'main' or 'start_backend' entrypoint")

