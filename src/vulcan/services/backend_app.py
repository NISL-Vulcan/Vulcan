"""
后端 Flask 应用加载/启动入口（安装态友好）。

说明：
- 历史实现仍在仓库根目录的 `scripts/backend_server.py`，体量很大且包含全部路由与业务逻辑。
- 为满足 `src/` 布局最佳实践，这里避免在包内直接导入仓库根目录脚本（那会依赖工作目录/源码树）。
- 采用 runpy 加载 legacy 脚本，提取其中的 Flask `app` 并由包内统一启动。

后续若要彻底“搬迁实现”，可以把 legacy 脚本内容逐步迁移到本模块/子模块中，
并最终删除对 `scripts/backend_server.py` 的依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os
import runpy
import sys


def _repo_root_from_src_layout() -> Path:
    # src/vulcan/services/backend_app.py -> services -> vulcan -> src -> repo_root
    return Path(__file__).resolve().parents[3]


def legacy_backend_script_path(repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root_from_src_layout()
    return root / "scripts" / "backend_server.py"


def load_legacy_namespace(repo_root: Path | None = None) -> Mapping[str, Any]:
    script_path = legacy_backend_script_path(repo_root=repo_root)
    if not script_path.exists():
        raise FileNotFoundError(f"cannot find legacy backend script at {script_path}")

    # 确保 legacy 脚本在“仓库根目录语义”下工作（它大量使用相对路径）。
    root = script_path.parent.parent
    prev_cwd = Path.cwd()
    try:
        os.chdir(root)
        return runpy.run_path(str(script_path), run_name="vulcan_legacy_backend")
    finally:
        os.chdir(prev_cwd)


def load_legacy_flask_app(repo_root: Path | None = None):
    ns = load_legacy_namespace(repo_root=repo_root)
    app = ns.get("app")
    if app is None:
        raise RuntimeError("legacy backend script did not define global 'app'")
    return app


def run_legacy_backend(
    *,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = True,
    use_reloader: bool = False,
    repo_root: Path | None = None,
) -> None:
    # 复用 legacy 的“实时输出”行为：尽可能禁用缓冲。
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    app = load_legacy_flask_app(repo_root=repo_root)
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)

