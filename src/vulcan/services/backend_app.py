"""
EN Flask EN/EN(EN).

EN:
- EN `scripts/backend_server.py`,EN.
- EN `src/` EN,EN(EN/EN).
- EN runpy EN legacy EN,EN Flask `app` EN.

EN"EN",EN legacy EN/EN,
EN `scripts/backend_server.py` EN.
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

    # EN legacy EN"EN"EN(EN).
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
    # EN legacy EN"EN"EN:EN.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    app = load_legacy_flask_app(repo_root=repo_root)
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)

