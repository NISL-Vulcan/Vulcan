"""
EN.

EN `vulcan.services.backend_server_app` EN.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys


def run_backend() -> None:
    # EN:EN,EN configs/,tools/ EN.
    repo_root = Path(__file__).resolve().parents[3]
    os.chdir(repo_root)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    from .backend_server_app import app

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

