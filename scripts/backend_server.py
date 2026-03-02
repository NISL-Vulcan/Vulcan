#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper: backend service entrypoint.

Core implementation is in `src/vulcan/services/backend_server_app.py`.
Preferred startup: console script `vulcan-backend`.
"""

from __future__ import annotations

from vulcan.services.backend_server import run_backend


if __name__ == "__main__":
    run_backend()

