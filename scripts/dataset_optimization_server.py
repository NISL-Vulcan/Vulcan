#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper: dataset optimization service entrypoint (legacy server wrapper).

Core implementation is in `src/vulcan/services/dataset_optimization_server_app.py`.
"""

from __future__ import annotations

from vulcan.services.dataset_optimization_server_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

