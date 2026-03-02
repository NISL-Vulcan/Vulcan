#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper: data collection API service entrypoint.

Core implementation is in `src/vulcan/services/data_collection_api_app.py`.
"""

from __future__ import annotations

from vulcan.services.data_collection_api_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)

