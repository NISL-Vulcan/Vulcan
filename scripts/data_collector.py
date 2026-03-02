#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper: data collection task entrypoint.

Core implementation is in `src/vulcan/datacollection/data_collector.py`.
"""

from __future__ import annotations

try:
    from vulcan.datacollection.data_collector import main
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing data collection dependencies. Install full dependencies first "
        "(e.g. `pip install -e .`) before running this script.\n"
        f"Original error: {e}"
    ) from e


if __name__ == "__main__":
    main()

