#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH=src:${PYTHONPATH}

echo "==> Running pytest"
pytest

