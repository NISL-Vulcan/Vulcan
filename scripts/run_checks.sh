#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! python -c "import vulcan" >/dev/null 2>&1; then
  echo "==> Installing package (editable) for tests"
  python -m pip install -e . --no-deps
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
  echo "==> Installing pytest"
  python -m pip install pytest
fi

echo "==> Running pytest"
python -m pytest

