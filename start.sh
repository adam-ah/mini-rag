#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python3}"

if ! "$PY" -c 'import convert; exit(0 if convert.check_deps() else 1)' 2>/dev/null; then
  echo "Missing dependencies. Please run: $PY -m pip install -r requirements.txt"
  exit 1
fi

echo "Syncing input/ → output/ (convert new/changed, prune removed)…"
"$PY" convert.py

exec "$PY" app.py
