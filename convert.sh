#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python3}"

command -v pandoc >/dev/null 2>&1 || echo "WARN: pandoc not found — .docx/.html conversion will be skipped"
if ! "$PY" -c 'import convert; exit(0 if convert.check_deps() else 1)' 2>/dev/null; then
  echo "Missing dependencies. Please run: $PY -m pip install -r requirements.txt"
  exit 1
fi

"$PY" convert.py "$@"
echo "Done. Restart the app (./start.sh) — the corpus rebuilds in memory on launch."
