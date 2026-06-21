#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python3}"
rc=0
export RUN_LIVE_AI_TESTS="${RUN_LIVE_AI_TESTS:-1}"
export AI_TEST_BASE_URL="${AI_TEST_BASE_URL:-http://10.0.0.10:8080/v1}"
export AI_TEST_MODEL="${AI_TEST_MODEL:-local}"

echo "Prerequisites"
echo "  python:      $("$PY" --version 2>&1)"

if "$PY" -c 'import convert; exit(0 if convert.check_deps() else 1)' 2>/dev/null; then
  echo "  dependencies: ok"
else
  echo "  dependencies: missing — please run: $PY -m pip install -r requirements.txt"
  exit 1
fi

if [ -d tests/big_corpus ] && [ -n "$(ls -A tests/big_corpus 2>/dev/null)" ]; then
  echo "  test corpora: ok ($(find tests -name '*.md' | wc -l | tr -d ' ') .md files under tests/)"
else
  echo "  test corpora: MISSING — tests/big_corpus and tests/small_corpus ship with the repo; e2e/eval tests will SKIP"
fi

if [ "$RUN_LIVE_AI_TESTS" = "1" ]; then
  if [ -n "${AI_TEST_BASE_URL:-}" ]; then
    echo "  live AI tests: enabled (dedicated endpoint: ${AI_TEST_BASE_URL})"
  else
    echo "  live AI tests: requested but AI_TEST_BASE_URL is unset; tests will skip"
  fi
else
  echo "  live AI tests: disabled by RUN_LIVE_AI_TESTS=$RUN_LIVE_AI_TESTS"
fi

echo
echo "Code quality"
if command -v ruff >/dev/null 2>&1; then
  if ruff check . ; then
    echo "  python lint: ok (ruff)"
  else
    echo "  python lint: FAILED — fix the issues above (run 'ruff check . --fix' or address manually)"
    rc=1
  fi
else
  echo "  python lint: skipped (install ruff: $PY -m pip install ruff)"
fi

if command -v node >/dev/null 2>&1; then
  js_failed=0
  for js_file in stream.js render.js; do
    if ! node --check "$js_file" 2>/dev/null; then
      echo "  js syntax: $js_file FAILED"
      js_failed=1
    fi
  done
  if "$PY" -c "import re,sys
html=open('templates/index.html', encoding='utf-8').read()
m=re.search(r'<script>(.*?)</script>', html, re.S)
if m: open('/tmp/inline_check.js','w').write(m.group(1))" 2>/dev/null && [ -s /tmp/inline_check.js ]; then
    if ! node --check /tmp/inline_check.js 2>/dev/null; then
      echo "  js syntax: inline script in templates/index.html FAILED"
      js_failed=1
    fi
  fi
  if [ "$js_failed" -eq 0 ]; then
    echo "  js syntax: ok"
  else
    rc=1
  fi
else
  echo "  js syntax: skipped (node not installed)"
fi

echo
echo "Running tests"
if [ "$#" -gt 0 ]; then
  tests=("$@")
else
  tests=(test_backend test_convert test_corpus test_e2e test_eval test_query_rescue test_refinement test_settings test_ui_contract test_browser)
fi
SEARCH_INPUT="$PWD/tests/small_corpus" SEARCH_OUTPUT="$PWD/tests/small_corpus" \
SEARCH_SETTINGS_FILE="$PWD/tests/.test-settings.json" \
"$PY" -m unittest -v "${tests[@]}"
rc=$?

if command -v node >/dev/null 2>&1; then
  echo
  echo "Running render tests (node)"
  node test_render.js || rc=1
  node test_stream.js || rc=1
else
  echo "  node not found — skipping JS render tests"
fi
exit $rc
