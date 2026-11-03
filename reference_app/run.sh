#!/usr/bin/env bash
# Set up (first run only) and start the reference app on macOS.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.13 || command -v python3.12 || true)
if [ -z "$PY" ]; then
  echo "Python 3.12+ not found. Install it with: brew install python@3.12"; exit 1
fi
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
echo "Open http://127.0.0.1:4177 in your browser (Ctrl+C to stop)."
exec .venv/bin/python -m uvicorn live_demo.server:app --host 127.0.0.1 --port 4177
