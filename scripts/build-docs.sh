#!/usr/bin/env bash
# Rebuild the sub-task documentation (Docs/subtasks/*.html) from the Markdown sources.
# Usage: scripts/build-docs.sh [--only N [N ...]]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv found. Create it first: uv venv && uv pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PY" -m docsgen.build "$@"
