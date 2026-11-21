#!/usr/bin/env bash
# Generate the explanatory GIFs (Docs/subtasks/animations/) and their gallery page.
# Usage: scripts/generate-animations.sh [--only N|NAME ...] [--quick] [--list] [--out DIR] [--steps N] ...
#        (see: scripts/generate-animations.sh --help)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv found. Create it first: uv venv && uv pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PY" -m animgen.cli "$@"
