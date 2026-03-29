#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo ".venv is missing. Run ./scripts/bootstrap.sh first."
  exit 1
fi
exec .venv/bin/python -m coach.cli serve --host 0.0.0.0 --port "${PORT:-8080}"
