#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data}"
mkdir -p "$ROOT/hackathon_data" "$ROOT/outputs" "$ROOT/runtime"

echo "Prepared Render disk directories:"
echo "  $ROOT/hackathon_data"
echo "  $ROOT/outputs"
echo "  $ROOT/runtime"
