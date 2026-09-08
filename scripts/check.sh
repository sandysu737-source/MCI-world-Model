#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m ruff format --check .
"$PYTHON_BIN" -m mypy src/mci_world_model
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" scripts/ai-verify/sensitive_scan.py src README.md CHANGELOG.md docs \
  --report .ai-governance/reports/sensitive-term-scan-latest.json
AI_REVIEW_CI=1 bash scripts/ai-verify/ai-guard.sh
