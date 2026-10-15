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

# 全仓口径审计（公开仓：**入库即发布**）。口径选择见 scripts/ai-verify/sensitive_scan.py 头注：
# 私有仓用「交付路径」，公开仓必须用「受版本控制文件」，否则给假绿。
# **非阻断**：存量命中待负责人裁定（脱敏 / 显式白名单 / 收窄公开面，台账 OODA-20261015-PUBSURFACE-016）；
# 但绝不静默——状态与命中数必须打印。明细只落 .ai-governance/reports/（已 gitignore），
# 因为文件名与**类别名本身**也是敏感面，不得进入公开日志或提交。
if ! "$PYTHON_BIN" scripts/ai-verify/sensitive_scan.py --tracked \
     --report .ai-governance/reports/sensitive-term-scan-repo.json; then
  echo "⚠️  全仓口径敏感扫描未通过（非阻断·pending-owner；见台账 OODA-20261015-PUBSURFACE-016）"
fi
AI_REVIEW_CI=1 bash scripts/ai-verify/ai-guard.sh
