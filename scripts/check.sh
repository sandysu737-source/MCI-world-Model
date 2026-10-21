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
# 棘轮（A-2，台账 -018/-019）：存量债不阻断，**新增泄漏阻断**，下降需人工下调基线。
# 计数会打印（数字不敏感）；类别名与词条原文只在报告文件里，而该目录已 gitignore。
"$PYTHON_BIN" scripts/ai-verify/sensitive_scan.py --tracked \
  --baseline .ai-ooda/evidence/public-surface-baseline.txt \
  --report .ai-governance/reports/sensitive-term-scan-repo.json || {
  echo "❌ 全仓口径命中数高于基线：新增敏感面已入库。请脱敏，或在白名单件中显式登记（reason/owner/过期日）。"
  exit 1
}

# L-13：台账是审计证据源，结构完整性必须与代码门禁同源；结构异常即阻断。
bash scripts/ai-verify/ledger-lint.sh .ai-ooda/ledger.md

AI_REVIEW_CI=1 bash scripts/ai-verify/ai-guard.sh
