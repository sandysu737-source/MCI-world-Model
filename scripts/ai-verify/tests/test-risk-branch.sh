#!/usr/bin/env bash
# 回归测试：PR 事件定级口径（F-25 治理 L）
# 覆盖: RISK_BRANCH 覆盖分支判断；detached HEAD 未设 RISK_BRANCH 时给警告并按非生产分支定级；
#       目标 main/release/* → 业务代码 L2；文档类仍 L1（不误升级）。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RC="$ROOT/scripts/ai-verify/risk-classify.sh"
fail_n=0; ok_n=0

max_level() { # <args...>  -> 打印 MAX_LEVEL 值（L0/L1/L2）
  local out; out="$(cd "$ROOT" && "$@" 2>/dev/null | grep '^MAX_LEVEL' | tail -1 | grep -oE 'L[012]$')"
  printf '%s' "$out"
}
check() { if [ "$2" = "$3" ]; then ok_n=$((ok_n+1)); echo "  ok: $1";
  else fail_n=$((fail_n+1)); echo "  FAIL: $1 (期望 [$2] 实际 [$3])"; fi; }

echo "[F-25 治理 L] PR 侧定级口径测试"
BIZ="$ROOT/benchmarks/real_world/p03_metrics.py"
DOC="$ROOT/README.md"

check "RISK_BRANCH=main → 业务代码 L2" "L2" "$(max_level env RISK_BRANCH=main bash "$RC" "$BIZ")"
check "RISK_BRANCH=release/1.0 → 业务代码 L2" "L2" "$(max_level env RISK_BRANCH=release/1.0 bash "$RC" "$BIZ")"
check "RISK_BRANCH=feat/x → 业务代码 L1" "L1" "$(max_level env RISK_BRANCH=feat/x bash "$RC" "$BIZ")"
check "RISK_BRANCH=main + 文档类 → L1（不误升级）" "L1" "$(max_level env RISK_BRANCH=main bash "$RC" "$DOC")"

# detached HEAD 且未设 RISK_BRANCH：警告 + 非生产分支口径
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
( cd "$tmp" && git init -q . && printf 'def f():\n    return 1\n' > biz.py && \
  git add biz.py && git -c user.email=t@t -c user.name=t commit -qm init && git checkout -q --detach ) >/dev/null 2>&1
err="$(cd "$tmp" && env -u RISK_BRANCH bash "$RC" biz.py 2>&1 >/dev/null)"
check "detached 未设 RISK_BRANCH 给出警告" "1" "$(printf '%s' "$err" | grep -c 'RISK_BRANCH')"
check "detached 未设 RISK_BRANCH → L1" "L1" "$(cd "$tmp" && env -u RISK_BRANCH bash "$RC" biz.py 2>/dev/null | grep '^MAX_LEVEL' | tail -1 | grep -oE 'L[012]$')"
check "detached + RISK_BRANCH=main → L2" "L2" "$(cd "$tmp" && env RISK_BRANCH=main bash "$RC" biz.py 2>/dev/null | grep '^MAX_LEVEL' | tail -1 | grep -oE 'L[012]$')"

echo "[F-25 治理 L] 通过 $ok_n 项, 失败 $fail_n 项"
[ "$fail_n" -eq 0 ]
