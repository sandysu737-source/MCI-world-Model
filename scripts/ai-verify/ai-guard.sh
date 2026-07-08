#!/usr/bin/env bash
# AI 工程化守卫校验（通用版，按栈自动探测）
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
FAIL=0
say(){ printf "\033[1m[ai-guard]\033[0m %s\n" "$1"; }

DIFF_LINES=$(git diff --cached 2>/dev/null | wc -l | tr -dc '0-9' || echo 0)
[ -z "$DIFF_LINES" ] && DIFF_LINES=0
MAX_DIFF=${MAX_DIFF_LINES:-800}
if [ "$DIFF_LINES" -gt "$MAX_DIFF" ]; then
  say "ERROR: 暂存改动 $DIFF_LINES 行 > $MAX_DIFF，疑似 Vibe Coding，拆分后提交"; FAIL=1
fi

# 空 catch / 空 except
if git diff --cached --name-only 2>/dev/null | grep -E '\.(py)$' \
  | xargs -r grep -nE 'except[^:]*:\s*$' 2>/dev/null; then
  say "ERROR: 检测到空 except"; FAIL=1
fi
if git diff --cached --name-only 2>/dev/null | grep -E '\.(js|ts|tsx)$' \
  | xargs -r grep -nE 'catch\s*\([^)]*\)\s*\{\s*\}' 2>/dev/null; then
  say "ERROR: 检测到空 catch"; FAIL=1
fi

# f-string 拼接 SQL
if git diff --cached 2>/dev/null | grep -nE '^\+.*\b(f["\x27].*(SELECT|INSERT|UPDATE|DELETE))' ; then
  say "ERROR: 疑似 f-string 拼接 SQL，必须参数化"; FAIL=1
fi

# 明文打印敏感字段
if git diff --cached 2>/dev/null | grep -nE '^\+.*(print|console\.log|logger\.[a-z]+)\(.*\b(id_card|password|secret|token|apikey)\b' ; then
  say "ERROR: 疑似明文打印敏感字段"; FAIL=1
fi

# 前端 console.log 进 src
if git diff --cached --name-only 2>/dev/null | grep -E 'src/.*\.(ts|tsx)$' \
  | xargs -r grep -n 'console\.log' 2>/dev/null; then
  say "ERROR: src 中检测到 console.log"; FAIL=1
fi

if [ "$FAIL" -eq 1 ]; then say "校验未通过，禁止提交"; exit 1; fi
say "ai-guard 通过 ✓"
