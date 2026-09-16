#!/usr/bin/env bash
# F-7(P1-G): pre-commit hook 完整性校验
# 本地: 比对实际安装 hook 与入库副本; CI: 比对入库副本与 hash 记录。
# 仅适用于 git-hook 模式项目; pre-commit 框架项目无入库副本时跳过。
set -o pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
HOOK="$ROOT/.git/hooks/pre-commit"
IN_REPO="$ROOT/scripts/ai-verify/hooks/pre-commit"
HASH_FILE="$ROOT/.ai-governance/hook-hash.txt"

sha(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
say(){ printf '\033[1m[hook-integrity]\033[0m %s\n' "$1"; }
FAIL=0

if [ ! -f "$IN_REPO" ]; then
  say "⚠️  无入库 hook 副本（pre-commit 框架项目或未 sync, 跳过）"; exit 0
fi
if [ ! -f "$HASH_FILE" ]; then
  say "❌ 缺 .ai-governance/hook-hash.txt（重跑 sync-governance.sh 生成）"; exit 1
fi

in_repo_hash="$(sha "$IN_REPO")"
recorded="$(head -1 "$HASH_FILE")"
if [ "$in_repo_hash" != "$recorded" ]; then
  say "❌ 入库 hook 与 hash 记录不符——scripts/ai-verify/hooks/pre-commit 被篡改"
  FAIL=1
fi

if [ -f "$HOOK" ]; then
  if [ "$(sha "$HOOK")" != "$in_repo_hash" ]; then
    say "❌ 实际安装的 pre-commit 与入库副本不符——本地 hook 被篡改, 重跑 sync-governance.sh 修复"
    FAIL=1
  else
    say "✅ 本地 hook 与入库副本一致"
  fi
else
  say "ℹ️  CI/无本地 hook 环境（跳过安装比对, 已完成入库副本校验）"
fi

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
