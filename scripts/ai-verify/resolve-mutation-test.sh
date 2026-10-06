#!/usr/bin/env bash
# 解析"改动源文件 → 变异测试目标"（F-25 治理 L）
# 解析顺序:
#   ① .ai-governance/test-map.tsv —— 覆盖率上下文推导的映射（权威源，可复现生成）
#   ② 同名启发式 tests/**/test_<模块>.py
# 输出: 命中时单行打印测试目标；未命中退出码 1（由调用方决定是否 fail-closed）。
# 用法: resolve-mutation-test.sh <repo 相对源文件路径>
# 环境: TEST_MAP_FILE 可覆盖映射文件路径（默认 .ai-governance/test-map.tsv）
set -uo pipefail

rel="${1:-}"
[ -n "$rel" ] || { echo "用法: resolve-mutation-test.sh <repo 相对源文件路径>" >&2; exit 2; }
rel="${rel#./}"
base="${rel##*/}"
base="${base%.py}"

# ① 映射表（覆盖率证据驱动的单一权威源）
MAP="${TEST_MAP_FILE:-.ai-governance/test-map.tsv}"
if [ -f "$MAP" ]; then
  hit="$(awk -F'\t' -v k="$rel" '!/^#/ && $1==k {print $2; exit}' "$MAP" 2>/dev/null || true)"
  if [ -n "$hit" ]; then
    if [ -e "$hit" ]; then printf '%s\n' "$hit"; exit 0; fi
    echo "resolve-mutation-test: 映射目标不存在($hit)，回退同名启发式: $rel" >&2
  fi
fi

# ② 同名启发式（历史口径，兼容未生成映射的仓库）
for cand in backend . ./backend; do
  found="$(find "$cand" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | head -1)"
  found="${found#./}"
  [ -n "$found" ] && { printf '%s\n' "$found"; exit 0; }
done

exit 1
