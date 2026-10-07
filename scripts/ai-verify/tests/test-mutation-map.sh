#!/usr/bin/env bash
# 回归测试：变异测试映射解析（F-25 治理 L）
# 覆盖: 映射命中优先；映射缺失回退同名启发式；两者皆无 → 退出码 1（fail-closed 前置）；
#       映射目标不存在 → 警告 + 回退同名。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RESOLVER="$ROOT/scripts/ai-verify/resolve-mutation-test.sh"
fail_n=0; ok_n=0
check() { if [ "$2" = "$3" ]; then ok_n=$((ok_n+1)); echo "  ok: $1";
  else fail_n=$((fail_n+1)); echo "  FAIL: $1 (期望 [$2] 实际 [$3])"; fi; }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
map="$tmp/test-map.tsv"
printf '# 测试映射\nbenchmarks/real_world/p03_metrics.py\ttests/test_p03_metrics.py\t36\n' > "$map"

echo "[F-25 治理 L] 变异测试映射解析测试"
rel="benchmarks/real_world/p03_metrics.py"
check "映射命中优先" "tests/test_p03_metrics.py" \
  "$(cd "$ROOT" && TEST_MAP_FILE="$map" bash "$RESOLVER" "$rel")"
check "映射缺失回退同名启发式" "tests/test_p03_metrics.py" \
  "$(cd "$ROOT" && TEST_MAP_FILE="$tmp/absent.tsv" bash "$RESOLVER" "$rel")"
check "无映射且无同名测试 → 退出码 1" "1" \
  "$(cd "$ROOT" && TEST_MAP_FILE="$tmp/absent.tsv" bash "$RESOLVER" "src/__no_such_module__.py" >/dev/null 2>&1; echo $?)"
printf 'benchmarks/real_world/p03_metrics.py\ttests/__missing__.py\t36\n' > "$map"
err="$(cd "$ROOT" && TEST_MAP_FILE="$map" bash "$RESOLVER" "$rel" 2>&1 >/dev/null)"
check "映射目标不存在 → 警告" "1" "$(printf '%s' "$err" | grep -c '回退同名启发式')"
check "映射目标不存在 → 回退同名结果" "tests/test_p03_metrics.py" \
  "$(cd "$ROOT" && TEST_MAP_FILE="$map" bash "$RESOLVER" "$rel" 2>/dev/null)"

echo "[F-25 治理 L] 通过 $ok_n 项, 失败 $fail_n 项"
[ "$fail_n" -eq 0 ]
