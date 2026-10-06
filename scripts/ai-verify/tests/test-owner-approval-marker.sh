#!/usr/bin/env bash
# 回归测试：单人模式审批指纹绑定 PR 号与日期（F-24 治理 K）
# 覆盖: 正向（当日/他日）、边界（跨日、跨 PR）、异常（非法参数）、旧静态指纹不再等价。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEN="$HERE/../owner-approval-marker.sh"
fail_n=0
ok_n=0

check() { # <描述> <期望> <实际>
  if [ "$2" = "$3" ]; then ok_n=$((ok_n+1)); echo "  ok: $1";
  else fail_n=$((fail_n+1)); echo "  FAIL: $1 (期望 [$2] 实际 [$3])"; fi
}
check_rc() { # <描述> <期望 rc> <实际 rc>
  if [ "$2" = "$3" ]; then ok_n=$((ok_n+1)); echo "  ok: $1";
  else fail_n=$((fail_n+1)); echo "  FAIL: $1 (期望 rc=$2 实际 rc=$3)"; fi
}

echo "[F-24 治理 K] 审批指纹绑定测试"
# 1. 正向：指定日期拼接正确
check "PR31 + 2026-10-06" "OWNER-APPROVED:PR31:2026-10-06" "$(bash "$GEN" 31 2026-10-06)"
# 2. 正向：默认日期为 Asia/Shanghai 当日
TODAY="$(TZ=Asia/Shanghai date +%F)"
check "默认日期=北京时间当日" "OWNER-APPROVED:PR42:$TODAY" "$(bash "$GEN" 42)"
# 3. 边界：跨 PR 同日期 → 指纹不同
[ "$(bash "$GEN" 31 2026-10-06)" != "$(bash "$GEN" 32 2026-10-06)" ] \
  && { ok_n=$((ok_n+1)); echo "  ok: 跨 PR 指纹不同"; } \
  || { fail_n=$((fail_n+1)); echo "  FAIL: 跨 PR 指纹相同"; }
# 4. 边界：同 PR 跨日 → 指纹不同（旧日指纹不可复用）
[ "$(bash "$GEN" 31 2026-10-05)" != "$(bash "$GEN" 31 2026-10-06)" ] \
  && { ok_n=$((ok_n+1)); echo "  ok: 跨日指纹不同"; } \
  || { fail_n=$((fail_n+1)); echo "  FAIL: 跨日指纹相同"; }
# 5. 异常：旧静态指纹不再等价于任何当日指纹
OLD='OWNER-APPROVED:2026-10-04:PR22'
[ "$OLD" != "$(bash "$GEN" 22 2026-10-04)" ] \
  && { ok_n=$((ok_n+1)); echo "  ok: 旧静态指纹不再被复现"; } \
  || { fail_n=$((fail_n+1)); echo "  FAIL: 旧静态指纹仍可复现"; }
# 6. 异常：非数字 / 空 / 非正 PR 号
bash "$GEN" abc 2026-10-06 >/dev/null 2>&1; check_rc "非数字 PR 号被拒" 2 $?
bash "$GEN" "" 2026-10-06 >/dev/null 2>&1; check_rc "空 PR 号被拒" 2 $?
bash "$GEN" 0 2026-10-06 >/dev/null 2>&1; check_rc "PR 号 0 被拒" 2 $?
bash "$GEN" 31 2026-10-6 >/dev/null 2>&1; check_rc "非法日期被拒" 2 $?
# 7. 异常：前缀可覆盖（供多项目复用，不改变默认口径）
check "GOV_OWNER_APPROVAL_PREFIX 覆盖" "OWNER-OK:PR7:2026-10-06" \
  "$(GOV_OWNER_APPROVAL_PREFIX=OWNER-OK bash "$GEN" 7 2026-10-06)"

echo "[F-24 治理 K] 通过 $ok_n 项, 失败 $fail_n 项"
[ "$fail_n" -eq 0 ]
