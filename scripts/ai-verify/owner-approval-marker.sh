#!/usr/bin/env bash
# 单人模式 L2 审批期望指纹生成器（F-24 治理 K）
#
# 设计：期望指纹 = <前缀>:PR<PR号>:<YYYY-MM-DD(Asia/Shanghai)>
#   - 绑定当前 PR 号 → 消除跨 PR 重放
#   - 绑定当日日期 → 消除跨日重放（"当日授权"语义）
# 用法: owner-approval-marker.sh <pr_number> [YYYY-MM-DD]
#   第二参数仅供测试注入（生产由 TZ=Asia/Shanghai date +%F 生成）。
# 输出: 单行期望指纹；参数非法时退出码 2。
set -euo pipefail

PR_NUMBER="${1:-}"
PREFIX="${GOV_OWNER_APPROVAL_PREFIX:-OWNER-APPROVED}"

usage() {
  echo "用法: owner-approval-marker.sh <pr_number> [YYYY-MM-DD]" >&2
}

if [ -z "$PR_NUMBER" ]; then usage; exit 2; fi
case "$PR_NUMBER" in ''|*[!0-9]*) echo "PR 号必须为正整数: $PR_NUMBER" >&2; exit 2;; esac
[ "$PR_NUMBER" -gt 0 ] || { echo "PR 号必须为正整数: $PR_NUMBER" >&2; exit 2; }

DAY="${2:-$(TZ=Asia/Shanghai date +%F)}"
case "$DAY" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) echo "日期格式须为 YYYY-MM-DD: $DAY" >&2; exit 2;;
esac

printf '%s:PR%s:%s\n' "$PREFIX" "$PR_NUMBER" "$DAY"
