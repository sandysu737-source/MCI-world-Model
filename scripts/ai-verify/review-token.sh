#!/usr/bin/env bash
# L2 review-token 签发与校验（H-05-16/17 审计后双语义修订）
#
# 语义（H-05-19）：多人群组 = 第二人签发（HMAC 共享密钥 + reviewer≠提交者）；
# 单人团队 = owner-confirmed（负责人显式确认，token 仅作留痕，非第二人证明）。
# 单人治理的正式通道为 ai-guard.sh 的 AI_REVIEW_CONFIRMED=1（H-05-19 选项 1）。
#
# 用法1 签发: review-token.sh issue <reviewer> [commit-ish]
#   读取 GOV_REVIEW_SECRET 环境变量(团队共享密钥), 输出 token
# 用法2 校验(ai-guard 内部): review-token.sh verify <token> <commit-ish>
#
# 安全模型（H-05-16 T1-T5 实锤后修正）:
#   - 共享 HMAC 密钥下签名只证明"某持钥者"，reviewer 为自报字段——**不能**作为
#     第二人 review 的不可伪造证明；多人群组场景须配合 CI reviewer≠提交者比对
#     （governance.yml G1）与 PR 双人 approve。
#   - 单人团队：token 仅作签发留痕，权威来源 = 负责人显式确认。
#   - 最终防线在 CI: 硬词扫描 + token 校验 + PR approved-reviews-count>=2。
set -o pipefail

SECRET="${GOV_REVIEW_SECRET:-}"
if [ -z "$SECRET" ]; then
  echo "ERROR: 未设置 GOV_REVIEW_SECRET 环境变量" >&2
  echo "  团队密钥由管理员分发; 个人签发需持有密钥" >&2
  exit 2
fi

cmd="${1:-verify}"
shift || true

hmac() {
  # 用 openssl 生成 HMAC-SHA256, 输出 hex
  printf '%s' "$2" | openssl dgst -sha256 -hmac "$1" 2>/dev/null | awk '{print $NF}'
}

case "$cmd" in
  issue)
    reviewer="${1:?用法: issue <reviewer> [commit-ish]}"
    commit="${2:-HEAD}"
    commit="${commit:-HEAD}"
    # 注意: $(cmd || echo fallback) 在 cmd 有 stdout 但退出非0时会拼接污染(unborn分支实测),
    # 故改为先取输出再判空。
    short="$(git rev-parse --short "$commit" 2>/dev/null)"
    [ -z "$short" ] && short="unborn"
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    [ -z "$branch" ] && branch="unborn"
    # F-3(P0-D): 绑定本次暂存内容 hash, 签发后偷换暂存即失效
    staged="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
    payload="L2:${branch}:${short}:${staged}:${reviewer}"
    sig="$(hmac "$SECRET" "$payload")"
    token="${payload}|${sig}"
    echo "$token"
    echo "  → 提交者使用: AI_REVIEW_TOKEN='$token' git commit ..." >&2
    ;;
verify)
    token="${1:?用法: verify <token> [commit-ish]}"
    commit="${2:-HEAD}"
    payload="${token%|*}"
    sig="${token##*|}"
    # 重新计算期望签名
    expect_sig="$(hmac "$SECRET" "$payload")"
    if [ -z "$expect_sig" ] || [ -z "$sig" ]; then
      echo "invalid: token 格式错误"; exit 1; fi
    if [ "$sig" != "$expect_sig" ]; then
      echo "invalid: 签名不匹配(密钥错误或token被篡改)"; exit 1; fi
    # 校验 commit 匹配(防 token 跨提交复用)
    token_commit="${payload##*:}"           # 取 payload 末段(原为 reviewer, 这里需调整)
    # payload 格式 L2:branch:short:reviewer, commit 是第3段
    token_branch="$(printf '%s' "$payload" | cut -d: -f2)"
    token_short="$(printf '%s' "$payload" | cut -d: -f3)"
    cur_short="$(git rev-parse --short "$commit" 2>/dev/null)"
    if [ -n "$cur_short" ] && [ "$token_short" != "$cur_short" ]; then
      echo "invalid: token 绑定的提交($token_short)与当前($cur_short)不符"; exit 1; fi
    # F-3(P0-D): 校验暂存内容 hash, 防签发后偷换（旧4段格式token因段位错位自然失效, 需重新签发）
    staged_now="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
    token_staged="$(printf '%s' "$payload" | cut -d: -f4)"
    if [ "$token_staged" != "$staged_now" ]; then
      echo "invalid: token 绑定的暂存内容与当前不符(暂存已变更, 需重新签发)"; exit 1; fi
    echo "valid: $payload"
    exit 0
    ;;
  *)
    echo "用法: review-token.sh issue|verify ..." >&2; exit 2;;
esac
