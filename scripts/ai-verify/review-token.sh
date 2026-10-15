#!/usr/bin/env bash
# L2 review-token 签发与校验
#
# 语义（0-1 修订，OODA-20261014-AIGUARD-FPAUDIT-003）：**本机 token 通道不是"第二人 review"**。
#   HMAC 共享密钥 + reviewer 自报字段，签发者即持钥者；在"单操作者 + 同一台机 + 同一个用户"
#   的拓扑下（本工作区实况），持钥者与提交者是同一主体 ⇒ 该 token 只证明"某持钥者签过"，
#   不构成独立复核，**不得当作"第二人已看过"的证据使用**。
#   实测依据：finance 仓 76 个 L2 放行标记 100% 为 `basis=owner-explicit`；历史提交中
#   `AI_REVIEW_TOKEN=` 出现 0 次 ⇒ 该通道从未被使用；且密钥曾明文写在 `~/.zshrc`
#   （本机每个 shell 都继承）⇒ 原来的"零摩擦签发"连留痕价值都很低。
#   L2 的正式通道是 ai-guard.sh 的 `AI_REVIEW_CONFIRMED=1`（owner 显式确认，非第二人）。
#   若**确有**第二主体（团队多人、各自持有不同密钥），此通道才有独立复核含义；本文件不假设其存在。
#
# 用法1 签发: review-token.sh issue <reviewer> [commit-ish]
#   密钥来源见下方「密钥来源」段；非 env 来源的签发**要求交互终端**（0-1 门禁），
#   以阻断无人值守进程冒充"第二人"自签。
# 用法2 校验(ai-guard 内部): review-token.sh verify <token> <commit-ish>
#
# 安全模型（H-05-16 T1-T5 实锤后修正）:
#   - 共享 HMAC 密钥下签名只证明"某持钥者"，reviewer 为自报字段——**不能**作为
#     第二人 review 的不可伪造证明；多人群组场景须配合 CI reviewer≠提交者比对
#     （governance.yml G1）与 PR 双人 approve——注意 CI 当前**不可用**（计费停摆，近 100 次 run 中
#     97 次失败），故该「配合」在本机不成立，不得据此认为已有独立复核（0-1 订正）。
#   - 单人团队：token 仅作签发留痕，权威来源 = 负责人显式确认。
#   - 最终防线在 CI: 硬词扫描 + token 校验 + PR approved-reviews-count>=2。
#
# F-3 暂存绑定的 CI 兼容（2026-10-09，OODA-20261009-WAVE1-353）：
#   F-3 把令牌绑定到签发时 `git diff --cached` 的内容 hash，防「签发后偷换暂存」。
#   但 CI 的干净检出里 `git diff --cached` 恒为空 ⇒ 硬比较会拒签**一切**本地合法令牌，
#   使 CI 的 token 通道永久失效——那是「静默削弱既有控制」，不可接受。
#   故显式降级规则：**当前暂存为空 ⇒ 无内容可偷换 ⇒ 跳过绑定比较，但打印告警行**；
#   暂存非空时一律硬比较（本地 pre-commit 恒为非空，保护不变）。
#   残余（本文件不负责）：`git commit --only <paths>`/`--no-verify` 这类不把内容放进
#   索引的路径，会让「暂存为空」成为事实，从而绕过绑定；这属索引式 pre-commit 的
#   通用盲区（同时绕过整个门禁），须由钩子侧「工作区 vs HEAD」比对单独覆盖。
set -o pipefail

# ---------- 密钥来源（0-1）----------
# 旧实现只认环境变量；而 `~/.zshrc` 曾 `export GOV_REVIEW_SECRET=...` ⇒ **本机每个 shell
# （含每个 AI 会话）的环境里都躺着这把密钥**，签发零摩擦，token 的"独立复核"语义随之落空。
# 现行顺序（第一个命中者胜出），并把「来源」作为是否允许无人值守签发的判据：
#   ① 环境变量 GOV_REVIEW_SECRET —— 显式注入（CI Secrets / 测试夹具）⇒ 允许非交互签发
#   ② macOS Keychain 服务 ai-guard-gov-review —— 本机管理员持有 ⇒ 签发需交互
#   ③ ~/.config/ai-guard/gov-review.secret（0600）—— 离线兜底 ⇒ 签发需交互
# 如实声明残余：同一用户下的任意进程仍可读 Keychain/该文件并自行计算 HMAC ⇒ 本机制提高的是
# "顺手绕过"的成本，**不是**不可伪造性（见 ai-guard 优化计划书 §1.4 的拓扑论证）。
KEYCHAIN_SERVICE="ai-guard-gov-review"
SECRET_FILE="${GOV_REVIEW_SECRET_FILE:-$HOME/.config/ai-guard/gov-review.secret}"
TOKEN_LOG="${GOV_TOKEN_LOG:-.ai-governance/reports/token-issue.log}"
SECRET=""; SECRET_SRC=""
if [ -n "${GOV_REVIEW_SECRET:-}" ]; then
  SECRET="$GOV_REVIEW_SECRET"; SECRET_SRC="env"
elif command -v security >/dev/null 2>&1 \
     && SECRET="$(security find-generic-password -s "$KEYCHAIN_SERVICE" -w 2>/dev/null)" \
     && [ -n "$SECRET" ]; then
  SECRET_SRC="keychain"
elif [ -f "$SECRET_FILE" ]; then
  SECRET="$(tr -d '\r\n' < "$SECRET_FILE" 2>/dev/null)"
  [ -n "$SECRET" ] && SECRET_SRC="file"
fi
if [ -z "$SECRET" ]; then
  echo "ERROR: 未找到 review 密钥（顺序: GOV_REVIEW_SECRET 环境变量 → Keychain(${KEYCHAIN_SERVICE}) → ${SECRET_FILE}）" >&2
  echo "  管理员分发路径: security add-generic-password -s ${KEYCHAIN_SERVICE} -a \$USER -w <密钥>" >&2
  exit 2
fi

# 签发留痕（0-1）：成功与拒绝都落一行，供事后审计（文件位于 .ai-governance/reports/，已 gitignore）
_log_issue(){
  mkdir -p "$(dirname "$TOKEN_LOG")" 2>/dev/null || true
  printf '%s %s src=%s tty=%s reviewer=%s ctx=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" "${SECRET_SRC:-none}" \
    "$([ -t 0 ] && echo tty || echo no-tty)" "$2" "${3:-}" >> "$TOKEN_LOG" 2>/dev/null || true
}

cmd="${1:-verify}"
shift || true

hmac() {
  # 用 openssl 生成 HMAC-SHA256, 输出 hex
  printf '%s' "$2" | openssl dgst -sha256 -hmac "$1" 2>/dev/null | awk '{print $NF}'
}

# 空暂存的内容 hash（动态计算，避免硬编码漂移）
EMPTY_STAGED_HASH="$(printf '' | shasum -a 256 | cut -c1-16)"

case "$cmd" in
  issue)
    reviewer="${1:?用法: issue <reviewer> [commit-ish]}"
    # 0-1 门禁：密钥由环境变量**显式**注入（CI/测试夹具）时不限；其余来源（Keychain/文件）
    # 要求**交互终端**（stdin 为 TTY），否则拒绝并留痕——阻断无人值守进程自签"复核人"令牌。
    # 只查 stdin 不查 stderr：`2>file` / `| tee` 这类正常用法不该被误伤（可用性优先于更严的门）。
    if [ "$SECRET_SRC" != "env" ] && [ ! -t 0 ]; then
      _log_issue DENIED "$reviewer" "unattended(src=${SECRET_SRC})"
      echo "🛑 拒绝签发：非交互环境（stdin 非 TTY）且密钥来源=${SECRET_SRC}" >&2
      echo "   本门禁用于阻断无人值守进程冒充『复核人』签发 L2 令牌。" >&2
      echo "   授权场景（CI/自动化）请由管理员显式注入环境变量：" >&2
      echo "     GOV_REVIEW_SECRET=xxx bash review-token.sh issue <reviewer>" >&2
      exit 3
    fi
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
    if [ "$staged" = "$EMPTY_STAGED_HASH" ]; then
      echo "警告: 签发时暂存区为空（无内容可绑定）；该令牌只在空暂存场景（如 CI）通过校验。" >&2
    fi
    payload="L2:${branch}:${short}:${staged}:${reviewer}"
    sig="$(hmac "$SECRET" "$payload")"
    token="${payload}|${sig}"
    _log_issue GRANTED "$reviewer" "branch=${branch} commit=${short}"
    echo "$token"
    echo "  → 提交者使用: AI_REVIEW_TOKEN='$token' git commit ..." >&2
    echo "  ⚠️  该 token 只证明「持钥者签过」，不是独立复核证据（0-1 语义订正）。" >&2
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
      if [ "$staged_now" = "$EMPTY_STAGED_HASH" ]; then
        echo "warning: 当前暂存区为空（CI/干净检出），跳过暂存绑定校验（无内容可偷换，F-3 本次降级）" >&2
      else
        echo "invalid: token 绑定的暂存内容与当前不符(暂存已变更, 需重新签发)"; exit 1
      fi
    fi
    echo "valid: $payload"
    exit 0
    ;;
  *)
    echo "用法: review-token.sh issue|verify ..." >&2; exit 2;;
esac
