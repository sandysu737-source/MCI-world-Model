#!/usr/bin/env bash
# AI 工程化提交守卫（governance 增强版）
# 兼容 macOS bash 3.2（不用 mapfile/关联数组）
# 串联：超大diff拦截 → 风险定级 → 质量门禁 → 按等级路由
# 路由: L0 放行 | L1 放行+评审标记 | L2 阻断待 AI_REVIEW_CONFIRMED=1 | 门禁失败阻断
set -o pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
# F-1(P0-A): 固化 kit 路径, 忽略 AI_ENG_KIT 环境变量, 防伪造 kit 劫持门禁（fail-closed）
unset AI_ENG_KIT
say(){ printf '\033[1m[ai-guard]\033[0m %s\n' "$1"; }

STAGED="$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null)"
[ -z "$STAGED" ] && { say "无暂存改动，跳过"; exit 0; }

if [ "${GITHUB_ACTIONS:-false}" = "true" ]; then
  KIT_GUARD_DIR="$ROOT/scripts/ai-verify"
else
  KIT_GUARD_DIR="$HOME/qoder m5pro/_ai-eng-kit/governance"
fi
if [ ! -d "$KIT_GUARD_DIR" ]; then
  printf '\033[1m[ai-guard]\033[0m ERROR: governance kit 不存在: %s（fail-closed, 禁止放行）\n' "$KIT_GUARD_DIR" >&2
  exit 1
fi

REPORT_DIR=".ai-governance/reports"
mkdir -p "$REPORT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$REPORT_DIR/$TS-guard.md"

# 暂存文件转数组（macOS bash 3.2 兼容，不用 mapfile），显式传给定级/门禁，
# 避免无参调用时 risk-classify 把工作区未暂存改动也纳入定级（pre-commit 只审本次提交内容）
STAGE_FILES=()
while IFS= read -r _f; do [ -n "$_f" ] && STAGE_FILES+=("$_f"); done <<EOF
$STAGED
EOF

DIFF_LINES=$(git diff --cached 2>/dev/null | wc -l | tr -dc '0-9')
[ -z "$DIFF_LINES" ] && DIFF_LINES=0
MAX_DIFF=${MAX_DIFF_LINES:-800}

{
  echo "# ai-guard 提交守卫报告"
  echo "- 时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- 分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "- 暂存改动: $DIFF_LINES 行 / 上限 $MAX_DIFF"
  echo
} > "$SUMMARY"

# ---------- 1. 超大 diff 拦截 ----------
if [ "$DIFF_LINES" -gt "$MAX_DIFF" ]; then
  say "❌ 暂存改动 $DIFF_LINES 行 > $MAX_DIFF，疑似 Vibe Coding，拆分后提交"
  echo "**❌ 超大 diff 拦截**" >> "$SUMMARY"
  exit 1
fi

# ---------- 2. 风险定级（显式传暂存文件，仅审本次提交内容） ----------
CLASSIFY_OUT="$(bash "$KIT_GUARD_DIR/risk-classify.sh" "${STAGE_FILES[@]}" 2>/dev/null || true)"
LEVEL_TAG="$(printf '%s\n' "$CLASSIFY_OUT" | grep '^MAX_LEVEL' | tail -1 | grep -oE 'L[012]$')"
[ -z "$LEVEL_TAG" ] && LEVEL_TAG=L2
LEVEL="${LEVEL_TAG#L}"
echo "- 风险等级: $LEVEL_TAG" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
printf '%s\n' "$CLASSIFY_OUT" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
say "风险定级: $LEVEL_TAG"

# ---------- 3. venv 工具自动探测 ----------
for vd in "backend/.venv/bin" ".venv/bin" "venv/bin"; do
  if [ -d "$ROOT/$vd" ]; then
    [ -x "$ROOT/$vd/radon" ]  && export RADON_BIN="$ROOT/$vd/radon"
    [ -x "$ROOT/$vd/ruff" ]   && export RUFF_BIN="$ROOT/$vd/ruff"
    [ -x "$ROOT/$vd/bandit" ] && export BANDIT_BIN="$ROOT/$vd/bandit"
    [ -x "$ROOT/$vd/pytest" ] && export PYTEST_BIN="$ROOT/$vd/pytest"
    break
  fi
done

# ---------- 4. 质量门禁 ----------
GATE_LOG="$REPORT_DIR/$TS-gate-$LEVEL_TAG.md"
if bash "$KIT_GUARD_DIR/quality-gate.sh" "$LEVEL_TAG" "${STAGE_FILES[@]}" >> "$GATE_LOG" 2>&1; then
  GATE_RC=0
else
  GATE_RC=$?
fi
if [ "$GATE_RC" -ne 0 ]; then
  say "❌ 质量门禁未通过（$LEVEL_TAG），详见 $GATE_LOG"
  echo "**❌ 质量门禁未通过** → $(basename "$GATE_LOG")" >> "$SUMMARY"
  # MAX_RETRY 实现：同一暂存内容累计失败次数，超限转人工（AI 应停止重试）。
  # F-17(P2-C): 计数移出项目 .ai-governance；损坏/非数字时阻断，不能重置。
  MAX_RETRY="${MAX_RETRY:-3}"
  DIFF_HASH="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
  [ -z "$DIFF_HASH" ] && DIFF_HASH="unknown"
  RETRY_STORE="$HOME/.ai-eng-kit/retry"
  if ! mkdir -p "$RETRY_STORE" 2>/dev/null; then
    say "🛑 无法创建全局重试计数目录: $RETRY_STORE（fail-closed）"
    exit 1
  fi
  RETRY_FILE="$RETRY_STORE/.retry-$DIFF_HASH"
  retry=0
  if [ -f "$RETRY_FILE" ]; then
    retry="$(cat "$RETRY_FILE" 2>/dev/null || true)"
    case "$retry" in
      ''|*[!0-9]*)
        say "🛑 重试计数损坏或被篡改: $RETRY_FILE（fail-closed）"
        exit 1
        ;;
    esac
  fi
  retry=$((retry+1))
  echo "$retry" > "$RETRY_FILE"
  # 清理 7 天前的重试记录
  find "$REPORT_DIR" -name '.retry-*' -mtime +7 -delete 2>/dev/null
  remaining=$((MAX_RETRY - retry + 1))
  if [ "$remaining" -le 0 ]; then
    say "🛑 已达 MAX_RETRY($MAX_RETRY) 上限，转人工。AI 必须停止自动重试。"
    say "   失败快照: $GATE_LOG ｜ 重试记录: $RETRY_FILE"
    echo "**🛑 达重试上限转人工** (内容hash $DIFF_HASH 失败 $retry 次)" >> "$SUMMARY"
  else
    say "   门禁失败第 $retry/$MAX_RETRY 次（剩余 $remaining 次自动重试机会）"
    echo "- 重试计数: $retry/$MAX_RETRY (内容hash $DIFF_HASH)" >> "$SUMMARY"
  fi
  exit 1
fi
# 门禁通过：清除该内容 hash 的重试计数
DIFF_HASH="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
[ -n "$DIFF_HASH" ] && rm -f "$REPORT_DIR/.retry-$DIFF_HASH" 2>/dev/null
say "✅ 质量门禁通过（$LEVEL_TAG）"

# ---------- 5. 分级路由 ----------
case "$LEVEL_TAG" in
  L0)
    say "✅ L0 自动放行"
    echo "**✅ L0 放行**" >> "$SUMMARY"
    exit 0
    ;;
  L1)
    say "✅ L1 放行 → 需【架构+用例评审】（不审源码）"
    echo "**✅ L1 放行(待架构/用例评审)**" >> "$SUMMARY"
    exit 0
    ;;
  L2)
    # L2 凭证: review-token(不可伪造, 需第二人签发) 或 CI 模式跳过本地(AI_REVIEW_CI=1留给CI校验)
    TOKEN="${AI_REVIEW_TOKEN:-}"
    if [ -n "$TOKEN" ]; then
      if bash "$KIT_GUARD_DIR/review-token.sh" verify "$TOKEN" HEAD >/tmp/gov_token.log 2>&1; then
        reviewer="$(grep -oE 'valid: .*' /tmp/gov_token.log | sed 's/.*://;s/ *$//')"
        say "⚠️  L2 review-token 有效(reviewer: $reviewer)，允许提交；CI 仍会复核 PR approved-reviews-count"
        echo "**⚠️ L2 放行(review-token valid, reviewer=$reviewer)**" >> "$SUMMARY"
        exit 0
      else
        say "🛑 L2 review-token 无效($(cat /tmp/gov_token.log 2>/dev/null))"
        echo "**🛑 L2 阻断(token invalid)**" >> "$SUMMARY"
        exit 1
      fi
    fi
    # 单人团队负责人显式确认通道（H-05-19 选项 1，D-04 修订）：
    # 仅由负责人在会话中明示授权后携带；AI 不得自行设置。summary 落
    # confirmed=owner-explicit 留痕；push 后 CI 硬词扫描仍为最终防线。
    if [ "${AI_REVIEW_CONFIRMED:-0}" = "1" ]; then
      say "⚠️  L2 owner-explicit 确认放行（单人治理 H-05-19，非第二人 review）"
      echo "**⚠️ L2 放行(owner-explicit confirm)**" >> "$SUMMARY"
      exit 0
    fi
    # F-1(P0-C): AI_REVIEW_CI=1 仅在真实 CI 环境（GITHUB_ACTIONS=true）生效, 本地设值一律拒绝
    if [ "${AI_REVIEW_CI:-0}" = "1" ] && [ "${GITHUB_ACTIONS:-false}" != "true" ]; then
      say "🛑 AI_REVIEW_CI=1 仅允许在 CI 环境（GITHUB_ACTIONS=true）使用, 本地禁止——疑似门禁绕过"
      echo "**🛑 L2 阻断(AI_REVIEW_CI 滥用, 非CI环境)**" >> "$SUMMARY"
      exit 1
    fi
    # CI 环境: 本地不阻断, 由 CI 服务端做最终校验(读 PR review count)
    if [ "${AI_REVIEW_CI:-0}" = "1" ]; then
      say "ℹ️  CI 模式: L2 本地不阻断, 由 CI 复核 PR approved-reviews-count>=2"
      echo "**ℹ️ L2 CI模式(待CI复核)**" >> "$SUMMARY"
      exit 0
    fi
    say "🛑 L2 高风险：阻断提交。"
    say "   放行方式(任一):"
    say "   1. 第二人 review 后签发 token: GOV_REVIEW_SECRET=xxx bash review-token.sh issue <reviewer>"
    say "      然后: AI_REVIEW_TOKEN='<token>' git commit ..."
    say "   2. 推到远端, 由 CI 校验 PR approved-reviews-count>=2"
    echo "**🛑 L2 阻断(待双人Review token)**" >> "$SUMMARY"
    exit 1
    ;;
esac
