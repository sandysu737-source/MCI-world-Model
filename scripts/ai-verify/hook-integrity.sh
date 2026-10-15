#!/usr/bin/env bash
# F-7(P1-G): git 钩子完整性校验
# 本地: 比对实际安装 hook 与入库副本; CI: 比对入库副本与 hash 记录。
# 仅适用于 git-hook 模式项目; pre-commit 框架项目无入库副本时跳过。
#
# [D-CI-7 / OODA-20261010-WAVE1-366] 由「只校验 pre-commit」扩展为**枚举受管辖钩子**：
#   受管辖 = pre-commit + commit-msg。存在即逐项校验：
#     ① 入库（随 git 分发，钩子被版本化）；② 内容符合预期形状。
#   预期形状：
#     pre-commit : 必须调用 scripts/ai-verify/ai-guard.sh（治理门禁在位）。
#     commit-msg : 若引用 scripts/ai-verify/l2-trailer-inject.sh，则必须是**非阻断**形态——
#                  调用行带 `|| true` 且以 `exit 0` 收尾。防止注入器被改成阻断器，
#                  掐断并发会话提交（D-CI-7 决策：只注入、不阻断；缺留痕的最终防线在推送侧）。
#   未配送的受管辖钩子只报信息、不判失败（其它项目可能不需要 commit-msg）。
set -o pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
# 生效钩子的解析必须尊重 core.hooksPath：本仓/部分项目用 `.githooks/` 分发钩子
# （钩子入库、随 git 走），此时 `.git/hooks/pre-commit` 被 git **整体忽略**——
# 拿它当基准会得到一个"永远一致"的死文件比对，给出假安全感（F-26b）。
HOOKS_SETTING="$(git config --get core.hooksPath 2>/dev/null || true)"
case "$HOOKS_SETTING" in
  "")   HOOKS_DIR="$ROOT/.git/hooks";;
  /*)   HOOKS_DIR="$HOOKS_SETTING";;
  *)    HOOKS_DIR="$ROOT/$HOOKS_SETTING";;
esac
HOOK="$HOOKS_DIR/pre-commit"
COMMIT_MSG_HOOK="$HOOKS_DIR/commit-msg"
IN_REPO="$ROOT/scripts/ai-verify/hooks/pre-commit"
HASH_FILE="$ROOT/.ai-governance/hook-hash.txt"

sha(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
say(){ printf '\033[1m[hook-integrity]\033[0m %s\n' "$1"; }
FAIL=0

# D-CI-7：commit-msg 必须是**非阻断**形态。仅当它引用治理注入器时校验（未引用的
# commit-msg 属项目自有钩子，不归本件管辖，只报信息）。
check_commit_msg_managed(){
  _h="$1"
  if ! grep -q 'scripts/ai-verify/l2-trailer-inject.sh' "$_h" 2>/dev/null; then
    say "ℹ️  commit-msg 未引用 l2-trailer-inject.sh（非治理托管，跳过形状校验）"
    return 0
  fi
  _rc=0
  if grep -qE 'l2-trailer-inject\.sh.*\|\|[[:space:]]*true' "$_h" 2>/dev/null; then
    say "✅ commit-msg 以非阻断方式调用注入器（|| true）"
  else
    say "❌ commit-msg 调用了注入器但未带 \`|| true\`——被改成阻断器会掐断并发会话提交"
    _rc=1
  fi
  _last="$(grep -vE '^[[:space:]]*(#|$)' "$_h" 2>/dev/null | tail -1)"
  if [ "$_last" = "exit 0" ]; then
    say "✅ commit-msg 以 \`exit 0\` 收尾（恒不阻断）"
  else
    say "❌ commit-msg 末行非 \`exit 0\`（实际: ${_last:-<空>}）——钩子可能阻断提交"
    _rc=1
  fi
  return $_rc
}

# F-26b: hooksPath 指向**入库**目录时，钩子完整性由 git 保证（版本化 + 随仓库分发）。
# 此时唯一有意义的校验是：① 生效钩子确实入库 ② 仍调用治理守卫 ③ 死文件不冒充防线。
HOOK_REL="${HOOK#"$ROOT"/}"
if [ -f "$HOOK" ] && git -C "$ROOT" ls-files --error-unmatch "$HOOK_REL" >/dev/null 2>&1; then
  say "ℹ️  hooksPath 模式（core.hooksPath=${HOOKS_SETTING}）：受管辖钩子入库随 git 分发"
  # ① pre-commit
  if grep -q 'scripts/ai-verify/ai-guard.sh' "$HOOK" 2>/dev/null; then
    say "✅ pre-commit 仍调用 ai-guard（治理门禁在位）"
  else
    say "❌ pre-commit 未调用 ai-guard——治理门禁被移除/改向"
    FAIL=1
  fi
  # ② commit-msg（D-CI-7）
  CM_REL="${COMMIT_MSG_HOOK#"$ROOT"/}"
  if [ -f "$COMMIT_MSG_HOOK" ] && git -C "$ROOT" ls-files --error-unmatch "$CM_REL" >/dev/null 2>&1; then
    say "✅ commit-msg 入库随 git 分发"
    check_commit_msg_managed "$COMMIT_MSG_HOOK" || FAIL=1
  elif [ -f "$COMMIT_MSG_HOOK" ]; then
    say "❌ commit-msg 存在但未入库——钩子不随 git 分发（本地私有篡改面）"
    FAIL=1
  else
    say "ℹ️  未配送 commit-msg（跳过形状校验）"
  fi
  if [ -f "$ROOT/.git/hooks/pre-commit" ]; then
    say "ℹ️  存在 .git/hooks/pre-commit 但 hooksPath 已改向 ⇒ 该文件被整体忽略（死文件，不作为防线）"
  fi
  [ "$FAIL" -eq 0 ] && exit 0 || exit 1
fi

# 1-4b（OODA-20261015-011）：manager 委派模式（husky 实证 mci-huan）。
# 现象：core.hooksPath=web/.husky/_ 指向**管理器生成目录**（不入库），而 .git/hooks/ 被 git 整体忽略，
#   scripts/ai-verify/hooks/pre-commit 也不是生效面 ⇒ 拿生成目录与入库副本比对**必然报红**，
#   但真实门禁链是「git → husky shim → **入库的** web/.husky/pre-commit → ai-guard」。
# 判据（fail-closed）：生效钩子不入库时，必须能在其父目录找到**入库**的用户钩子，且该钩子调用 ai-guard；
#   找不到就报红——「红」与「门禁不在」是两件事，但两者都必须说清原因。
# 注意：仅在 **hooksPath 非空**时适用。hooksPath 为空＝框架模式，`.git/hooks/pre-commit`
# 由 pre-commit 框架生成、本就不入库（回到下方框架分支判定）。首版漏此条件，
# 使根仓（框架模式）被误判为「未找到委派钩子」而红——门禁误报与门禁失效同样有害。
if [ -n "$HOOKS_SETTING" ] && [ -f "$HOOK" ] \
   && ! git -C "$ROOT" ls-files --error-unmatch "$HOOK_REL" >/dev/null 2>&1; then
  _hp="${HOOKS_SETTING%/}"
  case "$_hp" in /*) _deleg_dir="$(dirname "$_hp")";; *) _deleg_dir="$ROOT/$(dirname "$_hp")";; esac
  DELEG="$_deleg_dir/pre-commit"
  DELEG_REL="${DELEG#"$ROOT"/}"
  # 判据取 **HEAD**（1-4c）：委派钩子的「调用 ai-guard」必须是**已提交**的事实，
  # 否则「工作区改了但没提交」会被读成门禁在位（mci-huan 实证：web/.husky/pre-commit 的
  # ai-guard 调用当时只在工作区，HEAD 里没有 ⇒ 提交态下门禁并未接线）。
  if [ -f "$DELEG" ] && git -C "$ROOT" ls-files --error-unmatch "$DELEG_REL" >/dev/null 2>&1; then
    _deleg_head="$(git -C "$ROOT" show "HEAD:$DELEG_REL" 2>/dev/null || true)"
    if printf '%s' "$_deleg_head" | grep -q 'scripts/ai-verify/ai-guard.sh'; then
      say "✅ manager 委派模式：生效钩子由管理器生成（不入库），入库的 ${DELEG_REL} 调用 ai-guard（门禁在位，按 HEAD 判定）"
      say "ℹ️  ${HOOK_REL} 由管理器生成、不入库，不作为基准；${DELEG_REL} 随 git 分发"
      exit 0
    fi
    if grep -q 'scripts/ai-verify/ai-guard.sh' "$DELEG" 2>/dev/null; then
      say "❌ ${DELEG_REL} 的 ai-guard 调用**只在工作区、未提交** ⇒ 提交态下门禁未接线（1-4c：未提交的接线不计在位）"
      say "    修法：提交 ${DELEG_REL}（该改动即接线本身）"
      exit 1
    fi
  fi
  say "❌ 生效钩子不入库，且未找到调用 ai-guard 的入库委派钩子（fail-closed，不得静默放行）"
  say "    生效钩子: ${HOOK_REL}（不入库）"
  say "    已查委派: ${DELEG_REL}$( [ -f "$DELEG" ] || printf '（不存在）' )"
  exit 1
fi

if [ ! -f "$IN_REPO" ]; then
  # 框架模式（pre-commit 生成 .git/hooks/pre-commit）：无入库副本可哈希比对，
  # 改为校验「框架 hook 仍在位」+「配置仍声明必需治理钩子」，防静默卸载/删钩子。
  CFG="$ROOT/.pre-commit-config.yaml"
  if [ -f "$CFG" ]; then
    # M6c 回环修复（2026-10-15，CI run 38698270012）：CI 检出不安装本地钩子，
    # 「框架 hook 在位」断言在 CI 必然失败（且与防篡改意图无关 —— 防篡改只在本地有意义）。
    # 故 CI 环境下只报信息；本地仍严格断言（缺失/被换成非 pre-commit 生成即红）。
    if [ "${GITHUB_ACTIONS:-}" = "true" ] || [ "${CI:-}" = "true" ]; then
      say "ℹ️  CI 环境：无本地 hook 安装，跳过「框架 hook 在位」断言（该断言仅在本地有防篡改意义）"
    elif [ ! -f "$HOOK" ] || ! grep -q "File generated by pre-commit" "$HOOK" 2>/dev/null; then
      say "❌ .git/hooks/pre-commit 缺失或非 pre-commit 生成（可能被卸载/篡改）"
      FAIL=1
    else
      say "✅ 框架 hook 在位（pre-commit 生成）"
    fi
    if grep -qE '^\s+-\s+id:\s*ai-guard\s*$' "$CFG"; then
      say "✅ 配置仍声明 ai-guard 钩子"
    else
      say "❌ .pre-commit-config.yaml 缺少 ai-guard 钩子（治理门禁被移除）"
      FAIL=1
    fi
    [ -f "$COMMIT_MSG_HOOK" ] && { check_commit_msg_managed "$COMMIT_MSG_HOOK" || FAIL=1; }
    [ "$FAIL" -eq 0 ] && exit 0 || exit 1
  fi
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

# D-CI-7：非 hooksPath 模式下的 commit-msg（存在即校验形状；未引用注入器只报信息）。
if [ -f "$COMMIT_MSG_HOOK" ]; then
  check_commit_msg_managed "$COMMIT_MSG_HOOK" || FAIL=1
else
  say "ℹ️  未安装 commit-msg（跳过形状校验）"
fi

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
