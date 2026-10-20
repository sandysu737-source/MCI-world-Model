#!/usr/bin/env bash
# 提交—门禁证据事后审计（L-2 / 0-4）
#
# 存在的理由：本工作区仓为私有仓库且无 GitHub Pro ⇒ 分支保护不可用（实测
#   `gh api repos/.../branches/main/protection` → 403 Upgrade to GitHub Pro）；远端 CI 只是信号，
#   且当前计费停摆（近 100 次 run 中 97 次失败）。唯一强制点是本地 pre-commit，而
#   `git commit --no-verify` 可零成本跳过它 ⇒ 必须有一处在**提交发生之后**、且**不被同一次
#   --no-verify 跳过**的检测点：git 的 post-commit 不在 --no-verify 的跳过名单内
#   （跳过的是 pre-commit / prepare-commit-msg / commit-msg / pre-push）。
#
# 判据换锚（0-4 / OODA-20261014-AIGUARD-FPAUDIT-003）。旧判据「提交时刻 ±15 分钟内存在任意
#   *-guard.md 报告」有两个致命缺陷：
#     ① 无绑定：任何一次无关提交留下的报告都能为后来的 L2 提交"洗白"——报告与提交之间没有
#        任何可验证的对应关系；
#     ② 不自动：只有人工执行，而实际从未有人执行（⇒ 12.4% 可疑长期无人处理）。
#   新判据逐条核验**与该提交本体绑定**的放行证据（任一即可）：
#     E1 message 含 `L2-REVIEWED-BY: <非空>`——由 .githooks/commit-msg → l2-trailer-inject.sh
#        在 ai-guard 按 L2 放行时注入；随历史持久、不受清理影响（首选证据）
#     E2 gitdir 存在 `gov-l2-<本提交 tree>`——ai-guard mark_l2_pass 落盘；tree 绑定，并发会话
#        互不串台；缺点是被清理/换机器后失效（2-2 将补清理规则）
#     E3 message 含 `AI_REVIEW_TOKEN=<非空>`——legacy 手工留痕
#   三条皆无 ⇒ **可疑（可能 --no-verify 绕过）**。
#   旧「同时段报告」降级为 E4 弱佐证：只写进告警正文，**不再作为放行依据**。
#
# 生效纪元（避免把机制上线前的历史提交一律判可疑）：自举取「历史上最早一条带 E1/E3 的提交」的
#   时间为锚，无需硬编码日期；可用 `.ai-governance/commit-audit.since`（epoch 秒，或 YYYY-MM-DD）
#   显式覆盖。锚点之前的 L2 提交归为 `legacy`：只计数、不判可疑、不误报。
#
# 已知边界（如实声明，勿高估——本体系的教训正是"注释比实现乐观"）：
#   · 本件是**事后检测**，不是阻断：它把"绕过"从"无人知晓"降为"有留痕、待处理"。
#   · E1/E2/E3 都落在本地可写区，主动伪造者仍可自造；本件针对的真实失效模式是"顺手 --no-verify"
#     （历史实证：CN_SKIP_GUARD=1 / --no-verify 的实际使用），不针对主动伪造。
#   · L2 判定**复用门禁自己的定级器** scripts/ai-verify/risk-classify.sh（单源），不再自带一套
#     路径正则——否则审计与门禁会成为"两把尺子"。实测（finance 近 14 天 244 条）：
#     路径近似器判 L2 65 条 / 可疑 0；定级器判 L2 112 条 / 可疑 1（= 已知无留痕提交 d2850e14）
#     ⇒ **近似器漏掉了真实绕过**，且近似器命中集是定级器命中集的子集（独有命中 0）。
#     定级器缺失（仓未接线）时退化到路径近似器兜底；定级器 SKIP（文件已不存在）的条目按
#     fail-closed 用近似器再判一次。
#   · 纪元锚点精度为 1 秒（git author date），与锚点同秒的提交会被归入 legacy（残余，不放大到分钟级）。
#   · 实现细节坑（F-27）：`git log --pretty=format:` 的记录间是"分隔符"语义，最后一条**不带尾换行**，
#     `while read` 会静默丢掉窗口内**最老的一条**提交 ⇒ 必须用 `tformat:`（记录间是"终结符"语义）。
#
# 用法:
#   commit-audit.sh [天数=14] [窗口分钟=15]   # 全量扫描（人工/定时）；发现可疑 → 退出 1
#   commit-audit.sh --hook                    # 只审 HEAD（post-commit 自动调用）；恒退出 0，
#                                             #   可疑追加到 .ai-governance/reports/commit-audit.log
# 退出码: 全量 0=未发现可疑 / 1=发现可疑；--hook 恒 0（事后审计不得阻断提交）
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || { echo "[commit-audit] 非 git 仓库，跳过" >&2; exit 0; }
GITDIR="$(git rev-parse --git-dir 2>/dev/null || echo .git)"
REPORT_DIR=".ai-governance/reports"
AUDIT_LOG="$REPORT_DIR/commit-audit.log"
SINCE_FILE=".ai-governance/commit-audit.since"

MODE=scan
case "${1:-}" in
  --hook)    MODE=hook ;;
  -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
  --*)       echo "[commit-audit] 未知参数: $1（见 --help）" >&2; exit 2 ;;
esac

if [ "$MODE" = hook ]; then
  DAYS=0; WINDOW_MIN=15
else
  DAYS="${1:-14}"; WINDOW_MIN="${2:-15}"
fi

say(){ printf '\033[1m[commit-audit]\033[0m %s\n' "$1"; }

# L2 路径启发式（与 0-4 前保持一致，本步只换判据、不动定级近似）
L2_PATH_RE='(payment|refund|settle|auth|login|webhook|alipay|wechatpay|wxpay|credential)'
L2_CONFIG_RE='(\.env$|docker-compose|\.github/workflows/|/migrations/.*\.py$|alembic/versions/.*\.py$|\.ai-coverage-threshold$|\.ai-governance/|/scripts/ai-verify/|_ai-eng-kit/)'
# 单源定级器（首选判据）；上面的 L2_PATH_RE/L2_CONFIG_RE 退居**兜底**（定级器缺失或 SKIP 时）
CLASSIFY="$ROOT/scripts/ai-verify/risk-classify.sh"

# ---------- 证据判定 ----------
msg_of(){ git show -s --format=%B "$1" 2>/dev/null; }
has_trailer(){ msg_of "$1" | grep -qE '^L2-REVIEWED-BY:[[:space:]]*[^[:space:]]'; }
has_token(){   msg_of "$1" | grep -qE 'AI_REVIEW_TOKEN=[^[:space:]]'; }
has_marker(){
  _t="$(git rev-parse --verify --quiet "$1^{tree}" 2>/dev/null || true)"
  [ -n "$_t" ] || return 1
  [ -f "$GITDIR/gov-l2-$_t" ]
}

# 路径近似器（兜底判据；定级器可判时不使用，避免"两把尺子"）
_heuristic_hit(){
  printf '/%s' "$1" | grep -qE "$L2_CONFIG_RE" && return 0
  printf '/%s' "$1" | tr 'A-Z' 'a-z' | grep -qE "$L2_PATH_RE"
}

# 命中 L2 的第一个文件（无命中返回 1）。判据优先级：定级器 > 兜底近似器。
l2_hit_of(){
  _h="$1"
  _files=()
  while IFS= read -r _f; do [ -n "$_f" ] && _files+=("$_f"); done \
    < <(git -c core.quotePath=false show --pretty=format: --name-only "$_h" 2>/dev/null | grep -v '^$')
  [ "${#_files[@]}" -gt 0 ] || return 1

  if [ -f "$CLASSIFY" ]; then
    _raw="$(bash "$CLASSIFY" "${_files[@]}" 2>/dev/null)"
    _l2="$(printf '%s\n' "$_raw" | awk -F'\t' '$1=="L2"{print $2; exit}')"
    if [ -n "$_l2" ]; then printf '%s' "${_l2#"$ROOT"/}"; return 0; fi
    # 定级器无法定级（SKIP not-a-file：该提交里的文件现已不存在）→ 对这**部分**条目用
    # 近似器 fail-closed 兜底；能定的条目一律以定级器为准。
    _skips="$(printf '%s\n' "$_raw" | awk -F'\t' '$1=="SKIP"{print $2}')"
    if [ -n "$_skips" ]; then
      while IFS= read -r _sk; do
        [ -n "$_sk" ] || continue
        _rel="${_sk#"$ROOT"/}"
        if _heuristic_hit "$_rel"; then printf '%s' "$_rel"; return 0; fi
      done <<EOF
$_skips
EOF
    fi
    return 1
  fi

  for _f in "${_files[@]}"; do
    if _heuristic_hit "$_f"; then printf '%s' "$_f"; return 0; fi
  done
  return 1
}

# 生效纪元（自举；可在 SINCE_FILE 显式覆盖）
audit_epoch(){
  if [ -f "$SINCE_FILE" ]; then
    _v="$(tr -d '[:space:]' < "$SINCE_FILE" 2>/dev/null || true)"
    if [ -n "$_v" ]; then
      case "$_v" in
        *-*)
          _e="$(date -j -f "%Y-%m-%d" "$_v" "+%s" 2>/dev/null || date -d "$_v" "+%s" 2>/dev/null || true)"
          if [ -n "$_e" ]; then printf '%s' "$_e"; return 0; fi
          ;;
        *[!0-9]*) : ;;   # 非法值 → 回落自举
        *)        printf '%s' "$_v"; return 0 ;;
      esac
    fi
  fi
  git log --reverse --format='%at' \
      --grep='^L2-REVIEWED-BY:' --grep='AI_REVIEW_TOKEN=' 2>/dev/null | head -1
}
EPOCH="$(audit_epoch)"
[ -n "$EPOCH" ] || EPOCH=0

# E4 弱佐证：旧判据的「同时段守卫报告」列表（惰性加载，避免无谓 IO）
_report_epochs=""; _report_loaded=0
load_report_epochs(){
  [ "$_report_loaded" = 1 ] && return 0
  _report_loaded=1
  [ -d "$REPORT_DIR" ] || return 0
  for _rf in "$REPORT_DIR"/*-guard.md; do
    [ -f "$_rf" ] || continue
    _ts="$(basename "$_rf" | sed -n 's/^\([0-9]\{8\}-[0-9]\{6\}\)-guard\.md$/\1/p')"
    [ -z "$_ts" ] && continue
    _e="$(date -j -f "%Y%m%d-%H%M%S" "$_ts" "+%s" 2>/dev/null \
          || date -d "${_ts:0:8} ${_ts:9:2}:${_ts:11:2}:${_ts:13:2}" "+%s" 2>/dev/null)"
    # T11 同类缺陷：`$VAR` 紧跟非 ASCII 会被 bash 吞掉变量名，须写 `${VAR}`（下方文案同理）
    [ -n "$_e" ] && _report_epochs="${_report_epochs} ${_e}"
  done
  return 0
}
weak_report_near(){
  load_report_epochs
  for _re in $_report_epochs; do
    _d=$(( $1 - _re )); [ "$_d" -lt 0 ] && _d=$(( -_d ))
    if [ "$_d" -le $(( WINDOW_MIN * 60 )) ]; then
      printf '有（±%s 分钟内存在守卫报告）' "$WINDOW_MIN"; return 0
    fi
  done
  printf '无'
}

# ---------- 逐条审计 ----------
checked=0; l2=0; passed=0; legacy=0; suspect=0
process(){
  _h="$1"; _ctime="$2"; _csubj="$3"
  checked=$((checked+1))
  _hit="$(l2_hit_of "$_h" || true)"
  [ -n "$_hit" ] || return 0
  l2=$((l2+1))
  _ce="$(date -j -f "%Y-%m-%d %H:%M:%S" "$_ctime" "+%s" 2>/dev/null || date -d "$_ctime" "+%s" 2>/dev/null || echo 0)"

  _ev=""
  has_trailer "$_h" && _ev="E1(message 留痕 L2-REVIEWED-BY)"
  [ -n "$_ev" ] || { has_token "$_h" && _ev="E3(legacy message token)"; }
  [ -n "$_ev" ] || { has_marker "$_h" && _ev="E2(gov-l2 tree 标记)"; }

  if [ -n "$_ev" ]; then
    passed=$((passed+1))
    if [ "$MODE" = hook ]; then
      mkdir -p "$REPORT_DIR" 2>/dev/null || true
      printf 'PASS %s %s | L2: %s | %s\n' "$_h" "$_ctime" "$_hit" "$_ev" >> "$AUDIT_LOG" 2>/dev/null || true
    else
      say "✅ ${_h:0:8} 触及 L2（${_hit}）· 放行证据 ${_ev}"
    fi
    return 0
  fi

  if [ "$_ce" -gt 0 ] && [ "$_ce" -lt "$EPOCH" ]; then
    legacy=$((legacy+1))
    [ "$MODE" = scan ] && say "➖ ${_h:0:8} 触及 L2（${_hit}）· 纪元前历史（机制上线前，不判可疑）"
    return 0
  fi

  suspect=$((suspect+1))
  _weak="$(weak_report_near "$_ce")"
  if [ "$MODE" = hook ]; then
    mkdir -p "$REPORT_DIR" 2>/dev/null || true
    printf 'SUSPECT %s %s | L2: %s | 无绑定放行证据（E1/E2/E3 皆缺）| 旧判据弱佐证: %s | %s\n' \
      "$_h" "$_ctime" "$_hit" "$_weak" "$_csubj" >> "$AUDIT_LOG" 2>/dev/null || true
    printf '\033[1m[commit-audit]\033[0m 🛑 事后审计：本提交触及 L2（%s）但无放行证据 → 可能 --no-verify 绕过；已记入 %s\n' \
      "$_hit" "$AUDIT_LOG" >&2
  else
    say "🛑 ${_h:0:8} 触及 L2（${_hit}）但无放行证据 → 可疑（可能 --no-verify 绕过）"
    say "   ${_ctime} | ${_csubj}"
    say "   旧判据弱佐证（不作放行依据）：${_weak}"
  fi
  return 0
}

if [ "$MODE" = hook ]; then
  # 只审 HEAD：post-commit 每次提交后运行，必须廉价（仅 2~5 次 git 调用）
  git rev-parse --verify --quiet HEAD >/dev/null 2>&1 || exit 0
  _line="$(git log -1 --pretty=tformat:'%H|%ad|%s' --date=format:'%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  [ -n "$_line" ] || exit 0
  IFS='|' read -r _h _t _s <<EOF
$_line
EOF
  process "$_h" "$_t" "$_s"
  exit 0
fi

while IFS='|' read -r _chash _ctime _csubj; do
  [ -z "$_chash" ] && continue
  process "$_chash" "$_ctime" "$_csubj"
done < <(git log --since="$DAYS days ago" --date=format:'%Y-%m-%d %H:%M:%S' --pretty=tformat:'%H|%ad|%s' 2>/dev/null)

say "审计 ${checked} 条提交（近 ${DAYS} 天）：触及 L2 ${l2} 条 = 通过 ${passed} + 纪元前 ${legacy} + 可疑 ${suspect}"
say "判据: E1 message 留痕 / E2 gov-l2 tree 标记 / E3 legacy token; 旧「同时段报告」已降级为弱佐证"
if [ -f "$CLASSIFY" ]; then
  say "L2 识别: risk-classify.sh（单源定级器）"
else
  say "⚠️  L2 识别: 路径近似器（未找到 ${CLASSIFY#"$ROOT"/}，与门禁口径可能存在偏差）"
fi
if [ "$EPOCH" != "0" ]; then
  say "生效纪元: $(date -r "$EPOCH" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d "@$EPOCH" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)（此前为 legacy，不判可疑）"
else
  # fail-closed 必须让人看见：没有锚点意味着「纪元前豁免」不可用，本仓所有 L2 提交都会被判可疑。
  say "⚠️  未找到留痕锚点（本仓历史无 E1/E3 留痕，且无 ${SINCE_FILE}）→ 纪元前豁免不可用："
  say "    本仓全部 L2 提交按可疑处理（fail-closed）。接入 commit-msg 注入器后锚点会自动建立。"
fi
[ "$suspect" -eq 0 ] && exit 0 || exit 1
