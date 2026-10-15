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
REPORT_DIR=".ai-governance/reports"
TS="$(date +%Y%m%d-%H%M%S)"

# CI 兼容（空暂存先行）：无暂存改动时直接跳过, 不要求 external kit 存在——
# CI/干净工作区场景不应因机器未装 kit 而误报; 有暂存差异时才做 kit 完整性检查。
# F-26：这里只判「有没有暂存改动」。NUL 分隔的输出**不能**存进 shell 变量
# （bash 会丢弃 NUL，多个路径会被拼接成一个），故「空判定」与「列表构建」分开：
# 判定用行式（只数条数），列表用 -z（见下方 STAGE_FILES/GATE_FILES）。
STAGED_COUNT="$(git diff --cached --name-only --diff-filter=ACMRD 2>/dev/null | grep -c . || true)"
[ "${STAGED_COUNT:-0}" -eq 0 ] && { say "无暂存改动，跳过"; exit 0; }

# M4(OODA-20261008-055): 门禁引擎默认取**宿主仓副本**（scripts/ai-verify/），使本地与 CI
# 同源同口径；跨项目统一升级 / kit 回归时用 AI_ENG_GUARD_USE_KIT=1 显式切到 kit。
# kit 自身运行时 SELF==KIT → GUARD_DIR 仍指向 kit（行为不变）；kit 路径固定，无新劫持面。
SELF_GUARD_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_GUARD_DIR="$HOME/qoder m5pro/_ai-eng-kit/governance"
GUARD_DIR="$SELF_GUARD_DIR"
if [ "${AI_ENG_GUARD_USE_KIT:-0}" = "1" ] || [ "$SELF_GUARD_DIR" = "$KIT_GUARD_DIR" ]; then
  if [ -d "$KIT_GUARD_DIR" ]; then
    GUARD_DIR="$KIT_GUARD_DIR"
  else
    # 显式请求 kit 载体（AI_ENG_GUARD_USE_KIT=1）但 kit 不存在 ⇒ fail-closed，
    # 不得静默回退仓库副本（否则「切载体」这个动作会变成无声的假成功）。
    printf '\033[1m[ai-guard]\033[0m ERROR: 显式请求 kit 载体但不存在: %s（fail-closed, 禁止放行）\n' "$KIT_GUARD_DIR" >&2
    exit 1
  fi
fi
if [ "$GUARD_DIR" = "$KIT_GUARD_DIR" ]; then
  # T11：紧跟非 ASCII 字符必须写 ${VAR}，否则 bash 在 UTF-8 locale 下把
  # 非 ASCII 字节并入变量名 → 展开为空并输出乱码（本次实证：`${GUARD_DIR}` 写成
  # `$GUARD_DIR` 后紧跟全角括号，值被吞且输出乱码字节）。
  say "引擎载体: kit（${GUARD_DIR}）"
else
  say "引擎载体: repo（${GUARD_DIR}）"
fi
if [ ! -f "$GUARD_DIR/quality-gate.sh" ] || [ ! -f "$GUARD_DIR/risk-classify.sh" ]; then
  printf '\033[1m[ai-guard]\033[0m ERROR: 载体不完整（缺 quality-gate.sh/risk-classify.sh）: %s（fail-closed）\n' "$GUARD_DIR" >&2
  exit 1
fi
# 双载体分叉告警（不 fail-closed；硬门禁由 CI 的 test-guard-parity.sh 承担）
# 清单含 ai-guard.sh 自身与 hook-integrity.sh：前者是自举盲区（改它必然造成「本仓 ↔ kit」
# 分叉，若它不在清单里就永无人告警）；加入后改本文件必须 kit 先行 + 回灌，否则每次提交
# 都会自我告警。清单外的共享载体仍由 test-guard-parity.sh 的字节一致断言兜底。
_guard_sha(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
if [ -d "$KIT_GUARD_DIR" ] && [ "$SELF_GUARD_DIR" != "$KIT_GUARD_DIR" ]; then
  for _pf in mutation-check.sh risk-classify.sh ai-guard.sh hook-integrity.sh; do
    if [ -f "$SELF_GUARD_DIR/$_pf" ] && [ -f "$KIT_GUARD_DIR/$_pf" ] \
       && [ "$(_guard_sha "$SELF_GUARD_DIR/$_pf")" != "$(_guard_sha "$KIT_GUARD_DIR/$_pf")" ]; then
      printf '\033[1m[ai-guard]\033[0m ⚠️  双载体分叉: %s（仓库 %s ≠ kit %s）\n' \
        "$_pf" "$(_guard_sha "$SELF_GUARD_DIR/$_pf" | cut -c1-12)" "$(_guard_sha "$KIT_GUARD_DIR/$_pf" | cut -c1-12)"
    fi
  done
fi

# ---------- A5：门禁临时目录按「仓库+进程」隔离（2026-10-10 自 nutrition-system-hospital 回灌） ----------
# 缺陷：多仓共用 _ai-eng-kit 的 quality-gate.sh 时，固定 /tmp/qgate_* 会被兄弟项目并发写坏
# （2026-10-02 实证：覆盖率 JSON 被覆盖 → 假 22.88%）。quality-gate.sh 已支持读 QGATE_TMP_DIR
# （见其 §16 行），但**派生逻辑**此前只存在于宿主仓副本；kit 作为共享载体必须自带，
# 否则「kit 载体」路径下仍是共享 /tmp。CI 由 workflow 注入 runner.temp（显式值优先，此处不覆盖）。
if [ -z "${QGATE_TMP_DIR:-}" ]; then
  QGATE_TMP_DIR="${TMPDIR:-/tmp}/qgate-$(printf '%s' "$ROOT" | cksum | awk '{print $1}')-$$"
fi
export QGATE_TMP_DIR
if ! mkdir -p "$QGATE_TMP_DIR"; then
  printf '\033[1m[ai-guard]\033[0m ERROR: 无法创建门禁临时目录: %s（fail-closed）\n' "$QGATE_TMP_DIR" >&2
  exit 1
fi

# ---------- A6：执行前快照 $GUARD_DIR 并做 bash 语法预检（2026-10-10 自 nutrition-system-hospital 回灌） ----------
# 缺陷：kit 为跨仓共享目录，若有并发会话正在改写其中脚本（2026-10-08 实证：另一会话回灌
# mci-world-model 时 quality-gate.sh 正处于写入中），bash 边读边执行会读到半成品并抛
# "syntax error"，被误判成「质量门禁未通过」并白白消耗重试次数。
# 快照整目录而非单文件：quality-gate.sh 通过 $(dirname "$0") 定位同目录的 mutation-check.sh
# 等兄弟脚本，保持目录结构才能维持相对引用。
# 载体宣布（上一段「引擎载体: kit/repo」）保持**逻辑载体**不变，快照仅作执行载体，避免
# 「切载体」语义被快照路径掩盖；kit 自身运行（SELF==KIT）同样快照——共享目录的并发写风险
# 与「谁在跑」无关。快照目录落在 QGATE_TMP_DIR（不在载体目录内），不会自我递归。
GUARD_DIR_SRC="$GUARD_DIR"
GUARD_SNAPSHOT_OK=0
for _snp_try in 1 2 3; do
  _snp="$QGATE_TMP_DIR/guard-$_snp_try"
  if mkdir -p "$_snp" && cp -R "$GUARD_DIR_SRC"/. "$_snp"/ 2>/dev/null; then
    _snp_bad=0
    for _snp_sh in "$_snp"/*.sh; do
      [ -e "$_snp_sh" ] || continue
      bash -n "$_snp_sh" 2>/dev/null || { _snp_bad=1; break; }
    done
    if [ "$_snp_bad" -eq 0 ]; then
      GUARD_DIR="$_snp"
      GUARD_SNAPSHOT_OK=1
      break
    fi
  fi
  sleep 1
done
if [ "$GUARD_SNAPSHOT_OK" -ne 1 ]; then
  printf '\033[1m[ai-guard]\033[0m ERROR: 套件脚本快照校验失败（%s 可能正被并发会话改写）。\n' "$GUARD_DIR_SRC" >&2
  printf '\033[1m[ai-guard]\033[0m 本次为基础设施错误，不计入门禁失败次数，请稍后重试。\n' >&2
  exit 1
fi
# M4 ③ / U1: 工具口径统一指向仓内 .venv（显式环境变量优先；注入必须**对称**）
# U1（2026-10-15）不变式 I-tools：PYTEST_BIN 与 PYTHON_BIN 必须同属一个解释器环境。
# 只注入 pytest 不注入 python，等于基础层自己制造「运行时不一致」，最终表现为
# 仓内门禁假红（实证 su-memory-sdk .ai-governance/reports/20261015-170802-gate-L2.md）。
# 故把「对称注入 + 显式变量优先」抽成单一函数，供 §3 多布局探测复用（I3 单一维护源）。
pin_venv_tools(){   # $1=venv 的 bin 目录
  local _bindir="$1" _spec _n _b _cur
  [ -d "$_bindir" ] || return 0
  for _spec in RADON_BIN:radon RUFF_BIN:ruff PYTEST_BIN:pytest PYTHON_BIN:python; do
    _n="${_spec%%:*}"; _b="${_spec#*:}"
    eval "_cur=\${$_n:-}"
    if [ -z "$_cur" ] && [ -x "$_bindir/$_b" ]; then
      eval "export $_n=\"\$_bindir/$_b\""
    fi
  done
}
# U1（2026-10-15）「半套显式」补全：显式环境变量以**组**为单位生效。
# 实证（OODA-20261015-AIGUARD-KITSYNC-011）：开发者全局 export PYTEST_BIN=<系统 pytest>
# 而仓库另有 venv 时，旧逻辑只把 PYTHON_BIN 指向 venv → 基础层**自己**造出
# 「pytest 与 python 不同环境」，L2 直接红。正确语义：先把半套显式补成全套
# （从已显式者的 bin 目录取同环境伙伴），再用仓库 venv 补剩余空缺。
pin_tool_env_pair(){
  local _pd _c
  if [ -n "${PYTEST_BIN:-}" ] && [ -z "${PYTHON_BIN:-}" ]; then
    case "$PYTEST_BIN" in */?*)
      _pd="$(dirname "$PYTEST_BIN")"
      for _c in python python3; do
        [ -x "$_pd/$_c" ] && { export PYTHON_BIN="$_pd/$_c"; break; }
      done ;;
    esac
  elif [ -n "${PYTHON_BIN:-}" ] && [ -z "${PYTEST_BIN:-}" ]; then
    case "$PYTHON_BIN" in */?*)
      _pd="$(dirname "$PYTHON_BIN")"
      [ -x "$_pd/pytest" ] && export PYTEST_BIN="$_pd/pytest" ;;
    esac
  fi
}
pin_tool_env_pair
pin_venv_tools "$ROOT/.venv/bin"

mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/$TS-guard.md"

# 暂存文件转数组（macOS bash 3.2 兼容，不用 mapfile），显式传给定级/门禁，
# 避免无参调用时 risk-classify 把工作区未暂存改动也纳入定级（pre-commit 只审本次提交内容）
# H-05-17：定级含删除项（ACMRD，删除不得静默放行）；门禁只看仍存在的文件（ACMR），
# 否则 grep/bandit 会对已删除路径误报，且"仅删除"提交会退化成全仓扫描。
STAGE_FILES=()
while IFS= read -r -d '' _f; do STAGE_FILES+=("$_f"); done < <(
  git diff --cached --name-only -z --diff-filter=ACMRD 2>/dev/null)
GATE_FILES=()
while IFS= read -r -d '' _f; do GATE_FILES+=("$_f"); done < <(
  git diff --cached --name-only -z --diff-filter=ACMR 2>/dev/null)

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
# 判据本义：阻断「单次不可评审的大 blob」。diff 行数只是它的**代理量**（OODA v1.3.1 §5）。
# 对「治理载体同步」批次，评审对象在上游 kit，本仓动作是机械可验证的分发，
# 正确判据是「能否机械证明本批 = kit 单源同步」，而非行数。
# 豁免唯一通道（无环境变量、无开关，任一条不满足即维持硬拦截 ⇒ fail-closed）：
#   ① 引擎载体不是 kit 自身（kit 是评审点，kit 内不豁免）；
#   ② kit 载体目录存在；
#   ③ 本批**全部**暂存文件都落在 scripts/ai-verify/ 下，且 kit 存在同相对路径文件；
#   ④ 每件文件的**暂存对象**与 kit 对应文件 sha256 完全一致。
# 反滥用：想走此通道必须先把内容落进 kit（kit 侧另有独立门禁与评审），
# 仓内私自改一行即 sha256 不符 ⇒ 照旧硬拦截。禁止用 MAX_DIFF_LINES 放宽（审计已定性的绕过面）。
CARRIER_SYNC_PROOF=""
if [ "$DIFF_LINES" -gt "$MAX_DIFF" ]; then
  if [ "$SELF_GUARD_DIR" != "$KIT_GUARD_DIR" ] && [ -d "$KIT_GUARD_DIR" ]; then
    _all_sync=1; _n_sync=0
    for _f in "${STAGE_FILES[@]}"; do
      case "$_f" in
        scripts/ai-verify/*) _rel="${_f#scripts/ai-verify/}" ;;
        *) _all_sync=0; break ;;
      esac
      _kf="$KIT_GUARD_DIR/$_rel"
      [ -f "$_kf" ] || { _all_sync=0; break; }
      _stage_sha="$(git show ":$_f" 2>/dev/null | (sha256sum 2>/dev/null || shasum -a 256) | cut -d' ' -f1)"
      [ -n "$_stage_sha" ] && [ "$_stage_sha" = "$(_guard_sha "$_kf")" ] || { _all_sync=0; break; }
      _n_sync=$((_n_sync + 1))
    done
    if [ "$_all_sync" = "1" ] && [ "$_n_sync" -gt 0 ]; then
      CARRIER_SYNC_PROOF="kit 单源同步豁免：${_n_sync} 件载体暂存内容与 kit 逐件 sha256 一致"
    fi
  fi
  if [ -z "$CARRIER_SYNC_PROOF" ]; then
    say "❌ 暂存改动 $DIFF_LINES 行 > ${MAX_DIFF}，疑似 Vibe Coding，拆分后提交"
    echo "**❌ 超大 diff 拦截**" >> "$SUMMARY"
    echo "- 说明：非 kit 单源同步批次，行数判据全额生效" >> "$SUMMARY"
    exit 1
  fi
  say "⚠️  暂存改动 $DIFF_LINES 行 > ${MAX_DIFF} ⇒ ${CARRIER_SYNC_PROOF}"
  echo "**⚠️ 超大 diff：kit 单源同步豁免**" >> "$SUMMARY"
  echo "- 证据：${CARRIER_SYNC_PROOF}" >> "$SUMMARY"
  echo "- 判据：内容可机械证明为上游 kit 分发，非新增未评审逻辑" >> "$SUMMARY"
fi

# ---------- 2. 风险定级（显式传暂存文件，仅审本次提交内容） ----------
CLASSIFY_OUT="$(bash "$GUARD_DIR/risk-classify.sh" "${STAGE_FILES[@]}" 2>/dev/null || true)"
LEVEL_TAG="$(printf '%s\n' "$CLASSIFY_OUT" | grep '^MAX_LEVEL' | tail -1 | grep -oE 'L[012]$')"
[ -z "$LEVEL_TAG" ] && LEVEL_TAG=L2
LEVEL="${LEVEL_TAG#L}"
echo "- 风险等级: $LEVEL_TAG" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
printf '%s\n' "$CLASSIFY_OUT" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
say "风险定级: $LEVEL_TAG"

# ---------- 3. venv 工具自动探测（多布局；与 M4 ③ 同一函数，显式环境变量优先） ----------
# U1：本块原先无条件覆盖 RADON/RUFF/BANDIT/PYTEST，且**不含** PYTHON_BIN —— 既踩掉
# 显式环境变量，又制造 pytest/python 半套注入。改为复用 pin_venv_tools（对称 + 显式优先）。
# P1（2026-10-16 实证 · nutrition-system-hospital 开窗，台账 OODA-20261015-549/-551）：原探测表
# 缺 `backend/.venv-sdk/bin`（AGENTS §4.2 规定的「与生产镜像同源」解释器），且**首个存在目录即 break**
# ⇒ 该仓 `backend/.venv`（实测 0 个包的空壳 venv）排首位，把 PYTHON_BIN 钉到无依赖解释器，真正
# 带 pytest/bandit 的 `.venv-sdk` 永不被探测。修法两条：
#   ① 纳入 `.venv-sdk/bin` 并置首位（多布局并存时以「生产同源」优先）；
#   ② 去掉 break：pin_venv_tools 只填空缺，逐目录遍历才能做到「缺某工具的空壳 venv 不挡住
#      后一个 venv 提供该工具」；BANDIT_BIN 由 -z 守卫，不会被后序目录覆盖。
pin_tool_env_pair
for vd in "backend/.venv-sdk/bin" "backend/.venv/bin" ".venv/bin" "venv/bin"; do
  if [ -d "$ROOT/$vd" ]; then
    [ -x "$ROOT/$vd/bandit" ] && [ -z "${BANDIT_BIN:-}" ] && export BANDIT_BIN="$ROOT/$vd/bandit"
    pin_venv_tools "$ROOT/$vd"
  fi
done

# ---------- 3.1 工具口径自证（本次实际解析结果写入报告） ----------
# 背景（OODA-20261015-549/-551）：A 轮「官方读数」实跑在 /Users/mac/miniforge3/bin/python
# （上游 editable su_memory 4.4.1），与生产镜像口径（backend/.venv-sdk · 仓内快照 4.1.1）不同源，
# 读数按 AGENTS §4.2 作废。环境变量优先是**有意设计**（U1），不可取消；那就必须让「这轮到底跑了
# 哪个解释器」在报告里自证 —— 复核人据此一眼判定读数是否可用于生产，不靠事后翻日志考古。
{
  echo ""
  echo "## 工具口径（本次实际解析）"
  echo '```'
  for _tv in PYTEST_BIN PYTHON_BIN RADON_BIN RUFF_BIN BANDIT_BIN; do
    eval "_tval=\${$_tv:-}"
    if [ -n "$_tval" ]; then echo "$_tv=$_tval"; else echo "$_tv=(未设置)"; fi
  done
  case "${PYTEST_BIN:-}" in
    "") _src="未设置（留给 PATH 解析）" ;;
    "$ROOT"/*) _src="仓内 venv/工具" ;;
    *) _src="仓外路径 —— 须确认与生产同源（AGENTS §4.2）" ;;
  esac
  echo "PYTEST_BIN 来源: $_src"
  if [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
    _exe="$("$PYTHON_BIN" -c 'import sys;print(sys.executable)' 2>/dev/null || true)"
    [ -z "$_exe" ] && _exe="(探测失败)"
    echo "sys.executable=$_exe"
    _sdk="$("$PYTHON_BIN" -c 'import su_memory, mci_world_model as m; print("su_memory=" + su_memory.__version__ + " mci_world_model=" + m.__version__)' 2>/dev/null || true)"
    [ -z "$_sdk" ] && _sdk="(未安装 su_memory/mci_world_model)"
    echo "sdk=$_sdk"
  fi
  echo '```'
} >> "$SUMMARY"

# ---------- 3.5 内置提示词 ↔ DB active 漂移预检 ----------
# 来源：nutrition-system-hospital 回灌（2026-10-10，台账 OODA-20261007-114 / OODA-20261009-043）。
# 背景：`backend/prompts/*.txt` 是「内置提示词」的文件真源，运行期加载顺序为「DB active 优先、
# 文件回退」⇒ 只改 *.txt 而不同步 DB 时运行期仍用旧模板（静默漂移，见 OODA-20261009-042）。
# 跨项目适用性（kit 为多项目共享载体）：仅当本仓**同时**具备
#   ① `scripts/ai-verify/sync_builtin_prompts.py`（本机制工具件）与
#   ② 本批暂存含 `backend/prompts/*.txt`
# 时介入；其余仓库行为不变（无工具件时显式告警后跳过，不误伤）。
# 解释器与降级语义（OODA-20261009-065 / V4 6d-118 硬化）：
#   候选按「venv → PATH 上的 python3/python」探测，但**只接受能 import sqlalchemy 的候选**；
#   一个都没有时按「本机是否真有库」分流，绝不静默放行：
#     ① 有库（backend/data_persist/nutrition.db、data_persist/nutrition.db 或显式 DATABASE_URL）
#        ⇒ fail-closed（有库却读不到 = 真盲区）；
#     ② 无库（CI / 干净克隆）⇒ 运行期走文件回退、本无漂移风险，显式告警后放行；
#     ③ 连解释器都没有 ⇒ fail-closed（不得放宽）。
PROMPT_SYNC_TOOL="$ROOT/scripts/ai-verify/sync_builtin_prompts.py"
PROMPT_TXT_STAGED="$(printf '%s\n' "${STAGE_FILES[@]}" 2>/dev/null | grep -E '^backend/prompts/.*[.]txt$' || true)"
if [ -n "$PROMPT_TXT_STAGED" ] && [ ! -f "$PROMPT_SYNC_TOOL" ]; then
  say "⚠️  本批含 backend/prompts/*.txt，但本仓无 scripts/ai-verify/sync_builtin_prompts.py ⇒ 漂移预检跳过（非本机制适用范围）"
  PROMPT_TXT_STAGED=""
fi
if [ -n "$PROMPT_TXT_STAGED" ]; then
  PROMPT_PY=""
  PROMPT_PY_FALLBACK=""
  for _py in "backend/.venv-sdk/bin/python" "backend/.venv/bin/python" ".venv/bin/python" \
             "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
    [ -n "$_py" ] || continue
    case "$_py" in /*) _cand="$_py" ;; *) _cand="$ROOT/$_py" ;; esac
    [ -x "$_cand" ] || continue
    if "$_cand" -c "import sqlalchemy" >/dev/null 2>&1; then PROMPT_PY="$_cand"; break; fi
    [ -n "$PROMPT_PY_FALLBACK" ] || PROMPT_PY_FALLBACK="$_cand"
  done
  if [ -z "$PROMPT_PY" ]; then
    PROMPT_DB_SEEN=0
    for _db in "backend/data_persist/nutrition.db" "data_persist/nutrition.db"; do
      [ -f "$ROOT/$_db" ] && PROMPT_DB_SEEN=1
    done
    [ -n "${DATABASE_URL:-}" ] && PROMPT_DB_SEEN=1
    if [ "$PROMPT_DB_SEEN" = "1" ]; then
      say "❌ 本批含 backend/prompts/*.txt，且本机存在数据库，但没有任何解释器能 import sqlalchemy"
      say "   候选: ${PROMPT_PY_FALLBACK:-<无>} ⇒ 无法判定漂移（fail-closed）；装依赖后重试"
      echo "**❌ 提示词漂移预检无可用解释器（本机有库）**" >> "$SUMMARY"
      exit 1
    fi
    if [ -n "$PROMPT_PY_FALLBACK" ]; then
      say "⚠️  本机无库且无一解释器含 sqlalchemy（CI/干净克隆）⇒ 漂移预检跳过（运行期走文件回退）"
      echo "- 提示词漂移预检: SKIP（无库环境，${PROMPT_PY_FALLBACK##*/} 缺 sqlalchemy）" >> "$SUMMARY"
      say "内置提示词漂移预检: SKIP"
    else
      say "❌ 本批含 backend/prompts/*.txt，但找不到可用的 python 解释器做漂移预检（fail-closed）"
      echo "**❌ 提示词漂移预检无法执行（无 python）**" >> "$SUMMARY"
      exit 1
    fi
  else
    say "内置提示词漂移预检: $(basename "$PROMPT_PY") sync_builtin_prompts.py --check"
    if ! "$PROMPT_PY" "$PROMPT_SYNC_TOOL" --check; then
      say "❌ 内置提示词与 DB active 版本漂移：先执行 sync_builtin_prompts.py 迁移后重试"
      echo "**❌ 提示词漂移预检未通过**" >> "$SUMMARY"
      exit 1
    fi
    echo "- 提示词漂移预检: PASS（$(basename "$PROMPT_PY")）" >> "$SUMMARY"
  fi
fi

# ---------- 4. 质量门禁 ----------
GATE_LOG="$REPORT_DIR/$TS-gate-$LEVEL_TAG.md"
if [ "${#GATE_FILES[@]}" -eq 0 ]; then
  # 仅删除提交：无剩余文件可查，跳过文件级门禁（删除本身已按 L2/L1 定级）
  say "本次提交仅含删除，跳过文件级质量门禁"
  { echo "- 仅删除提交：文件级门禁跳过（无剩余文件）"; } > "$GATE_LOG"
  GATE_RC=0
elif bash "$GUARD_DIR/quality-gate.sh" "$LEVEL_TAG" "${GATE_FILES[@]}" >> "$GATE_LOG" 2>&1; then
  GATE_RC=0
else
  GATE_RC=$?
fi
if [ "$GATE_RC" -ne 0 ]; then
  say "❌ 质量门禁未通过（${LEVEL_TAG}），详见 $GATE_LOG"
  echo "**❌ 质量门禁未通过** → $(basename "$GATE_LOG")" >> "$SUMMARY"
  # MAX_RETRY 实现：同一暂存内容累计失败次数，超限转人工（AI 应停止重试）。
  # F-17(P2-C): 计数移出项目 .ai-governance；损坏/非数字时阻断，不能重置。
  MAX_RETRY="${MAX_RETRY:-3}"
  DIFF_HASH="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
  [ -z "$DIFF_HASH" ] && DIFF_HASH="unknown"
  RETRY_STORE="$HOME/.ai-eng-kit/retry"
  if ! mkdir -p "$RETRY_STORE" 2>/dev/null; then
    say "🛑 无法创建全局重试计数目录: ${RETRY_STORE}（fail-closed）"
    exit 1
  fi
  RETRY_FILE="$RETRY_STORE/.retry-$DIFF_HASH"
  retry=0
  if [ -f "$RETRY_FILE" ]; then
    retry="$(cat "$RETRY_FILE" 2>/dev/null || true)"
    case "$retry" in
      ''|*[!0-9]*)
        say "🛑 重试计数损坏或被篡改: ${RETRY_FILE}（fail-closed）"
        exit 1
        ;;
    esac
  fi
  retry=$((retry+1))
  echo "$retry" > "$RETRY_FILE"
  # 清理 7 天前的重试记录（与创建/成功清理同一目录: $HOME/.ai-eng-kit/retry）
  find "$RETRY_STORE" -name '.retry-*' -mtime +7 -delete 2>/dev/null
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
# 门禁通过：清除该内容 hash 的重试计数（与失败登记同一目录 $HOME/.ai-eng-kit/retry, F-17）
DIFF_HASH="$(git diff --cached 2>/dev/null | shasum -a 256 | cut -c1-16)"
[ -n "$DIFF_HASH" ] && rm -f "$HOME/.ai-eng-kit/retry/.retry-$DIFF_HASH" 2>/dev/null
say "✅ 质量门禁通过（${LEVEL_TAG}）"

# D-CI-7 / OODA-20261010-WAVE1-366（自 clinical-nutrition-finance-prod 回灌）：
# L2 放行标记（tree 绑定，供 .githooks/commit-msg 自动注入留痕）。放行凭证是环境变量，
# 提交后不留痕迹；这里按「index tree」落一条标记，commit-msg 据此把「谁/依据什么放行」
# 写进 message。标记失败不阻断放行（留痕是增强，不是门禁）。
# 仅当本仓**确实装配了注入器**（生效 commit-msg 引用 l2-trailer-inject.sh）时才落标记——
# kit 被多项目共享，未接入该机制的项目不该在 .git 里留下无消费者的垃圾文件。
mark_l2_pass(){
  _basis="$1"; _owner="$2"; _tree=""; _gitdir=""; _hp=""; _cm=""
  _gitdir="$(git rev-parse --git-dir 2>/dev/null || true)"
  [ -n "$_gitdir" ] || return 0
  _hp="$(git config --get core.hooksPath 2>/dev/null || true)"
  case "$_hp" in
    "")   _cm="$_gitdir/hooks/commit-msg";;
    /*)   _cm="$_hp/commit-msg";;
    *)    _cm="$ROOT/$_hp/commit-msg";;
  esac
  grep -q 'scripts/ai-verify/l2-trailer-inject.sh' "$_cm" 2>/dev/null || return 0
  _tree="$(git write-tree 2>/dev/null || true)"
  [ -n "$_tree" ] || return 0
  printf 'owner=%s\nbasis=%s\nat=%s\n' "$_owner" "$_basis" "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
    > "$_gitdir/gov-l2-$_tree" 2>/dev/null || true
  return 0
}

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
    # L2 凭证（0-1 语义订正）: review-token 只证明「某持钥者签过」，**不是**第二人独立复核，也非不可伪造；
    # owner 自证的正式通道是下方 AI_REVIEW_CONFIRMED=1。AI_REVIEW_CI=1 留给 CI 侧校验。
    TOKEN="${AI_REVIEW_TOKEN:-}"
    if [ -n "$TOKEN" ]; then
      if bash "$GUARD_DIR/review-token.sh" verify "$TOKEN" HEAD >${TMPDIR:-/tmp}/gov_token.log 2>&1; then
        reviewer="$(grep -oE 'valid: .*' ${TMPDIR:-/tmp}/gov_token.log | sed 's/.*://;s/ *$//')"
        say "⚠️  L2 review-token 有效(reviewer: $reviewer)，允许提交（该 token 为留痕凭证，非独立复核证明）"
        echo "**⚠️ L2 放行(review-token valid, reviewer=$reviewer)**" >> "$SUMMARY"
        mark_l2_pass review-token "$reviewer"
        exit 0
      else
        say "🛑 L2 review-token 无效($(cat ${TMPDIR:-/tmp}/gov_token.log 2>/dev/null))"
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
      mark_l2_pass owner-explicit "${GOV_OWNER:-${OWNER:-sandy}}"
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
    say "   1. 签发 token（需确有第二持钥人；非 env 来源要求交互终端，无人值守会被拒）: bash review-token.sh issue <reviewer>"
    say "      然后: AI_REVIEW_TOKEN='<token>' git commit ..."
    say "   2. 推到远端, 由 CI 校验 PR approved-reviews-count>=2"
    echo "**🛑 L2 阻断(待双人Review token)**" >> "$SUMMARY"
    exit 1
    ;;
esac
