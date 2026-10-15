#!/usr/bin/env bash
# 轻量变异测试器（governance 自带，跨平台，不依赖 mutmut）
# 原理: 对源文件逐条应用典型变异(运算符翻转/常量变更),跑测试
#   测试仍 PASS = 变异存活 = 测试覆盖有洞; FAIL = 变异被杀 = 测试有效
#   变异分数 = 被杀 / 有效变异总数 × 100%
# F-23(P0-J) 回灌（2026-10-08，源自 mci-world-model 窗口），两项：
#   ① f-string 整段不计分：Python ≥3.12 把 f-string 拆成 FSTRING_START/MIDDLE/END，
#      只匹配 STRING 会漏排除其字面量里的运算符文本 → 同一文件在 3.11 与 3.13/3.14 落点不同。
#   ② 语法破坏型变异不计分：`->` 变 `-<` 之类变异后无法 `ast.parse`，不构成"测试有效性"
#      证据，计为杀死会虚高分数（apply_mut 退出码 3 = 跳过，汇总里单独计数）。
# 用法: mutation-check.sh <源文件> <测试目标[,测试目标…]> [pytest前缀...]
# 多目标（OODA-20261013-227，源自 -224 §四提案）：测试目标可用**逗号**分隔多个
#   —— 让「契约断言分散在多个测试文件」的模块也能被变异臂完整看见（旧版只认单目标，
#   异名文件里的断言不在视野，表现为「有测试却仍判覆盖不足」）。
# 目标解析（§15.7-1 双向合并，2026-10-15）：含逗号 ⇒ 逗号切分（每段一个目标，允许段内含
#   空格，兼容仓根带空格的绝对路径）；不含逗号 ⇒ shlex 空白切分（兼容多目标空白写法与带
#   引号参数）。pytest 前缀一律经 shlex 解析为参数数组后直接 exec，**不再经 `sh -c` 二次
#   求值**（消除引号被重解析的执行面）。
# 并发互斥（源自 su-memory-sdk，2026-10-15 上游化）：本脚本**原地改写被变异源文件**
#   （`$SRC.mutorig` 备份 + EXIT 还原）。同仓两进程同时跑会互相覆盖变异版本，得到错误
#   分数却不报错 ⇒ 默认加仓级互斥锁；`MUTATION_LOCK=0` 显式关闭（不承诺并发安全）。
#   stale 锁自动回收：写锁时记 pid，pid 已消失即判残留（SIGKILL 场景）并接管。
# 证据链（源自 su-memory-sdk，2026-10-15 上游化）：`MUTATION_REPORT_PATH` 非空时写机器
#   可读报告（schema_version=2）：target/tests/tool sha256、运行中漂移检测、输出日志
#   sha256、起止时间、锁断言；本仓若存在 `workspace-manifest.py` +
#   `source-scope-manifest.py` 与对应 JSON，则额外绑定其 fingerprint，不存在时记
#   `manifest_binding=absent`（不伪造指纹）。
# 行级跳过豁免（源自 su-memory-sdk，2026-10-15 上游化）：`MUTATION_SKIP_RES` 以 ';;'
#   分隔行级正则；本仓存在 `docs/evidence/mutation-skip-allowlist.json` 时**强制**逐条校验
#   「已审批 + 未过期」，缺失时不静默放行，而是在日志与报告中标注未审批。
# 可选环境变量: MUTATION_SCOPE=changed|all / MUTATION_DIFF_BASE / MUTATION_MIN /
#   MUTATION_SKIP_RES / MUTATION_REPORT_PATH / MUTATION_LOCK / MUTATION_LOCK_PATH /
#   MUTATION_OUTPUT_PATH / MUTATION_SKIP_ALLOWLIST_PATH
# 退出码: 0=达标或合法 N/A 1=不达标 2=环境错误
set -o pipefail
SRC="${1:?用法: mutation-check.sh <源文件> <测试> [pytest]}"
TESTS="${2:?缺测试目标（多个用逗号分隔）}"
if [ "$#" -ge 3 ]; then PYTEST="${*:3}"; else PYTEST="pytest"; fi
PY="${PYTHON_BIN:-python3}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
MUTATION_SKIP_RES="${MUTATION_SKIP_RES:-}"
MUTATION_REPORT_PATH="${MUTATION_REPORT_PATH:-}"
MUTATION_LOCK="${MUTATION_LOCK:-1}"
MIN="${MUTATION_MIN:-80}"

[ -f "$SRC" ] || { echo "ERROR: 源文件不存在: $SRC"; exit 2; }
# 路径规范化：git 的 toplevel 恒为物理路径（macOS `/tmp` → `/private/tmp`），
# 源路径若为逻辑路径会让 `git diff -- <rel>` 落到仓外并误报「无法读取 worktree diff」。
SRC="$(cd "$(dirname "$SRC")" && pwd -P)/$(basename "$SRC")"

# ---------- 仓库根 / 日志落点 ----------
MUT_REPO_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$MUT_REPO_ROOT" ] || MUT_REPO_ROOT="$PWD"
MUT_REPO_ROOT="$(cd "$MUT_REPO_ROOT" && pwd -P)"
MUTATION_REPORT_MODE=0
if [ -n "$MUTATION_REPORT_PATH" ]; then
  MUTATION_REPORT_MODE=1
  MUTATION_OUTPUT_PATH="${MUTATION_OUTPUT_PATH:-.ai-governance/reports/mutation-output-$(date +%Y%m%d-%H%M%S)-$$.log}"
  case "$MUTATION_OUTPUT_PATH" in /*) MUT_LOG="$MUTATION_OUTPUT_PATH";; *) MUT_LOG="$MUT_REPO_ROOT/$MUTATION_OUTPUT_PATH";; esac
  case "$MUTATION_REPORT_PATH" in /*) :;; *) MUTATION_REPORT_PATH="$MUT_REPO_ROOT/$MUTATION_REPORT_PATH";; esac
  mkdir -p "$(dirname "$MUT_LOG")" "$(dirname "$MUTATION_REPORT_PATH")" || { echo "ERROR: 报告目录不可写" >&2; exit 2; }
else
  MUTATION_OUTPUT_PATH=""
  MUT_LOG="${MUT_TMP_DIR:-/tmp}/mut_run.log"
fi
: > "$MUT_LOG" 2>/dev/null || { echo "ERROR: 无法写入日志: $MUT_LOG" >&2; exit 2; }
MUTATION_TARGET="$SRC"
MUTATION_TARGET_SHA256="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$SRC")"
MUTATION_TOOL_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
MUTATION_TOOL_SHA256="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MUTATION_TOOL_PATH")"
MUTATION_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MUTATION_RUN_FINISHED_AT=""
MUTATION_RUN_ID=0
MUTATION_PYC_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mutation-pycache.XXXXXX")"
MUT_TARGETS_FILE="$(mktemp)"
MUTATION_SKIP_ALLOWLIST_PATH="${MUTATION_SKIP_ALLOWLIST_PATH:-docs/evidence/mutation-skip-allowlist.json}"
MUTATION_EVIDENCE_WORKSPACE="${MUTATION_EVIDENCE_WORKSPACE:-docs/evidence/workspace-manifest.json}"
MUTATION_SKIP_ALLOWLIST_SHA256=""
MUTATION_SKIP_APPROVED="false"
MUTATION_LOCK_PATH="${MUTATION_LOCK_PATH:-.ai-governance/reports/locks/mutation-check.lock}"
case "$MUTATION_LOCK_PATH" in /*) MUT_LOCK_ABS="$MUTATION_LOCK_PATH";; *) MUT_LOCK_ABS="$MUT_REPO_ROOT/$MUTATION_LOCK_PATH";; esac
MUTATION_LOCK_ACQUIRED=0
MUTATION_LOCK_MODE="disabled"
MUTATION_MANIFEST_BINDING="absent"
MUTATION_WS_CHECK_BEFORE=""
MUTATION_WS_CHECK_AFTER=""

restore_src(){
  if [ -f "$SRC.mutorig" ]; then cp "$SRC.mutorig" "$SRC" 2>/dev/null || true; rm -f "$SRC.mutorig"; fi
}
cleanup(){
  restore_src
  [ "$MUTATION_LOCK_ACQUIRED" -eq 1 ] && rm -rf "$MUT_LOCK_ABS"
  rm -rf "$MUTATION_PYC_ROOT"
  rm -f "$MUT_TARGETS_FILE" "$CHANGED_LINES_FILE" "$CHANGED_LINES_FILE.point" \
        "$CHANGED_LINES_FILE.diff" "$CHANGED_LINES_FILE.cached.diff"
}
trap cleanup EXIT

# ---------- 并发互斥（能力源自 su-memory-sdk） ----------
acquire_lock(){
  [ "$MUTATION_LOCK" = "0" ] && { MUTATION_LOCK_MODE="disabled"; return 0; }
  mkdir -p "$(dirname "$MUT_LOCK_ABS")" 2>/dev/null || true
  if mkdir "$MUT_LOCK_ABS" 2>/dev/null; then
    MUTATION_LOCK_ACQUIRED=1; MUTATION_LOCK_MODE="repo-scoped-mkdir-exclusive"
    echo "$$" > "$MUT_LOCK_ABS/pid"
    return 0
  fi
  # 有界等待：多会话/多窗口并发提交时，等待优于直接判红（假红会削弱门禁可信度）；
  # 等待上限 MUTATION_LOCK_WAIT 秒（默认 60），超时才 fail-closed。
  _waited=0
  while [ "$_waited" -lt "${MUTATION_LOCK_WAIT:-60}" ]; do
    _holder="$(cat "$MUT_LOCK_ABS/pid" 2>/dev/null || true)"
    # stale 接管：持锁进程已消失（典型 SIGKILL/断电）时不得永久堵死门禁
    if [ -z "$_holder" ] || ! kill -0 "$_holder" 2>/dev/null; then
      echo "[mut] 警告: 回收残留 mutation lock（pid=${_holder:-未知} 已不存在）"
      rm -rf "$MUT_LOCK_ABS"
      break
    fi
    [ "$_waited" = "0" ] && echo "[mut] 等待并发 mutation 运行结束（pid=${_holder}，上限 ${MUTATION_LOCK_WAIT:-60}s）..."
    sleep 1; _waited=$((_waited + 1))
  done
  if mkdir "$MUT_LOCK_ABS" 2>/dev/null; then
    MUTATION_LOCK_ACQUIRED=1; MUTATION_LOCK_MODE="repo-scoped-mkdir-exclusive"
    echo "$$" > "$MUT_LOCK_ABS/pid"
    return 0
  fi
  echo "ERROR: mutation lock 等待 ${MUTATION_LOCK_WAIT:-60}s 仍被占用，禁止并发运行: $MUTATION_LOCK_PATH" >&2
  exit 2
}
acquire_lock

# ---------- 证据链绑定（manifest 存在才绑定，缺失不伪造） ----------
manifest_binding(){
  local ws_py="$MUT_REPO_ROOT/scripts/ai-verify/workspace-manifest.py"
  local sc_py="$MUT_REPO_ROOT/scripts/ai-verify/source-scope-manifest.py"
  if [ -f "$ws_py" ] && [ -f "$sc_py" ] \
     && [ -f "$MUT_REPO_ROOT/$MUTATION_EVIDENCE_WORKSPACE" ] \
     && [ -f "$MUT_REPO_ROOT/docs/evidence/source-scope-manifest.json" ]; then
    MUTATION_MANIFEST_BINDING="bound"
    "$PY" "$ws_py" --check >/dev/null 2>&1 || { echo "ERROR: workspace manifest 校验失败" >&2; exit 2; }
    "$PY" "$sc_py" --check >/dev/null 2>&1 || { echo "ERROR: source scope manifest 校验失败" >&2; exit 2; }
    MUTATION_WS_CHECK_BEFORE="true"
  else
    MUTATION_MANIFEST_BINDING="absent"
  fi
}

if [ "$MUTATION_REPORT_MODE" -eq 1 ]; then
  manifest_binding
  if [ -f "$MUT_REPO_ROOT/$MUTATION_SKIP_ALLOWLIST_PATH" ]; then
    MUTATION_SKIP_ALLOWLIST_SHA256="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MUT_REPO_ROOT/$MUTATION_SKIP_ALLOWLIST_PATH")"
    if ! printf '%s' "$MUTATION_SKIP_RES" | "$PY" -c 'import json,sys
from datetime import datetime, timezone
from pathlib import Path
allowlist = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
skips = [item for item in sys.stdin.read().split(";;") if item]
approved = allowlist.get("approved_skips", [])
for pattern in skips:
    matches = [item for item in approved if item.get("pattern") == pattern]
    if len(matches) != 1:
        raise ValueError("skip 正则未审批或重复: " + pattern)
    item = matches[0]
    if not item.get("approved_by") or item.get("approved_by") == "pending":
        raise ValueError("skip 正则缺审批人: " + pattern)
    if datetime.fromisoformat(item["expires_at"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise ValueError("skip 正则授权过期: " + pattern)' "$MUT_REPO_ROOT/$MUTATION_SKIP_ALLOWLIST_PATH" >/dev/null 2>&1; then
      echo "ERROR: MUTATION_SKIP_RES 未通过 allowlist 校验（${MUTATION_SKIP_ALLOWLIST_PATH}）" >&2
      exit 2
    fi
    MUTATION_SKIP_APPROVED="true"
  elif [ -n "$MUTATION_SKIP_RES" ]; then
    echo "[mut] 警告: MUTATION_SKIP_RES 非空但本仓无 $MUTATION_SKIP_ALLOWLIST_PATH —— 跳过硬编码在报告中标注 unapproved"
  fi
fi

# ---------- 目标/前缀解析（能力源自 su-memory-sdk） ----------
parse_pytest_command(){
  local parsed
  parsed="$(printf '%s' "$PYTEST" | "$PY" -c 'import shlex, sys
for token in shlex.split(sys.stdin.read()):
    print(token)')" || return 1
  pytest_command=()
  while IFS= read -r _t; do [ -n "$_t" ] && pytest_command+=("$_t"); done <<< "$parsed"
  [ "${#pytest_command[@]}" -gt 0 ]
}

# 逗号优先（每段一个目标，允许段内空格；带引号时再 shlex）；无逗号时 shlex 空白切分。
parse_selected_tests(){
  local parsed
  parsed="$(printf '%s' "$TESTS" | "$PY" -c 'import shlex, sys
raw = sys.stdin.read()
if "," in raw:
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "\"" in piece or chr(39) in piece:
            for token in shlex.split(piece):
                print(token)
        else:
            print(piece)
else:
    for token in shlex.split(raw):
        print(token)')" || return 1
  selected_tests=()
  while IFS= read -r _t; do [ -n "$_t" ] && selected_tests+=("$_t"); done <<< "$parsed"
  [ "${#selected_tests[@]}" -gt 0 ]
}

# 内层 pytest 固定附加 `-p no:cacheprovider` + 独立 PYCACHE 前缀：内层用例路径可能位于仓外，
# rootdir/cache 解析会落到仓外并触发 `.pytest_cache` 写入与路径解析抖动（A218 §9.12 登记）。
run_pytest(){
  MUTATION_RUN_ID=$((MUTATION_RUN_ID + 1))
  PYTHONPYCACHEPREFIX="$MUTATION_PYC_ROOT/$MUTATION_RUN_ID" PYTHONDONTWRITEBYTECODE=1 \
    "${pytest_command[@]}" -p no:cacheprovider -x -q --no-header "${selected_tests[@]}" >>"$MUT_LOG" 2>&1
}

echo "[mut] 基准测试..."
if ! parse_pytest_command; then echo "ERROR: pytest 前缀解析失败: $PYTEST" >&2; exit 2; fi
if ! parse_selected_tests; then echo "ERROR: 测试目标解析失败: $TESTS" >&2; exit 2; fi
printf '%s\n' "${selected_tests[@]}" > "$MUT_TARGETS_FILE"
if ! run_pytest; then
  echo "ERROR: 基准测试未通过,先修测试:"; tail -5 "$MUT_LOG" >&2; exit 2
fi
echo "[mut] 基准通过 ✓"

cp "$SRC" "$SRC.mutorig"

# ---------- 证据链接口（能力源自 su-memory-sdk，键名保持兼容） ----------
write_mutation_report(){
  [ "$MUTATION_REPORT_MODE" -eq 1 ] || return 0
  MUTATION_RUN_FINISHED_AT="${MUTATION_RUN_FINISHED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
  export MUTATION_ROOT="$MUT_REPO_ROOT" MUTATION_OUTPUT_PATH MUTATION_TARGET MUTATION_TESTS="$TESTS" \
    MUT_TARGETS_FILE MUTATION_TOOL="$MUTATION_TOOL_PATH" MUTATION_TOOL_SHA256 \
    MUTATION_TARGET_SHA256 MUTATION_RUN_STARTED_AT MUTATION_RUN_FINISHED_AT MUTATION_MIN \
    MUTATION_LOCK_PATH MUTATION_LOCK_MODE MUTATION_REPORT_MODE \
    MUTATION_SKIP_RES MUTATION_SKIP_ALLOWLIST_PATH MUTATION_SKIP_ALLOWLIST_SHA256 \
    MUTATION_SKIP_APPROVED MUTATION_MANIFEST_BINDING MUTATION_WS_CHECK_BEFORE MUTATION_WS_CHECK_AFTER \
    MUTATION_EVIDENCE_WORKSPACE \
    MUTATION_COMMAND="${MUTATION_COMMAND:-}"
  "$PY" - "$MUTATION_REPORT_PATH" "$1" "$2" "$3" "$4" "$5" <<'PYREPORT'
import hashlib, json, os, sys
from pathlib import Path

report_path, exit_code, passed, killed, survived, applied = sys.argv[1:7]
root = Path(os.environ["MUTATION_ROOT"]).resolve()
def rel(p):
    p = Path(p)
    return p if p.is_absolute() else root / p
def sha(p):
    f = rel(p)
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else ""
output_path = os.environ.get("MUTATION_OUTPUT_PATH", "")
output_sha256 = sha(output_path) if output_path else ""
target_rel = os.environ.get("MUTATION_TARGET", "")
target_file = rel(target_rel)
target_sha256 = hashlib.sha256(target_file.read_bytes()).hexdigest()
if target_sha256 != os.environ["MUTATION_TARGET_SHA256"]:
    raise SystemExit("ERROR: mutation target 哈希在运行过程中漂移: " + target_rel)
test_files = []
targets_file = os.environ.get("MUT_TARGETS_FILE", "")
if targets_file and Path(targets_file).exists():
    for raw_path in [x for x in Path(targets_file).read_text(encoding="utf-8").split("\n") if x]:
        if raw_path.startswith("-"):
            continue
        f = rel(raw_path)
        test_files.append({"path": raw_path, "sha256": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else ""})
score = int(killed) * 100 // int(applied) if int(applied) else None
bound = os.environ.get("MUTATION_MANIFEST_BINDING", "absent") == "bound"
workspace_fingerprint = ""
source_scope_fingerprint = ""
if bound:
    workspace = json.loads((root / os.environ.get("MUTATION_EVIDENCE_WORKSPACE", "docs/evidence/workspace-manifest.json")).read_text(encoding="utf-8"))
    source_scope = json.loads((root / "docs/evidence/source-scope-manifest.json").read_text(encoding="utf-8"))
    workspace_fingerprint = workspace["workspace_fingerprint"]
    source_scope_fingerprint = source_scope["scope_sha256"]
attestation = {
    "lock_mode": os.environ.get("MUTATION_LOCK_MODE", "disabled"),
    "lock_path": os.environ.get("MUTATION_LOCK_PATH", ""),
    "mode": "canonical" if os.environ.get("MUTATION_REPORT_MODE") == "1" else "informal",
    "manifest_binding": os.environ.get("MUTATION_MANIFEST_BINDING", "absent"),
    "output_hash_algorithm": "sha256",
    "skip_regexes": [item for item in os.environ.get("MUTATION_SKIP_RES", "").split(";;") if item],
    "skip_allowlist_approved": os.environ.get("MUTATION_SKIP_APPROVED", "false") == "true",
    "skip_allowlist_path": os.environ.get("MUTATION_SKIP_ALLOWLIST_PATH", ""),
    "skip_allowlist_sha256": os.environ.get("MUTATION_SKIP_ALLOWLIST_SHA256", ""),
    "source_scope_fingerprint": source_scope_fingerprint,
    "test_files": test_files,
    "workspace_check_after_pass": os.environ.get("MUTATION_WS_CHECK_AFTER") == "true" if bound else None,
    "workspace_check_before_pass": os.environ.get("MUTATION_WS_CHECK_BEFORE") == "true" if bound else None,
    "workspace_fingerprint": workspace_fingerprint,
}
report = {
    "applied_mutants": int(applied),
    "command": os.environ.get("MUTATION_COMMAND", ""),
    "executed": True,
    "exit_code": int(exit_code),
    "invalid_mutants": 0,
    "killed_mutants": int(killed),
    "minimum_score": int(os.environ.get("MUTATION_MIN", "80")),
    "output_path": output_path,
    "output_sha256": output_sha256,
    "passed": passed == "true",
    "run_finished_at": os.environ.get("MUTATION_RUN_FINISHED_AT", ""),
    "run_started_at": os.environ.get("MUTATION_RUN_STARTED_AT", ""),
    "schema_version": 2,
    "score": score,
    "seed": 0,
    "seed_mode": "deterministic-order",
    "source_scope_fingerprint": source_scope_fingerprint,
    "survived_mutants": int(survived),
    "target": target_rel,
    "target_sha256": os.environ["MUTATION_TARGET_SHA256"],
    "tests": os.environ.get("MUTATION_TESTS", ""),
    "timeout_mutants": 0,
    "tool": os.environ.get("MUTATION_TOOL", ""),
    "tool_sha256": os.environ.get("MUTATION_TOOL_SHA256", ""),
    "workspace_fingerprint": workspace_fingerprint,
    "attestation": attestation,
}
Path(report_path).write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PYREPORT
}

# 变异范围只取「本次改动」的行，避免历史大文件掩盖本次逻辑盲区；
# 无差异（CI 干净检出）或差异不含代码行时回退全文件，防 0 变异点让门禁空转。
# 行号口径（OODA-20261007-103）：apply_mut 变异的是**工作区磁盘文件**、基准与变异
# 后的测试也跑在该文件上，故行号必须取自 `git diff HEAD`（工作区 vs HEAD）。历史实现
# 优先取 `git diff --cached`，当索引停留在旧的 `git add` 快照（典型：pre-commit 失败后
# 改了代码但未重新 add）时行号整体错位，变异点全部落空 → 输出 MUTATION_SCORE=N/A 且被
# quality-gate 记 pass，属静默空转（改完代码不重新 add 等于门禁在验旧版）。
MUTATION_SCOPE="${MUTATION_SCOPE:-changed}"
CHANGED_LINES_FILE="$(mktemp)"
if [ "$MUTATION_SCOPE" = "all" ]; then
  echo "[mut] 变异范围: 全文件（显式 MUTATION_SCOPE=all）"
elif [ "$MUTATION_SCOPE" = "changed" ]; then
  SRC_ROOT="$(git -C "$(dirname "$SRC")" rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -z "$SRC_ROOT" ]; then
    echo "ERROR: changed 模式需要 Git 仓库，或设置 MUTATION_SCOPE=all" >&2
    exit 2
  fi
  SRC_REL="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$SRC" "$SRC_ROOT")"
  # M6(OODA-20261009-WAVE1-347): 波次/CI 模式下的行号口径。
  # 门禁在「干净检出」场景（CI 与 local-ci 都是）无 worktree diff → 旧逻辑回退全文件，
  # 于是「历史大文件里本次未改动的逻辑」的覆盖盲区被记成本次分数（实测 backend/utils.py 0%、
  # finance_modules_api.py 10%，而本波次真正新增的代码 100%）→ 属假红，且让波次门禁不可达。
  # 口径: 调用方（quality-gate 由 local-ci/CI 传入 GATE_DIFF_BASE）显式给出基线时，
  # 行号取 `基线..HEAD` 的 diff —— 与 complexity 检查「只查改动函数」同源，
  # 度量的是**本次交付自己的改动行**。未给出基线时行为不变（worktree diff + 全文件兜底）。
  if [ -n "${MUTATION_DIFF_BASE:-}" ]; then
    if ! git -C "$SRC_ROOT" rev-parse --verify --quiet "${MUTATION_DIFF_BASE}^{commit}" >/dev/null; then
      echo "ERROR: MUTATION_DIFF_BASE 不是有效提交: $MUTATION_DIFF_BASE" >&2
      exit 2
    fi
    git -C "$SRC_ROOT" diff "${MUTATION_DIFF_BASE}..HEAD" -U0 -- "$SRC_REL" > "$CHANGED_LINES_FILE.diff" || {
      echo "ERROR: 无法读取波次 diff（$MUTATION_DIFF_BASE..HEAD）" >&2
      exit 2
    }
    echo "[mut] 变异范围基线: ${MUTATION_DIFF_BASE}..HEAD（波次/CI 模式，行号按本次交付的改动行取）"
  else
  git -C "$SRC_ROOT" diff HEAD -U0 -- "$SRC_REL" > "$CHANGED_LINES_FILE.diff" || {
    echo "ERROR: 无法读取 worktree diff" >&2
    exit 2
  }
  fi
  # 索引与工作区不一致（staged ≠ 磁盘）时显式告警：本次提交内容 ≠ 实际被变异/测试的文件。
  # 此时门禁结果虽为真，但验的不是将要入库的版本，提交前必须重新 `git add`。
  git -C "$SRC_ROOT" diff --cached -U0 -- "$SRC_REL" > "$CHANGED_LINES_FILE.cached.diff" 2>/dev/null || true
  if [ -s "$CHANGED_LINES_FILE.cached.diff" ] \
     && ! cmp -s "$CHANGED_LINES_FILE.cached.diff" "$CHANGED_LINES_FILE.diff"; then
    echo "[mut] 警告: 索引与工作区不一致（staged ≠ 磁盘）——行号已按工作区文件取；提交前请重新 git add"
  fi
  "$PY" - "$CHANGED_LINES_FILE.diff" "$CHANGED_LINES_FILE" <<'PYLINES'
import re, sys
changed=set()
for line in open(sys.argv[1], encoding="utf-8"):
    m=re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
    if not m:
        continue
    start=int(m.group(1)); count=int(m.group(2) or 1)
    changed.update(range(start, start+count))
if not changed:
    sys.exit(0)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, sorted(changed))))
PYLINES
  if [ -s "$CHANGED_LINES_FILE.diff" ] && [ ! -s "$CHANGED_LINES_FILE" ]; then
    # 纯删除变更（diff 非空，但所有 hunk 的新文件侧计数为 0）：新文件里没有可变异
    # 的新增行 → 合法 N/A，**不得**回退全文件。
    # 口径（OODA-20261007-124）：回退全文件会把「本次未改动的历史代码」的覆盖盲区
    # 记成这次变更的分数——基准测试明明通过、日志却报「测试套件有覆盖盲区」，属
    # 治理报告里的假结论（假红），也与 104「N/A 是合法结果」的口径相冲突。
    # 删除类变更的有效性由「基准测试仍通过」（上方已跑）+ 架构/回归门禁承担。
    echo "[mut] 变异范围: 纯删除变更（新文件侧无可变异行）"
    echo "[mut] 说明: 删除类变更无新增可执行行 → N/A（基准测试已通过）；非『行号未命中』"
    # M5(OODA-20261007-053): N/A 必须携带机器可读原因码，供 quality-gate 判定
    # 「合法无可评变异」与「静默放行」；人工确认通道据此可追溯（不再依赖中文散文解析）。
    echo "MUTATION_NA_REASON=pure-delete"
    echo "MUTATION_NA_LINES=0"
    echo "MUTATION_SCORE=N/A"
    restore_src
    MUTATION_RUN_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_mutation_report 0 true 0 0 0
    exit 0
  fi
  if [ -s "$CHANGED_LINES_FILE" ]; then
    # 注意: 单行改动集无结尾换行, wc -l 会少算 1 行, 故用非空行计数
    echo "[mut] 变异范围: 改动行 $(tr -s ' \n' '\n' < "$CHANGED_LINES_FILE" | grep -c .) 行"
  else
    # 无差异（如 CI 干净检出）或差异不含代码行：删除白名单文件 →
    # apply_mut 以 allowed=None 走全文件，避免 MUTATION_SCORE=N/A 的静默空转
    rm -f "$CHANGED_LINES_FILE"
    echo "[mut] 变异范围: 无差异 → 回退全文件"
  fi
else
  echo "ERROR: MUTATION_SCOPE 仅支持 changed/all" >&2
  exit 2
fi

# 变异规则: 描述|正则|替换。用 python 应用(单点替换第1处匹配)
# 跳过区间: 字符串/注释(tokenize) + 类型注解区域(ast) + 返回箭头 ->。
# 普通规则与整数+1 共用同一段跳过/匹配逻辑, 避免两套偏移算法漂移。
apply_mut(){
  # 第 5 参：变异点行号落盘文件（OODA-20261009-WAVE1-348）。
  # 旧输出只给「存活: 布尔 True→False」这类描述、不给行号 → 存活点不可直接定位，只能人工反查
  # （本波次实测：9 个存活点里 6 个需反查）。加行号后存活点可直接跳到源行。
  "$PY" - "$SRC" "$2" "$3" "$CHANGED_LINES_FILE" "$CHANGED_LINES_FILE.point" <<'PYMUT'
import os, sys, re, tokenize, io, ast
path, pat, rep, lines_file, point_file = (sys.argv[1], sys.argv[2], sys.argv[3],
                                          sys.argv[4], sys.argv[5])
code = open(path, encoding="utf-8").read()
MUTATION_SKIP_RES = os.environ.get("MUTATION_SKIP_RES", "")


def line_at(code, pos):
    start = code.rfind("\n", 0, pos) + 1
    end = code.find("\n", pos)
    return code[start:] if end < 0 else code[start:end]


def in_line_skip(pos):
    if not MUTATION_SKIP_RES:
        return False
    line = line_at(code, pos)
    return any(re.search(p, line) for p in MUTATION_SKIP_RES.split(";;") if p)
try:
    _raw = open(lines_file, encoding="utf-8").read().split()
except FileNotFoundError:
    allowed = None
else:
    # 空文件（MUTATION_SCOPE=all / 无改动行）表示"不做行过滤"，必须为 None 而非空集合，
    # 否则所有变异点都会被误跳过，门禁静默空转成 MUTATION_SCORE=N/A
    allowed = {int(x) for x in _raw} if _raw else None


def line_allowed(pos):
    return allowed is None or code.count("\n", 0, pos) + 1 in allowed

def get_skip_offsets(src):
    skips = []
    lines = src.splitlines(True)
    def off(line, col):  # 1-based line, 0-based char col (tokenize 口径)
        return sum(len(l) for l in lines[:line - 1]) + col
    def off_b(line, bcol):  # ast 口径: 0-based UTF-8 字节列 → 字符偏移
        return sum(len(l) for l in lines[:line - 1]) \
            + len(lines[line - 1].encode("utf-8")[:bcol].decode("utf-8"))
    def span(node):
        return (off_b(node.lineno, node.col_offset),
                off_b(node.end_lineno, node.end_col_offset))
    # 1) 字符串与注释
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                skips.append((off(*tok.start), off(*tok.end)))
        toks = None
    except tokenize.TokenError:
        toks = None
    # 1b) f-string 整体（Python ≥3.12 拆成 FSTRING_START/MIDDLE/END）：
    #     只匹配 STRING 会漏掉 f-string 字面量里的运算符文本，导致 skip 口径随解释器版本漂移
    fstart = getattr(tokenize, "FSTRING_START", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    if fstart is not None and fend is not None:
        stack = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == fstart:
                    stack.append(tok)
                elif tok.type == fend and stack:
                    start = stack.pop()
                    if not stack:
                        skips.append((off(*start.start), off(*tok.end)))
        except tokenize.TokenError:
            pass
    # 2) 返回箭头 ->：CPython tokenize 将 -> 发为单个 OP token;
    #    兼容个别实现拆成相邻 '-' '>' 的情况。该序列仅出现于注解语法。
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        toks = []
    for i, a in enumerate(toks):
        if a.type == tokenize.OP and a.string == "->":
            skips.append((off(*a.start), off(*a.end)))
        elif (a.type == tokenize.OP and a.string == "-"
                and i + 1 < len(toks) and toks[i + 1].type == tokenize.OP
                and toks[i + 1].string == ">"
                and toks[i + 1].start[0] == a.start[0]
                and toks[i + 1].start[1] == a.end[1]):
            skips.append((off(*a.start), off(*toks[i + 1].end)))
    # 3) 类型注解区域: 参数注解/返回注解/变量注解（ast 节点跨度, 支持多行与 UTF-8）
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in (list(getattr(args, "posonlyargs", [])) + list(args.args)
                          + list(getattr(args, "kwonlyargs", []))
                          + [x for x in (args.vararg, args.kwarg) if x is not None]):
                    if a.annotation is not None:
                        skips.append(span(a.annotation))
                if node.returns is not None:
                    skips.append(span(node.returns))
            elif isinstance(node, ast.AnnAssign):
                skips.append(span(node.annotation))
            elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                skips.append(span(node))
    return skips

def in_skip(pos, skips):
    for s, e in skips:
        if s <= pos < e:
            return True
    return False

skips = get_skip_offsets(code)
try:
    rx = re.compile(pat)
except re.error:
    idx = code.find(pat)
    while idx != -1 and (in_skip(idx, skips) or not line_allowed(idx)
                       or in_line_skip(idx)):
        idx = code.find(pat, idx + 1)
    if idx == -1:
        sys.exit(2)
    mut_pos = idx
    new = code[:idx] + rep + code[idx + len(pat):]
else:
    m = rx.search(code)
    while m and (in_skip(m.start(), skips) or not line_allowed(m.start())
               or in_line_skip(m.start())):
        m = rx.search(code, m.end())
    if not m:
        sys.exit(2)
    if rep == "__inc__":
        sub = f"({int(m.group(1))}+1)"
    else:
        try:
            sub = m.expand(rep)
        except (re.error, ValueError):
            sub = rep
    mut_pos = m.start()
    new = code[:m.start()] + sub + code[m.end():]
if new == code:
    sys.exit(2)
# 语法破坏型变异（如 -> 变 -<）不构成"测试有效性"的证据，按跳过处理（退出码 3）
try:
    ast.parse(new)
except SyntaxError:
    sys.exit(3)
# 变异点行号：1-based，供调用方在存活/杀死明细里直接定位源行
with open(point_file, "w", encoding="utf-8") as fh:
    fh.write(str(code.count("\n", 0, mut_pos) + 1))
open(path, "w", encoding="utf-8").write(new)
PYMUT
}

MUTS=(
  "比较 == 变 !=|==|!="
  "比较 != 变 ==|!=|=="
  "比较 >= 变 >|>=|>"
  "比较 <= 变 <|<=|<"
  "比较 > 变 <|[^<>=!]>(?!=)|<"
  "比较 < 变 >|[^<>=!]<(?!=)|>"
  "布尔 True→False|True|False"
  "布尔 False→True|False|True"
  "None 变 0|\\bNone\\b|0"
  "数值常量 +1 (整数)|\\b(\\d+)\\b|__inc__"
)

killed=0; survived=0; applied=0; skipped=0; surv_list=""
for m in "${MUTS[@]}"; do
  desc="${m%%|*}"; rest="${m#*|}"; pat="${rest%%|*}"; rep="${rest##*|}"
  cp "$SRC.mutorig" "$SRC"
  # 数值+1 与普通规则共用 apply_mut（rep=__inc__ 时替换为 (n+1)）, 跳过逻辑单一定义
  apply_mut "$desc" "$pat" "$rep"; rc=$?
  if [ "$rc" -eq 3 ]; then skipped=$((skipped+1)); continue; fi
  [ "$rc" -ne 0 ] && continue
  diff -q "$SRC.mutorig" "$SRC" >/dev/null 2>&1 && continue
  applied=$((applied+1))
  mut_line="$(cat "$CHANGED_LINES_FILE.point" 2>/dev/null || echo "?")"
  echo "[mut] 应用: ${desc}（行 ${mut_line}）"
  echo "[mut] 应用: ${desc}（行 ${mut_line}）" >> "$MUT_LOG"
  if run_pytest; then
    survived=$((survived+1)); surv_list="$surv_list\n  - 存活: $desc (行 $mut_line)"
  else
    killed=$((killed+1))
  fi
done

if [ "$applied" -eq 0 ]; then
  # N/A 语义显式化（OODA-20261007-103②）：「改动行内没有可定位的变异算子」与
  # 「未定位到变异点（行号口径错误）」是两件事，不能共用同一个绿。前者不阻断，
  # 后者自 2026-10-07 起由上面的「索引/工作区不一致」告警 + 工作区行号口径消除。
  echo "[mut] 未产生有效变异"
  echo "[mut] 说明: 改动行内无 fitting 变异算子（0 个候选），本条无可变逻辑；非『行号未命中』"
  # M5(OODA-20261007-053): 同 pure-delete，原因码 + 变异范围行数一并输出。
  # 量化（2026-10-09）：受约束文件 26/300 任何改动必落 N/A；近 40 提交 15 个改动单元
  # 中 5 个落 N/A → 该通道是常态路径，不是边角，必须显式可追溯。
  echo "MUTATION_NA_REASON=no-operator-in-changed-lines"
  if [ -s "$CHANGED_LINES_FILE" ]; then
    echo "MUTATION_NA_LINES=$(tr -s ' \n' '\n' < "$CHANGED_LINES_FILE" | grep -c .)"
  else
    echo "MUTATION_NA_LINES=all"
  fi
  echo "MUTATION_SCORE=N/A"
  restore_src
  MUTATION_RUN_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_mutation_report 0 true 0 0 0
  exit 0
fi
score=$((killed * 100 / applied))
echo "[mut] 杀死 $killed / 存活 $survived / 有效变异 $applied / 语法破坏跳过 $skipped"
echo "[mut] 变异分数: ${score}%"
[ -n "$surv_list" ] && { echo "[mut] 存活变异(测试未捕获):"; printf "$surv_list\n"; }
echo "MUTATION_SCORE=$score"
restore_src
MUTATION_RUN_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$MUTATION_REPORT_MODE" -eq 1 ] && [ "$MUTATION_MANIFEST_BINDING" = "bound" ]; then
  if ! "$PY" "$MUT_REPO_ROOT/scripts/ai-verify/workspace-manifest.py" --check >/dev/null 2>&1 \
     || ! "$PY" "$MUT_REPO_ROOT/scripts/ai-verify/source-scope-manifest.py" --check >/dev/null 2>&1; then
    echo "ERROR: 运行后 manifest 校验失败" >&2
    MUTATION_WS_CHECK_AFTER="false"
    write_mutation_report 2 false "$killed" "$survived" "$applied"
    exit 2
  fi
  MUTATION_WS_CHECK_AFTER="true"
fi
if [ "$score" -ge "$MIN" ]; then
  echo "[mut] ✅ 达标(≥${MIN}%)"
  write_mutation_report 0 true "$killed" "$survived" "$applied"
  exit 0
else
  echo "[mut] ❌ 不达标(<${MIN}%), 测试套件有覆盖盲区"
  write_mutation_report 1 false "$killed" "$survived" "$applied"
  exit 1
fi
