#!/usr/bin/env bash
# AI 代码质量门禁（按风险等级差异化阈值）
# 用法: quality-gate.sh <L0|L1|L2> [file1 file2 ...]   不传文件则扫暂存改动
# 退出码: 0=通过  1=失败  2=缺少工具(跳过,警告不阻断)
set -o pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

LEVEL="${1:-L2}"; shift || true
case "${LEVEL}" in L0|L1|L2) ;; *) LEVEL=L2;; esac

REPORT_DIR=".ai-governance/reports"
# 门禁临时文件目录：本机可能同时运行多个仓库的 governance 作业（共用 _ai-eng-kit），
# 固定 /tmp 路径会被并发写坏（2026-10-02 实证：覆盖率 JSON 出现 "Extra data" 解析失败 → 假红）。
# CI 传 runner.temp 获得每作业唯一目录；本地默认 /tmp 保持兼容（自测脚本仍写 /tmp/qgate_cov.json）。
GATE_TMP_DIR="${QGATE_TMP_DIR:-/tmp}"
mkdir -p "$GATE_TMP_DIR"
COV_JSON="$GATE_TMP_DIR/qgate_cov.json"
PYTEST_LOG="$GATE_TMP_DIR/qgate_pytest.log"
MUT_LOG="$GATE_TMP_DIR/qgate_mut.log"
# 工具路径：优先环境变量，再 PATH。支持 venv（如 RADON_BIN=backend/.venv/bin/radon）
RADON_BIN="${RADON_BIN:-radon}"
RUFF_BIN="${RUFF_BIN:-ruff}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
# U1（2026-10-15）工具运行时口径 —— 不变式 I-tools：pytest 与 python 必须同属一个解释器环境。
# 判据是「同 bin 目录」，**不是**「字符串相等」：同一 venv 的 pytest 与 python 必然不等，
# 用字符串相等会把自己人当外人拦掉（实证 su-memory-sdk 20261015-170802-gate-L2.md ——
# 该仓以 PYTEST_BIN 表示「运行 pytest 的解释器」，与 kit 的「pytest 可执行文件」同名异义）。
# 只设了其中一个时**对称派生**另一个，避免「半套注入」自己制造不一致。
if [ -z "${PYTHON_BIN:-}" ]; then
  case "${PYTEST_BIN:-}" in
    */*) _pbd="$(dirname "$PYTEST_BIN")"
         for _c in python python3; do
           [ -x "$_pbd/$_c" ] && { PYTHON_BIN="$_pbd/$_c"; break; }
         done ;;
  esac
fi
BANDIT_BIN="${BANDIT_BIN:-bandit}"
# JS 静态检查：ESLint 走「治理配置」（零依赖核心规则），避免依赖项目本地 npm 配置
# 背景：纯 JS 的 L2 改动此前因缺 eslint/配置必然 fail-closed（台账 OODA-20261003-002）
ESLINT_BIN="${ESLINT_BIN:-eslint}"
ESLINT_GOV_CONFIG="${ESLINT_GOV_CONFIG:-$ROOT/.ai-governance/eslint.governance.config.mjs}"
has(){ command -v "$1" >/dev/null 2>&1; }
mkdir -p "$REPORT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
REPORT="$REPORT_DIR/$TS-gate-${LEVEL}.md"

if [ "$#" -gt 0 ]; then FILES=("$@"); else
  FILES=()
  # F-26：NUL 分隔枚举，避免非 ASCII 路径被 git 引号化后与磁盘不匹配
  # （详见 risk-classify.sh 的注释）
  while IFS= read -r -d '' l; do FILES+=("$l"); done < <(
    git diff --cached --name-only -z --diff-filter=ACMR 2>/dev/null | sort -zu)
fi
[ "${#FILES[@]}" -eq 0 ] && FILES=(".")

say(){ printf '\033[1m[gate]\033[0m %s\n' "$1"; }

# 差异化阈值表
case "${LEVEL}" in
  L0) COV_CORE=85; COV_BR=80; COV_UTIL=70;  CC_MAX=15; DUP=5;;
  L1) COV_CORE=90; COV_BR=85; COV_UTIL=80;  CC_MAX=15; DUP=5;;
  L2) COV_CORE=95; COV_BR=90; COV_UTIL=85;  CC_MAX=15; DUP=5;;
esac
# F-20(P0-F) 回灌（2026-10-08，源自 mci-world-model 窗口）：阈值必须可按仓库实测基线显式校准，
# 否则无法落地的阈值会被整体绕过。口径: 阈值 = 实测基线-5%，只许上调；未注入时沿用等级默认值。
# 优先级：等级默认值 < env（COV_CORE_THRESHOLD/COV_BR_THRESHOLD）< 仓库 .ai-coverage-threshold 文件。
COV_CORE="${COV_CORE_THRESHOLD:-$COV_CORE}"
COV_BR="${COV_BR_THRESHOLD:-$COV_BR}"

{
  echo "# 质量门禁报告"
  echo "- 时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- 风险等级: ${LEVEL}"
  echo "- 阈值: 核心覆盖≥${COV_CORE}% 分支≥${COV_BR}% 工具≥${COV_UTIL}% 圈复杂度≤${CC_MAX} 重复≤${DUP}%"
  echo "- 文件: ${FILES[*]}"
  echo
} > "$REPORT"

FAIL=0
WARN=0
pass(){ echo "- ✅ $1" | tee -a "$REPORT"; }
fail(){ echo "- ❌ $1" | tee -a "$REPORT" >&2; FAIL=1; }
warn(){ echo "- ⚠️  $1" | tee -a "$REPORT"; WARN=1; }

# pytest 失败诊断：CI 上 pytest 输出被重定向到 $PYTEST_LOG，若只报「pytest 失败」
# 则完全无法定位失败用例（2026-10-08 实证：governance 门禁连续假盲）。
# 这里把失败用例名与日志尾部带到 stdout，并区分「用例失败」与「覆盖率未达阈值」。
# （2026-10-10 自 nutrition-system-hospital 回灌，台账 OODA-20261009-077）
pytest_diag(){
  [ -f "$PYTEST_LOG" ] || { warn "pytest 日志缺失: $PYTEST_LOG"; return; }
  local n
  n=$(grep -cE '^(FAILED|ERROR) ' "$PYTEST_LOG" 2>/dev/null || true)
  n="${n:-0}"
  echo "  ↳ 失败/错误用例 ${n} 条（完整日志 ${PYTEST_LOG}）"
  [ "$n" -gt 0 ] && grep -E '^(FAILED|ERROR) ' "$PYTEST_LOG" 2>/dev/null | head -30 | sed 's/^/    /'
  [ "$n" -gt 30 ] && echo "    ...（其余 $((n - 30)) 条见日志）"
  if grep -qE 'Required test coverage of .* not reached' "$PYTEST_LOG" 2>/dev/null; then
    echo "  ↳ 判定: 用例全通过但覆盖率未达 --cov-fail-under 阈值"
  elif [ "$n" -gt 0 ]; then
    echo "  ↳ 判定: 存在失败/错误用例（见上，逐个修复）"
  fi
  echo "  ↳ 日志尾部:"
  tail -n 15 "$PYTEST_LOG" 2>/dev/null | sed 's/^/    /'
}

# ---------- F-1(P0-B): 工具完整性校验, 伪造二进制 fail-closed ----------
verify_tool_bin(){
  local name="$1"
  local bin="${!name:-}"
  [ -z "$bin" ] && return 0
  case "$bin" in
    true|false|*/true|*/false) fail "$name 指向可疑二进制($bin), 疑似门禁绕过"; return;;
  esac
  if [[ "$bin" != */* ]]; then
    local resolved
    resolved="$(command -v -- "$bin" 2>/dev/null || true)"
    if [ -z "$resolved" ]; then
      fail "$name 不存在或不可执行: $bin"
      return
    fi
    bin="$resolved"
  fi
  if [ ! -x "$bin" ]; then fail "$name 不存在或不可执行: $bin"; return; fi
  # M6c 回环修复（2026-10-15，CI run 38697626731）：kit 版直接在工作区根探测
  # `--version`，在 CI 上 radon 探测为空 → 误判「疑似伪造二进制」fail-closed。
  # 恢复本仓既有实现：① 在临时目录内探测（隔离 cwd 依赖）；② 不用管道 + head
  # （避免 pipefail/SIGPIPE 交互），改由 bash 截首行。
  local probe_dir v
  probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/tool-probe.XXXXXX")"
  v="$(cd "$probe_dir" && "$bin" --version 2>/dev/null)"
  rm -rf "$probe_dir"
  v="${v%%$'\n'*}"
  [ -z "$v" ] && fail "$name 无法输出 --version, 疑似伪造二进制: $bin"
}
verify_tool_bin RADON_BIN
verify_tool_bin RUFF_BIN
verify_tool_bin BANDIT_BIN
verify_tool_bin PYTEST_BIN

# U1 不变式校验（同环境）：仅当两侧都是**显式路径**时才可判定归属
# （PATH 上的裸命令名无从判定环境，不误报）。
runtime_env_conflict(){
  local _pb="$1" _py="$2" _a _b
  case "$_pb" in */*) ;; *) return 1;; esac
  case "$_py" in */*) ;; *) return 1;; esac
  _a="$(cd "$(dirname "$_pb")" 2>/dev/null && pwd -P)" || return 1
  _b="$(cd "$(dirname "$_py")" 2>/dev/null && pwd -P)" || return 1
  [ "$_a" != "$_b" ]
}
if [ -n "${PYTHON_BIN:-}" ] && runtime_env_conflict "$PYTEST_BIN" "$PYTHON_BIN"; then
  fail "工具运行时不一致: pytest 与 python 不同环境（PYTEST_BIN=${PYTEST_BIN} / PYTHON_BIN=${PYTHON_BIN}）"
  fail "  请让两者同属一个 venv：或只设 PYTHON_BIN，或只设 PYTEST_BIN（门禁会派生同环境伙伴）"
fi
tool_missing(){
  if [ "$LEVEL" = "L2" ]; then fail "$1（L2 fail-closed）"; else warn "$1"; fi
}
if [ "$FAIL" -eq 1 ]; then
  say "工具完整性校验未通过, fail-closed（伪造工具即视为门禁绕过）"
  echo "**结论：❌ 未通过（工具完整性校验）**" >> "$REPORT"
  exit 1
fi

# ---------- 1. 基础静态门禁（无外部工具，必跑） ----------
# 1a. 空 except / 空 catch
py_files=(); js_files=()
for f in "${FILES[@]}"; do
  case "$f" in
    *.py) py_files+=("$f");;
    *.js|*.ts|*.tsx) js_files+=("$f");;
  esac
done

# 治理 ESLint 只处理纯 JS（.ts/.tsx 需 TS 解析器，交由项目自身 lint 流程）
js_lint_files=()
for f in "${js_files[@]}"; do
  case "$f" in *.js|*.cjs|*.mjs) js_lint_files+=("$f");; esac
done

if [ "${#py_files[@]}" -gt 0 ]; then
  # 用 AST 精确检测空 except（body 仅 pass/.../docstring），避免跨行误判
  python3 - "${py_files[@]}" <<'PYAST' 2>/dev/null && true
import ast, sys
hit=[]
for p in sys.argv[1:]:
    try: tree=ast.parse(open(p,encoding="utf-8").read(), filename=p)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            b=n.body
            # 跳过纯 docstring
            body=[x for x in b if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant))]
            if not body or all(isinstance(x,(ast.Pass,)) for x in body):
                hit.append(f"{p}:{n.lineno}: 空 except (仅 pass/docstring)")
    # pass-through handlers (Ellipsis ...)
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant))]
            if body and all(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and x.value.value is ... for x in body):
                hit.append(f"{p}:{n.lineno}: 空 except (仅 ...)")
if hit:
    print("\n".join(hit[:10])); sys.exit(1)
PYAST
  rc=$?
  if [ "$rc" -ne 0 ]; then fail "检测到空 except（Python，AST 精确匹配）"; fi

  # 1a2. R5 会话泄漏门禁：函数内直接调用 SessionLocal() 却无 finally-close 且未用 with（database.py 工厂豁免）
  python3 - "${py_files[@]}" <<'PYSESS' 2>/dev/null && true
import ast, sys, os
hit=[]
for p in sys.argv[1:]:
    if os.path.basename(p) == "database.py":
        continue  # db_session_scope/get_db 工厂定义地
    try:
        tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
    except Exception:
        continue
    # F-10(P1-E): 收集 SessionLocal 的 import 别名, 防别名绕过（from database import SessionLocal as SL）
    aliases = {"SessionLocal"}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "database" in n.module:
            for a in n.names:
                if a.name == "SessionLocal":
                    aliases.add(a.asname or a.name)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", "") in aliases]
        if not calls:
            continue
        # 工厂豁免：所有 SessionLocal() 调用都直接 return（生命周期归属调用方，同 get_db）
        if all(any(isinstance(x, ast.Return) and x.value is c for x in ast.walk(fn)) for c in calls):
            continue
        has_with = any(isinstance(n, ast.With) for n in ast.walk(fn))
        def _has_close(stmts):
            return any(isinstance(x, ast.Call) and getattr(x.func, "attr", "") == "close"
                       for s in stmts for x in ast.walk(s))
        has_finally_close = any(
            isinstance(n, ast.Try) and _has_close(n.finalbody)
            for n in ast.walk(fn))
        if not has_with and not has_finally_close:
            hit.append(f"{p}:{calls[0].lineno}: SessionLocal() 无 finally-close/with，会话泄漏风险（请用 db_session_scope()）")
if hit:
    print("\n".join(hit[:10])); sys.exit(1)
PYSESS
  rc=$?
  if [ "$rc" -ne 0 ]; then fail "检测到会话泄漏模式（R5 门禁）"; fi
fi
if [ "${#js_files[@]}" -gt 0 ]; then
  if grep -lnE 'catch\s*\([^)]*\)\s*\{\s*\}' "${js_files[@]}" 2>/dev/null \
     || grep -lnE 'catch\s*\([^)]*\)\s*\{\s*/\*[^*]*\*/\s*\}' "${js_files[@]}" 2>/dev/null \
     || grep -lnE 'catch\s*\([^)]*\)\s*\{\s*//[^\r\n]*\r?\n\s*\}' "${js_files[@]}" 2>/dev/null; then
    fail "检测到空 catch（JS/TS）"; fi
fi

# 1b. f-string 拼接 SQL（F-10/P1-E: AST 化——覆盖跨行 f-string 与大小写变体）
python3 - "${py_files[@]}" <<'PYSQL' 2>/dev/null && true
import ast, sys, re
pat = re.compile(r"\b(select|insert|update|delete|from|where|join)\b", re.I)
hit = []
for p in sys.argv[1:]:
    if any(x in p for x in ("_test", ".test.", "/tests/", "/test/")):
        continue
    try:
        tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
    except Exception:
        continue
    seen = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.JoinedStr) or id(n) in seen:
            continue
        seen.add(id(n))
        text = "".join(v.value for v in n.values
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
        if pat.search(text):
            hit.append(f"{p}:{n.lineno}: f-string 拼接 SQL")
if hit:
    print("\n".join(hit[:10])); sys.exit(1)
PYSQL
rc=$?
[ "$rc" -ne 0 ] && fail "疑似 f-string 拼接 SQL，必须参数化"

# 1b2. nosec 豁免闭环（F-14/P2-A: 必须 TICKET + 登记路径 + 登记人）
if [ "${#py_files[@]}" -gt 0 ]; then
  python3 - "${py_files[@]}" <<'PYNOSEC' 2>/dev/null && true
import os, re, sys
ticket_re = re.compile(r"#\s*nosec\s*:\s*(TICKET-[A-Za-z0-9._-]+)\b", re.I)
loose_re = re.compile(r"#\s*nosec\b", re.I)
allowfile = os.path.join(os.getcwd(), ".ai-governance/nosec-allowlist.tsv")
registered = {}
try:
    with open(allowfile, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and all(parts[:4]):
                registered[parts[0].upper()] = parts[1]
except FileNotFoundError:
    pass
hit = []
for p in sys.argv[1:]:
    rel = os.path.relpath(os.path.abspath(p), os.getcwd())
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not loose_re.search(line):
                continue
            m = ticket_re.search(line)
            if not m:
                hit.append(f"{rel}:{lineno}: 裸 nosec（必须写成 # nosec: TICKET-xxx）")
            elif m.group(1).upper() not in registered:
                hit.append(f"{rel}:{lineno}: nosec {m.group(1)} 未登记")
            elif registered[m.group(1).upper()] != rel:
                hit.append(f"{rel}:{lineno}: nosec {m.group(1)} 登记路径不符")
if hit:
    print("\n".join(hit[:10])); sys.exit(1)
PYNOSEC
  rc=$?
  if [ "$rc" -ne 0 ]; then
    fail "nosec 豁免未闭环（登记: .ai-governance/nosec-allowlist.tsv，字段=TICKET/路径/登记人/原因）"
  fi
fi

# 1c. 硬编码密钥 / 明文敏感字段
if grep -rnE '(password|secret|apikey|api_key|token)\s*=\s*["\x27][^"\x27]{6,}' "${FILES[@]}" \
  --include='*.py' --include='*.js' --include='*.ts' 2>/dev/null | grep -v 'env\|config\|getenv\|os\.\|test\|conftest\|fake\|Test\|fixture' | head -5; then
  fail "疑似硬编码密钥/敏感字段"; fi

# 1d. 裸 print 进生产（仅 src 层）
if grep -rn '^\s*print(' "${FILES[@]}" --include='*.py' 2>/dev/null \
  | grep -v '_test\|/tests/\|scripts/' | head -5; then
  warn "检测到 print（生产应用 logging）"; fi

# ---------- 2. 圈复杂度（按栈选工具） ----------
check_cc() {
  if [ "${#py_files[@]}" -eq 0 ] && [ "${#js_files[@]}" -eq 0 ]; then
    pass "圈复杂度检查无目标文件"
    return
  fi
  if has "$RADON_BIN" && [ "${#py_files[@]}" -gt 0 ]; then
    # 拿改动行（暂存区 -U0），与 radon 函数行做交集，只判改动函数复杂度
    python3 - "$RADON_BIN" "$CC_MAX" "${py_files[@]}" <<'PYCC' 2>/dev/null
import ast, subprocess, sys, os
radon, cc_max = sys.argv[1], int(sys.argv[2])
files = sys.argv[3:]
norm=lambda p: os.path.relpath(os.path.abspath(p), os.getcwd())
# 1. 每个文件的改动行集合(暂存区 -U0)
in_git = subprocess.run(["git","rev-parse","--is-inside-work-tree"],
                        capture_output=True,text=True).returncode==0
# M6(OODA-20261009-WAVE1-347): 门禁的改动行口径必须认「显式基线」。
# local-ci.sh 早已 export GATE_DIFF_BASE 并在报告里声明「质量门禁选集: 波次基线 X..HEAD」，
# 但本脚本此前从未读取它 → 干净检出（CI 与 local-ci 都是）下 `git diff --cached` 恒为空 →
# 回退**全文件**扫描，「只判改动函数」的既定口径失效，历史大文件的存量高复杂函数被记成本次问题
# （实测：同一波次 9 处假红，逐函数核对改动行命中数均为 0）。现按显式基线取改动行；
# 未给基线时行为不变（仍是暂存区 + 全文件兜底）。
_base = os.environ.get("GATE_DIFF_BASE", "").strip()
changed={}
for pf in files:
    ch=set(); saw_hunk=False
    if in_git:
        if _base:
            r=subprocess.run(["git","diff","-U0",_base+"..HEAD","--",pf],
                             capture_output=True,text=True)
        else:
            r=subprocess.run(["git","diff","--cached","-U0","--",pf],
                             capture_output=True,text=True)
        for ln in r.stdout.splitlines():
            if "@@" in ln: saw_hunk=True
            # @@ -a,b +c,d @@ : 改动后行从 c 起 b 行(省略=1)
            m=__import__("re").search(r"\+(\d+)(?:,(\d+))?", ln.split("@@")[-2] if "@@" in ln else "")
            if not m: continue
            c=int(m.group(1)); n=int(m.group(2) or 1)
            ch.update(range(c,c+n))
    # 见过 hunk 但无改动行 = 纯删除 → 空集(无函数被改动,不查)；
    # 连 hunk 都没解析到 = 异常/非git → None 全查兜底。
    # （原逻辑 ch 空集即 None 全查，纯删除会误报全文件史前高复杂函数）
    changed[norm(pf)] = (ch if (ch or saw_hunk) else None)
# 2. radon 复杂度
out=subprocess.run([radon,"cc","-s"]+files,capture_output=True,text=True).stdout
cc_map={}   # (file, funcname) -> cc
cur=None
import re as _re
for ln in out.splitlines():
    if not ln.startswith(" ") and ln.endswith(".py"):
        cur=norm(ln); continue
    m=_re.match(r"\s*[FMC]\s+(\d+):\d+\s+(.+?)\s+-\s+[A-F]\s+\((\d+)\)",ln)
    if m and cur: cc_map[(cur,m.group(2))]=int(m.group(3))
# 3. AST 算每个函数的行范围, 与改动行求交
hits=[]
for pf in files:
    fn=norm(pf); ch=changed.get(fn)
    try: tree=ast.parse(open(pf,encoding="utf-8").read())
    except Exception: continue
    _lines=open(pf,encoding="utf-8").read().splitlines()
    for node in ast.walk(tree):
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        fl=node.lineno; end=getattr(node,"end_lineno",node.lineno)
        # 函数被改动 = 无改动信息(全查) 或 任一改动行落在函数行范围
        touched = (ch is None) or any(fl<=l<=end for l in (ch or []))
        if not touched: continue
        cc=cc_map.get((fn,node.name))
        if cc and cc>cc_max:
            _dl=_lines[fl-1] if fl<=len(_lines) else ""
            if "complexity-ok" in _dl: continue
            hits.append(f"{pf}:{fl} {node.name} 复杂度{cc}>{cc_max}")
if hits:
    print("\n".join(hits[:10])); sys.exit(1)
PYCC
    rc=$?
    if [ "$rc" -ne 0 ]; then
      fail "改动函数圈复杂度超标(>${CC_MAX}，仅查改动函数)"
    else
      pass "圈复杂度检查通过(radon,≤${CC_MAX},仅查改动函数)"
    fi
  elif [ "${#js_files[@]}" -gt 0 ]; then
    if has "$ESLINT_BIN" && [ -f "$ESLINT_GOV_CONFIG" ]; then
      pass "圈复杂度交由治理 ESLint 配置（complexity warn ≤${CC_MAX}）: ${ESLINT_GOV_CONFIG##*/}"
    else
      tool_missing "JS 圈复杂度未校验：缺 eslint 或治理 ESLint 配置（sync-governance / npm i -g eslint）"
    fi
  else
    pass "圈复杂度检查无目标文件"
  fi
}
check_cc

# ---------- 3. Lint（按栈探测） ----------
if [ "${#py_files[@]}" -gt 0 ] && has "$RUFF_BIN"; then
  # ruff 棘轮门禁：基线内的既有 lint 债务不阻断（否则门长期红→被绕过→失效），
  # 仅拦截"新增/回退"的错误。优先用 ratchet，回退到裸 ruff。
  RATCHET="scripts/ai-verify/ruff_ratchet.py"
  if [ -f "$RATCHET" ] && has python3; then
    if python3 "$RATCHET" check >/dev/null 2>&1; then pass "ruff 棘轮门禁通过（挡增量，基线债允许偿还）";
    else fail "ruff 检测到 lint 增量（回退），详见 python3 $RATCHET check"; fi
  else
    if "$RUFF_BIN" check "${py_files[@]}" >/dev/null 2>&1; then pass "ruff 通过";
    else fail "ruff 检测到问题（$RUFF_BIN check 查看详情）"; fi
  fi
  # 格式门禁（D-GATE-01A 棘轮）：CI 的 `Lint & Test (backend)` 会跑 `ruff format --check`；
  # 本地缺此步会出现「本地绿、CI 红」的假绿（2026-10-05 实证：PR #119 `lint` 因格式未过卡红）。
  # 只查本次改动文件；基线（.ai-governance/ruff-format-baseline.tsv）内的历史格式债跳过，
  # 基线外文件仍硬挡增量；基线内文件偿还后从其删除（只减不增，见基线 header）。
  FMT_BASELINE=".ai-governance/ruff-format-baseline.tsv"
  fmt_files=(); fmt_skipped=0
  for f in "${py_files[@]}"; do
    f="${f#./}"
    if [ -f "$FMT_BASELINE" ] && grep -qxF "$f" "$FMT_BASELINE"; then
      fmt_skipped=$((fmt_skipped+1))
    else
      fmt_files+=("$f")
    fi
  done
  if [ "${#fmt_files[@]}" -eq 0 ]; then
    pass "ruff format 棘轮：${fmt_skipped} 个改动文件在基线内跳过（历史格式债，未新增）"
  elif "$RUFF_BIN" format --check "${fmt_files[@]}" >/dev/null 2>&1; then
    pass "ruff format --check 通过（与 CI 同源；棘轮跳过 ${fmt_skipped} 个基线内文件）"
  else
    fail "ruff 格式未通过（基线外增量）：执行 $RUFF_BIN format ${fmt_files[*]} 后重试（CI 同源检查）"
  fi
fi
# JS 静态检查：任何含 JS 的提交都要跑（修复混合 PR 静默跳过 JS 检查的漏洞）
if [ "${#js_lint_files[@]}" -gt 0 ]; then
  if has "$ESLINT_BIN"; then
    if [ -f "$ESLINT_GOV_CONFIG" ]; then
      eslint_out="$("$ESLINT_BIN" --config "$ESLINT_GOV_CONFIG" "${js_lint_files[@]}" 2>&1)"; eslint_rc=$?
      if [ "$eslint_rc" -eq 0 ]; then
        eslint_debt="$(printf '%s\n' "$eslint_out" | grep -cE 'warning .*(complexity|max-depth)' || true)"
        pass "eslint（治理配置）通过；复杂度基线债 ${eslint_debt:-0} 处（warn 不阻断）"
      else
        printf '%s\n' "$eslint_out" | tail -12
        fail "eslint（治理配置）检测到问题"
      fi
    else
      tool_missing "缺少治理 ESLint 配置: ${ESLINT_GOV_CONFIG}（bash _ai-eng-kit/sync-governance.sh 同步后重试）"
    fi
  else
    tool_missing "未安装 eslint（JS 静态检查），CI 由 governance.yml 安装"
  fi
fi

# ---------- 4. 测试覆盖率（仅 L1/L2 强制；工具缺失转人工确认） ----------
run_coverage() {
  [ "${LEVEL}" = "L0" ] && { warn "L0 不强制覆盖率门禁（建议≥${COV_CORE}%）"; return; }
  # 向上搜索测试配置位置（项目根 或 backend/ 等子目录）
  local cfgdir=""
  for cand in "." "./backend" "./server" "./api"; do
    if [ -f "$cand/pytest.ini" ] || [ -f "$cand/pyproject.toml" ] || [ -f "$cand/setup.cfg" ]; then
      cfgdir="$cand"; break; fi
  done
  if [ -z "$cfgdir" ]; then warn "无 Python 测试配置，覆盖率门禁转人工确认"; return; fi
  if ! has "$PYTEST_BIN"; then tool_missing "未安装 pytest，覆盖率门禁无法执行（设 PYTEST_BIN）"; return; fi
  # 真门禁模式：仓库根 .ai-coverage-threshold 存在即启用
  # 格式(注释行以 # 开头):
  #   第一行: <阈值%> <测试目录> <cov源1> [cov源2...]（阈值=实测基线-5%，只许上调）
  #   可选第二行: branch <branch-floor%>（项目 branch 基线-5pp; 缺省=风险等级默认值）
  # 路径均相对 pytest 配置目录(cfgdir), pytest 从 cfgdir 执行
  # M4 ④(OODA-20261008-055): 未显式指定时，按**与 CI 同一规则**自动判定覆盖范围与强度
  # （CI 的 quality_scope：改动面含 src|adapters|benchmarks 的 .py → full+硬阻断，否则 changed+转人工确认）。
  # 旧口径本地恒为 full+硬阻断：任何合法提交都要跑全量覆盖率（实测 mci-world-model 本机 ~60min）→
  # 本地门禁实际不可用，只能靠 SKIP=ai-guard，等于没有本地拦截；且阈值模式绕过 scope，
  # 使 CI 的纯文档 PR 也被迫全量。显式传 PYTEST_SCOPE / COVERAGE_HARD_GATE 时一律以显式值为准
  # （CI 工作流即属此路径，行为不变）。
  local hard="${COVERAGE_HARD_GATE:-}"
  local scope="${PYTEST_SCOPE:-}"
  if [ -z "$hard" ] || [ -z "$scope" ]; then
    local _full=0 _f _rel
    if [ "${#FILES[@]}" -eq 0 ]; then _full=1; fi
    for _f in "${FILES[@]}"; do
      _rel="${_f#"$ROOT/"}"; _rel="${_rel#./}"
      case "$_rel" in
        .|src/*|adapters/*|benchmarks/*) _full=1;;
      esac
    done
    [ -z "$scope" ] && { if [ "$_full" = "1" ]; then scope=full; else scope=changed; fi; }
    [ -z "$hard" ] && { if [ "$_full" = "1" ]; then hard=1; else hard=0; fi; }
    say "覆盖率口径(自动): scope=${scope} hard=${hard}（规则同 CI quality_scope）"
  fi
  # 真门禁模式（阈值 = 仓库基线、恒硬阻断）只在 full 口径下启用：
  # changed 口径跑的是"改动文件的映射测试"子集，用全仓阈值判定无意义（会造出假红），
  # 交由下方映射路径处理（hard=0 时转人工确认）。
  if [ "$scope" = "full" ] && [ -f "$ROOT/.ai-coverage-threshold" ]; then
    local spec threshold testdir sources
    spec=$(grep -v '^[[:space:]]*#' "$ROOT/.ai-coverage-threshold" | head -1)
    threshold=$(echo "$spec" | awk '{print $1}')
    testdir=$(echo "$spec" | awk '{print $2}')
    sources=$(echo "$spec" | cut -s -d' ' -f3-)
    # 项目 branch floor: 未配置时保留风险等级默认值（禁止 COVERAGE_HARD_GATE=0 绕过: 阈值模式恒为硬门禁）
    local bspec
    bspec=$(grep -E '^[[:space:]]*branch[[:space:]]+[0-9]+([.][0-9]+)?[[:space:]]*$' "$ROOT/.ai-coverage-threshold" | head -1 | awk '{print $2}')
    [ -n "$bspec" ] && COV_BR="$bspec"
    say "阈值来源: .ai-coverage-threshold（核心≥${threshold}% 分支≥${COV_BR}%，测试目录 ${testdir}）"
    echo "- 阈值(实际): 核心覆盖≥${threshold}% 分支≥${COV_BR}%（来源 .ai-coverage-threshold）" >> "$REPORT"
    local cov_args=() s
    for s in $sources; do cov_args+=("--cov=$s"); done
    if (cd "$cfgdir" && "$PYTEST_BIN" "$testdir" -q "${cov_args[@]}" --cov-branch \
        --cov-report=term-missing --cov-report=json:"$COV_JSON" \
        --cov-fail-under="$threshold" >"$PYTEST_LOG" 2>&1); then
      local branch
      branch="$(python3 - "$COV_JSON" <<'PYCOV'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    totals = json.load(f)["totals"]
covered = totals.get("covered_branches", 0)
num = totals.get("num_branches", 0)
print("100" if num == 0 else f"{100 * covered / num:.1f}")
PYCOV
)"
      if [ -z "$branch" ] || [ "$(python3 -c "print(1 if float('$branch') < float('$COV_BR') else 0)")" = "1" ]; then
        fail "分支覆盖率 ${branch:-?}% < ${COV_BR}%（真门禁，见 ${PYTEST_LOG}）"
      else
        pass "覆盖率真门禁通过（$testdir 全量 核心≥${threshold}% 分支=${branch}% ≥${COV_BR}%，明细 ${PYTEST_LOG}）"
      fi
    else
      pytest_diag
      fail "pytest 失败或覆盖率 < ${threshold}%（真门禁，见 ${PYTEST_LOG}）"
    fi
    return
  fi
  # 在配置目录跑测试（F-GOVP0: pytest 必须从 cfgdir 执行——嵌套配置项目(如 MCI-BMI
  # 根目录无 pytest 配置、backend/pyproject.toml 有)从仓库根跑会加载错误 rootdir/env,
  # 导致大量 DB/env 依赖测试假失败; markers/asyncio/timeout 等 pytest 配置随 cwd 正常生效）
  # hard/scope 已在上方统一解析（M4 ④ 自动口径或显式注入）。
  local pytest_args=()
  if [ "$scope" = "changed" ]; then
    # 为每个改动源文件找对应测试: <path>/<mod>.py -> tests/**/test_<mod>.py
    local rel tmods=() t
    for ff in "${FILES[@]}"; do
      rel="${ff#"$ROOT/"}"; rel="${rel#./}"
      case "$rel" in *test*|*spec*) continue;; esac
      local base="${rel##*/}"; base="${base%.py}"
      [ "$base" = "__init__" ] && continue
      while IFS= read -r t; do tmods+=("$t"); done < <(find "$cfgdir" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | grep -vE '/\.venv|/vendor/|/node_modules/|/site-packages/')
    done
    if [ "${#tmods[@]}" -gt 0 ]; then
      # find 结果相对 cfgdir 表示, 使 pytest 在 cfgdir cwd 下正确解析
      local rel_t
      for t in "${tmods[@]}"; do
        rel_t="${t#"${cfgdir%/}"/}"
        pytest_args+=("$rel_t")
      done
    else warn "改动文件无对应单元测试，覆盖率门禁转人工确认（建议补 test_<mod>.py）"; return; fi
  fi
  # 2026-10-16（并发假红修复）：pytest 的 tmp_path 根（pytest-of-$USER/pytest-N）是全局共享的，
  # 并发跑两个 pytest（本仓 gate + 他线验收）会抢同一个 N，后启动方清理旧目录 ⇒ 正在运行的测试
  # 目录被删、git commit 假红。实证：test_tracked_scope_scans_versioned_files_only 全量运行
  # FAILED、单跑必过。故把 tmp 根钉进本 gate 的唯一目录。
  if (cd "$cfgdir" && "$PYTEST_BIN" "${pytest_args[@]}" -q --cov --cov-branch \
    --basetemp="$GATE_TMP_DIR/pytest-basetemp" \
    --cov-report=term-missing --cov-report=json:"$COV_JSON" >"$PYTEST_LOG" 2>&1); then
    local cov; cov=$(grep -E '^TOTAL' "$PYTEST_LOG" | tail -1 | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    # R1-3（A-1）：读数缺失时**不得**落进「通过」分支——「无数 → 绿」与「0% → 绿」同属静默降级。
    if [ -z "$cov" ]; then
      warn "覆盖率读数缺失（日志无 TOTAL 行，见 ${PYTEST_LOG}）→ 不作为通过依据"
    elif [ "$cov" -lt "${COV_CORE}" ] 2>/dev/null; then
      if [ "$hard" = "1" ]; then fail "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（等级 ${LEVEL}）"
      else warn "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（默认警告；COVERAGE_HARD_GATE=1 硬阻断）"; fi
    else
      local branch
      branch="$(python3 - "$COV_JSON" <<'PYCOV'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    totals = json.load(f)["totals"]
covered = totals.get("covered_branches", 0)
num = totals.get("num_branches", 0)
print("100" if num == 0 else f"{100 * covered / num:.1f}")
PYCOV
)"
      if [ -n "$branch" ] && [ "$(python3 -c "print(1 if float('$branch') < float('$COV_BR') else 0)")" = "1" ]; then
        if [ "$hard" = "1" ]; then
          fail "分支覆盖率 ${branch}% < 阈值 ${COV_BR}%（等级 ${LEVEL}）"
        else
          warn "分支覆盖率 ${branch}% < 阈值 ${COV_BR}%（默认警告；COVERAGE_HARD_GATE=1 硬阻断）"
        fi
      else
        pass "pytest+coverage 通过 (核心 ${cov:-?}% ≥ ${COV_CORE}% 分支 ${branch:-?}% ≥ ${COV_BR}%, scope=$scope)"
      fi
    fi
  else
    # 测试失败：硬阻断（测试不过绝不能放行）
    pytest_diag
    warn "pytest 失败（见 ${PYTEST_LOG}, scope=${scope}；pre-commit stash 可能导致假阳性, CI 做最终验证）"
  fi
}
run_coverage

# ---------- 4.4 治理脚本覆盖率换量尺（A-1 ／ 台账 OODA-20261015-018、-020；R1-2a） ----------
# 背景（实测）：`scripts/ai-verify/**` 的用例以 **subprocess** 方式调用被测脚本，进程内覆盖率
# **结构性**取不到数据 ⇒ 该路径读数恒 0.00%，而 scope=changed 对「0% < 阈值」只 warn，
# 于是形成「报 0% 却放行」。修法：改动面触及该路径时，用 COVERAGE_PROCESS_START ＋ PYTHONPATH
# 注入**自包含** sitecustomize 钩子（不改动 venv、不依赖任何 .pth），令子进程自采集后 combine 出读数。
# 判据：**不新增阈值**（既不收紧也不放宽既有门禁），只关闭静默。四态显式：
#   采到读数（被测语句 > 0）⇒ 打印读数 ＋ pass；
#   采不到读数／0 语句 ⇒ **fail**（I3：禁止以「0%」冒充结论）；
#   未触及该路径／无映射用例 ⇒ 打印**显式**「不适用（原因）」行，不留空、不冒绿。
run_gov_cov() {
  gov_say(){ say "治理覆盖率判据: $1"; echo "- 治理覆盖率判据: $1" >> "$REPORT"; }
  # 与 run_coverage 同口径：L0 不启用（其边界本就不含逻辑改动，见 OODA §3 L0 边界）
  if [ "${LEVEL}" = "L0" ]; then
    gov_say "不适用（L0 不强制覆盖率门禁）"
    return 0
  fi
  local face=() rel f base
  for f in "${FILES[@]}"; do
    rel="${f#"$ROOT/"}"; rel="${rel#./}"
    case "$rel" in scripts/ai-verify/*.py) face+=("$rel");; esac
  done
  if [ "${#face[@]}" -eq 0 ]; then
    gov_say "不适用（本次改动面未触及 scripts/ai-verify/**.py）"
    return 0
  fi
  local tmods=() t
  for rel in "${face[@]}"; do
    base="${rel##*/}"; base="${base%.py}"
    [ "$base" = "__init__" ] && continue
    while IFS= read -r t; do tmods+=("$t"); done < <(find "$ROOT" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | grep -vE '/\.venv|/vendor/|/node_modules/|/site-packages/')
  done
  if [ "${#tmods[@]}" -eq 0 ]; then
    gov_say "不适用（无映射用例：请补 tests/**/test_<mod>.py）"
    warn "治理脚本改动（scripts/ai-verify/**.py）无映射用例，覆盖率无法采集 → 转人工确认"
    return 0
  fi
  local py="${PYTHON_BIN:-python3}"
  if ! "$py" -c 'import coverage' >/dev/null 2>&1; then
    gov_say "不适用（解释器缺 coverage 模块）"
    tool_missing "治理覆盖率采集需要 coverage 模块（${py} -m pip install coverage）"
    return 0
  fi
  local tmp hook rcfile data clog line stmts covp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/qgate-govcov.XXXXXX")" || { warn "无法创建临时目录，治理覆盖率转人工确认"; return 0; }
  hook="$tmp/hook"; mkdir -p "$hook"
  printf '%s\n' \
    'import os' \
    'if os.getenv("COVERAGE_PROCESS_START"):' \
    '    try:' \
    '        import coverage' \
    '        coverage.process_startup()' \
    '    except Exception:' \
    '        pass' > "$hook/sitecustomize.py"
  rcfile="$tmp/coveragerc"; data="$tmp/.coverage"; clog="$tmp/pytest.log"
  # source 必须写**绝对路径**：用例以 `subprocess.run(..., cwd=tmp_path)` 调被测脚本，
  # 相对 source 在被测进程的 cwd 下解析不到 ⇒ 采数为 0（实测：绝对 62% vs 相对 38%，
  # 且单例调用时相对口径直接掉到 0%）。
  printf '[run]\nsource = %s/scripts/ai-verify\nparallel = true\ndata_file = %s\n' "$ROOT" "$data" > "$rcfile"
  local targs=()
  for t in "${tmods[@]}"; do targs+=("${t#"$ROOT/"}"); done
  # `-o addopts=`：多仓在 addopts 里写死全局 --cov-fail-under（只跑子集必假红）；本次只采数据，
  # 阈值判定仍归上方 run_coverage，避免「换量尺」被误读为「换阈值」。
  if ! ( cd "$ROOT" && COVERAGE_PROCESS_START="$rcfile" COVERAGE_FILE="$data" \
        PYTHONPATH="$hook${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTEST_BIN" -o addopts= -q -p no:cacheprovider "${targs[@]}" ) >"$clog" 2>&1; then
    gov_say "未通过（映射用例失败，无读数：${targs[*]}）"
    fail "治理脚本映射用例未通过（${targs[*]}，日志 ${clog}）"
    return 0
  fi
  ( cd "$ROOT" && "$py" -m coverage combine --data-file="$data" ) >/dev/null 2>&1
  line="$( cd "$ROOT" && "$py" -m coverage report --data-file="$data" --include='*/scripts/ai-verify/*' 2>/dev/null | awk '$1=="TOTAL"{print}' )"
  stmts="$(printf '%s' "$line" | awk '{print $2}')"; covp="$(printf '%s' "$line" | awk '{print $4}')"
  if [ -z "${line:-}" ] || [ -z "${stmts:-}" ] || [ "${stmts:-0}" -le 0 ] 2>/dev/null; then
    gov_say "❌ 未采到读数（scripts/ai-verify/**）"
    fail "治理覆盖率读数缺失（scripts/ai-verify/** 无数据）：采集链路或用例路径失效，禁止以 0% 冒充结论（日志 ${clog}）"
    return 0
  fi
  echo "- 治理覆盖率判据: ${covp:-?}（scripts/ai-verify/** 实测 ${stmts} 语句；口径＝COVERAGE_PROCESS_START ＋ sitecustomize 子进程采集；日志 ${clog}）" >> "$REPORT"
  pass "治理覆盖率读数（scripts/ai-verify/**）: ${covp:-?}（${stmts} 语句）"
}
run_gov_cov

# ---------- 4.5 AI Eval 回归（L5；改动触及 AI 模块时触发） ----------
# 触发路径＝本组织各仓的 AI 面：`backend/ai_agent/**`、`backend/ai_eval/**`（题库本身），
# 以及 `backend/services/ai_engine/**`（营养系统的 AI 引擎真实位置）。
# 背景（OODA-20261012-195）：触发条件原本只认前两者，而某仓 AI 面在 `services/ai_engine/**`
# ⇒ 条件永不成立、函数首行即 return，「门禁脚本在」被读成「AI 回归已覆盖」。**条件不成立＝门禁不存在**。
# 命中即跑规则层 eval（route/refuse/perm/anchor/template），题库 `backend/ai_eval/evalset.yaml`；
# 用例条数**由题库实测规模决定**，不再在文案里硬编码（硬编码会让「空转」与「真跑」看起来一样）。
run_ai_eval() {
  local ai_hit=0 ff
  for ff in "${FILES[@]}"; do
    case "$ff" in
      backend/ai_agent/*|backend/ai_eval/*|backend/services/ai_engine/*) ai_hit=1; break;;
    esac
  done
  [ "$ai_hit" -eq 0 ] && return 0
  if ! has "$PYTEST_BIN"; then tool_missing "未安装 pytest，AI Eval 门禁无法执行（设 PYTEST_BIN）"; return; fi
  local cfgdir=""
  for cand in "." "./backend"; do
    if [ -f "$cand/pytest.ini" ] || [ -f "$cand/pyproject.toml" ]; then cfgdir="$cand"; break; fi
  done
  [ -z "$cfgdir" ] && { tool_missing "无 pytest 配置目录，AI Eval 门禁无法执行"; return; }
  # D-GATE-02①：向上搜索命中的配置目录(如根 pyproject.toml)可能不含 AI Eval 题库测试；
  # 当该 cfgdir 缺测试文件而 backend/ 存在时回落到 backend/（否则 pytest 找不到文件而假红）。
  if [ ! -f "$cfgdir/tests/unit/test_ai_eval_rules.py" ] && [ -f "./backend/tests/unit/test_ai_eval_rules.py" ]; then
    cfgdir="./backend"
  fi
  local eval_test="$cfgdir/tests/unit/test_ai_eval_rules.py"
  local evalset="$cfgdir/ai_eval/evalset.yaml"
  local ai_eval_log="$GATE_TMP_DIR/qgate_ai_eval.log"
  # 触发成立但**无载体**时按用途分口径（OODA-20261012-195）：
  #   ① 有题库、无测试 ⇒ fail-closed（有语料却不跑＝门禁空转，与「条件永不成立」同罪）；
  #   ② 题库与测试皆无 ⇒ 显式 warn（把「本仓未覆盖」说出来，不再让 404 被读成「规则退化」；
  #      不 fail-closed，避免对尚未建设题库的仓造成跨仓连坐）。
  if [ ! -f "$eval_test" ]; then
    if [ -f "$evalset" ]; then
      tool_missing "AI Eval 题库存在但缺测试 ${eval_test}（有语料不跑＝门禁空转，见 OODA-20261012-195）"
    else
      warn "AI 面已命中但本仓无 AI Eval 载体（$evalset 与 $eval_test 均缺）⇒ 该门禁在本仓**未覆盖**，不视为已通过（OODA-20261012-195）"
    fi
    return
  fi
  if (cd "$cfgdir" && "$PYTEST_BIN" tests/unit/test_ai_eval_rules.py -q >"$ai_eval_log" 2>&1); then
    local n
    n=$(grep -oE '[0-9]+ passed' "$ai_eval_log" | tail -1 | grep -oE '^[0-9]+' || true)
    pass "AI Eval 规则层回归通过（用例 ${n:-?} 条，明细 ${ai_eval_log}）"
  else
    fail "AI Eval 规则层回归失败（工具路由/图谱/权限退化，见 ${ai_eval_log}）"
  fi
}
run_ai_eval

# ---------- 5. SAST（L2 强制；误报需闭环记录） ----------
# P1（2026-10-16 实证 · nutrition-system-hospital 开窗）：重定向路径原写 `${QGATE_TMP_DIR:-/tmp}`
# 且**不加引号** ⇒ QGATE_TMP_DIR 含空格时 bash 在解析阶段即报 ambiguous redirect，bandit
# **根本没执行**；紧接着 grep 读不到日志得 `highs=0` ⇒ 落 warn（假绿），High 被静默掩盖。
# 触发面定性（2026-10-16 更正）：原注释写「本仓当前不触发，属潜在缺陷」——**与事实相反**。
# 该仓 A 轮（OODA-20261015-549）与 A2 轮（-552）以门禁自身形式调用，显式传
# `QGATE_TMP_DIR="$PWD/$OUT/gate-A"`，而 `$PWD`（含空格的仓路径）**本身就是含空格值** ⇒ 两轮 SAST
# 均未执行、判红被伪装成 warn（原始报错在案 docs/deployment/evidence/20261016-window-step1-hu2-01/
# step1-A-gate.log.gz:10）。故本缺陷是**在途事故**，不是潜在缺陷；引用受影响的读数必须注明
# 「SAST 面无证据」。未触发的只有**经 ai-guard.sh 的间接调用**（QGATE_TMP_DIR 由
# `TMPDIR=/var/folders/…` 派生，无空格）与 CI（未设该变量 ⇒ `/tmp`）。修法两条：
#   ① 复用脚本已定义的 GATE_TMP_DIR 并加引号（`-r -q "${FILES[@]}"` 早已引用，属同类漏改）；
#   ② 加 fail-closed 判据「日志不存在或为空 ⇒ SAST 未执行」，把假绿变真红。
# ③（2026-10-16 自身对抗测试补严）仅判「非空」仍有洞：日志路径被占为**目录**时 `test -s` 对目录
#   恒真 ⇒ 跳过 fail-closed、bandit 未跑却仍落 warn（已实测复现）。改为**先显式截断**
#   （`! : >"$BANDIT_LOG"` 失败即判红），既堵目录/不可写路径，也消灭「读到陈旧日志」的假绿。
if [ "${LEVEL}" = "L2" ]; then
  if has "$BANDIT_BIN"; then
    BANDIT_LOG="$GATE_TMP_DIR/qgate_bandit.log"
    if ! : >"$BANDIT_LOG" 2>/dev/null; then
      fail "无法写入 SAST 日志（${BANDIT_LOG}）⇒ SAST 无法执行（fail-closed，禁止判通过）"
    elif "$BANDIT_BIN" -r -q "${FILES[@]}" >"$BANDIT_LOG" 2>&1; then pass "bandit SAST 通过";
    elif [ ! -s "$BANDIT_LOG" ]; then
      fail "bandit 未产生日志（${BANDIT_LOG}）⇒ SAST 疑似未执行（fail-closed，禁止判通过）"
    else
      highs=$(grep -ciE 'severity:[[:space:]]+high' "$BANDIT_LOG" 2>/dev/null || true)
      highs=${highs:-0}
      if [ "$highs" -gt 0 ]; then fail "bandit 发现 $highs 个高危（误报须走豁免闭环: # nosec: TICKET-xxx 且写入 nosec-allowlist.tsv）";
      else warn "bandit 仅中低危告警，需人工确认"; fi
    fi
  else tool_missing "未安装 bandit，L2 SAST 无法执行（设 BANDIT_BIN）"; fi
fi

# ---------- 6. 变异测试（L0 强制；L1/L2 可选,抓逻辑缺陷,门禁抓不到的）----------
# 静态门禁(ruff/radon/bandit)不做路径敏感分析,逻辑缺陷(除零/None/边界)只能靠测试覆盖。
# 变异测试自动注入缺陷验证测试套件有效性——这是"L0 免逐行审核"成立的科学依据。
MUTMIN="${MUTATION_MIN:-80}"
run_mutation() {
  # F-12(P1-B): L0/L2 默认武装；L1 默认跳过，可用 MUTATION_GATE=1 诊断开启。
  case "$LEVEL" in
    L0|L2)
      if [ "${MUTATION_GATE:-1}" != "1" ]; then return; fi
      ;;
    *)
      if [ "${MUTATION_GATE:-0}" != "1" ]; then return; fi
      ;;
  esac
  local strict=0
  case "$LEVEL" in L0|L2) strict=1;; esac
  # F-22(P0-H) 回灌（OODA-20261015-058 / 本仓 OODA-20261004-038）：push（合并后审计）
  # 事件下变异门禁不得 fail-closed —— 已合入 main 的改动无法再通过「补同名测试」回滚门禁，
  # 沿用严格模式会让 main 的 gate 永久留红，把「门禁红」退化为常态噪声。
  # 仅当调用方显式注入 MUTATION_STRICT 时覆盖；未注入时行为完全不变（PR 合入拦截点仍严格）。
  if [ -n "${MUTATION_STRICT:-}" ]; then strict="$MUTATION_STRICT"; fi
  # 找每个改动 py 文件对应测试 + 用 governance 自带 mutation-check.sh(不依赖 mutmut)
  # F-GOVP0: 先查受版本控制的显式映射 .ai-governance/mutation-test-map.tsv
  # (格式: 源文件<TAB>测试目标; 目标可为文件或目录; 精确源路径匹配;
  #  多目标以**逗号**分隔 —— OODA-20261013-227，源自 -224 §四提案),
  # 映射命中但目标不存在 → 不回退, 直接 fail-closed; 无映射条目 → 同名 fallback。
  local target base tfile found
  local pairs=()
  local source_files=()
  local mapfile=".ai-governance/mutation-test-map.tsv"
  local map_hit=0 msrc mtgt
  for target in "${py_files[@]}"; do
    base="${target##*/}"; base="${base%.py}"
    case "$base" in test_*|*_test) continue;; esac
    case "$target" in */tests/*) continue;; esac
    source_files+=("$target")
    tfile=""
    map_hit=0
    if [ -f "$mapfile" ]; then
      while IFS=$'\t' read -r msrc mtgt || [ -n "$msrc" ]; do
        mtgt="${mtgt%$'\r'}"
        case "$msrc" in ''|'#'*) continue;; esac
        if [ "$msrc" = "$target" ]; then
          map_hit=1
          # 多测试目标（OODA-20261013-227）：值可用逗号分隔登记多个测试目标，只保留
          # **实际存在**的；一个都不存在 ⇒ tfile 为空 ⇒ 仍走 map_hit=1 的 fail-closed
          # （原有口径不变）。用途：契约断言分散在多个测试文件时，变异臂能全部看见
          # （旧版只取单目标，异名文件里的断言不在视野 → 「有测试却仍判覆盖不足」）。
          _mt=""; _save_ifs="$IFS"; IFS=','
          for _one in $mtgt; do
            [ -n "$_one" ] || continue
            [ -e "$_one" ] && _mt="${_mt:+$_mt,}$_one"
          done
          IFS="$_save_ifs"
          [ -n "$_mt" ] && tfile="$_mt"
          break
        fi
      done < "$mapfile"
    fi
    if [ "$map_hit" -eq 0 ]; then
      for cand in "backend" "." "./backend"; do
        found="$(find "$cand" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | head -1)"
        [ -n "$found" ] && { tfile="$found"; break; }
      done
    fi
    if [ -z "$tfile" ]; then
      if [ "$strict" = "1" ]; then
        fail "改动文件 $base 无对应测试,变异门禁无法验证(L0/L2 fail-closed)"
      else
        warn "改动文件 $base 无对应测试,变异测试跳过(逻辑缺陷门禁抓不到)"
      fi
      continue
    fi
    pairs+=("$target|$tfile")
  done
  if [ "${#pairs[@]}" -eq 0 ]; then
    if [ "$strict" = "1" ] && [ "${#source_files[@]}" -gt 0 ]; then
      fail "所有改动 py 文件均无对应测试,变异门禁无法验证(L0/L2 fail-closed)"
    fi
    return
  fi
  # 定位 mutation-check.sh(转绝对路径, 因后续可能 cd)
  local mcheck=""
  for cand in "$KIT_GUARD_DIR/mutation-check.sh" "$(dirname "$0")/mutation-check.sh" "$PWD/scripts/ai-verify/mutation-check.sh"; do
    if [ -f "$cand" ]; then
      case "$cand" in /*) mcheck="$cand";; *) mcheck="$PWD/$cand";; esac
      break
    fi
  done
  if [ -z "$mcheck" ]; then
    if [ "$strict" = "1" ]; then
      fail "未找到 mutation-check.sh,变异门禁无法验证(L0/L2 fail-closed)"
    else
      warn "未找到 mutation-check.sh,变异测试跳过"
    fi
    return
  fi
  # 把 venv/bin 加入 PATH(若 RADON_BIN 来自 venv), 让 mutation-check 用短名 pytest
  local savedpath="$PATH"
  [ -n "${RADON_BIN:-}" ] && [ -d "$(dirname "${RADON_BIN}")" ] && export PATH="$(dirname "${RADON_BIN}"):$PATH"
  say "运行变异测试(覆盖 ${#pairs[@]} 个文件, 阈值≥${MUTMIN}%)..."
  local pybin="${PYTHON_BIN:-python3}"
  local abs_src abs_tst
  local hard="${MUTATION_HARD_GATE:-1}"
  local rc overall_rc=0 pair ms na_total=0
  for pair in "${pairs[@]}"; do
    target="${pair%%|*}"; tfile="${pair#*|}"
    base="${target##*/}"; base="${base%.py}"
    case "$target" in /*) abs_src="$target";; *) abs_src="$PWD/$target";; esac
    case "$tfile" in
      /*) abs_tst="$tfile";;
      *)
        if [ "${tfile#*,}" != "$tfile" ]; then
          # 多目标：逐个转绝对路径再以逗号拼接（mutation-check 认逗号分隔）
          abs_tst=""; _save_ifs="$IFS"; IFS=','
          for _one in $tfile; do
            [ -n "$_one" ] || continue
            abs_tst="${abs_tst:+$abs_tst,}$PWD/$_one"
          done
          IFS="$_save_ifs"
        else
          abs_tst="$PWD/$tfile"
        fi;;
    esac
    local srcdir="$PWD"; local _d="$(dirname "${abs_tst%%,*}")"
    while [ "$_d" != "/" ]; do
      if [ -f "$_d/pyproject.toml" ] || [ -f "$_d/conftest.py" ]; then srcdir="$_d"; break; fi
      _d="$(dirname "$_d")"
    done
    cd "$srcdir"
      # M6: 变异范围同样认波次/CI 基线（否则干净检出下回退全文件，把历史大文件记成本次分数）
      MUTATION_MIN="$MUTMIN" MUTATION_DIFF_BASE="${GATE_DIFF_BASE:-}" PYTHON_BIN="$pybin" \
        bash "$mcheck" "$abs_src" "$abs_tst" "pytest --override-ini=addopts=" >"$MUT_LOG" 2>&1
    rc=$?
    cd "$ROOT"
    [ "$rc" -ne 0 ] && overall_rc=1
    # 分数取值（OODA-20261007-104）：`MUTATION_SCORE=N/A` 是「改动行内无 fitting 变异算子」的
    # 合法结果（不阻断）。旧正则 `[0-9NA]+` 会把 "N/A" 截成 "N"，日志里渲染成假分数「N%」——
    # 门禁日志声称了一个不存在的分数，属「N/A 语义未显式裁决」的同族缺陷。
    ms_raw="$(grep -oE 'MUTATION_SCORE=[0-9NA/]+' "$MUT_LOG" | cut -d= -f2)"
    # M5(OODA-20261007-053): N/A 不是达标，且在严格模式（L0/L2＝合入拦截点）转人工确认。
    # 不 fail-closed：量化（2026-10-09）受约束文件 26/300 任何改动必落 N/A、近 40 提交
    # 15 个改动单元中 5 个落 N/A → fail-closed 只会制造假红。
    na_reason="$(grep -oE 'MUTATION_NA_REASON=[a-z0-9-]+' "$MUT_LOG" | head -1 | cut -d= -f2)"
    na_lines="$(grep -oE 'MUTATION_NA_LINES=[a-z0-9]+' "$MUT_LOG" | head -1 | cut -d= -f2)"
    if [ "$rc" -eq 0 ] && { [ "$ms_raw" = "N/A" ] || [ -z "$ms_raw" ]; }; then
      na_total=$((na_total+1))
      if [ "$strict" = "1" ]; then
        warn "变异测试无可评变异($base: N/A, 原因=${na_reason:-unknown}, 变异范围=${na_lines:-?}) —— L0/L2 不得记为达标，需人工确认改动确无逻辑语义（声明/重导出/纯删除/注解类）"
      else
        say "变异测试无可评变异($base: N/A, 原因=${na_reason:-unknown}, 变异范围=${na_lines:-?})，push 合并后审计不阻断"
      fi
    elif [ "$rc" -eq 0 ]; then
      pass "变异测试通过($base: ${ms_raw}% ≥ ${MUTMIN}%, L0 可免逐行审核的科学依据)"
    elif [ "$rc" -eq 1 ]; then
      if [ "$hard" = "1" ]; then
        fail "变异分数 $base: ${ms_raw}% < ${MUTMIN}%(${LEVEL} 逻辑缺陷未被测试捕获,需补测试)"
      else
        warn "变异分数 $base: ${ms_raw}% < ${MUTMIN}%(${LEVEL} 默认警告；MUTATION_HARD_GATE=1 硬阻断)"
      fi
      grep -E '^  - 存活' "$MUT_LOG" | head -5 | while read -r l; do say "   $l"; done
    else
      if [ "$strict" = "1" ]; then
        fail "变异测试环境错误($base rc=$rc),L0/L2 fail-closed 无法验证测试有效性: $(tail -1 "$MUT_LOG" 2>/dev/null)"
      else
        warn "变异测试环境错误($base rc=$rc): $(tail -1 "$MUT_LOG" 2>/dev/null)"
        warn "无法自动验证测试有效性,建议人工抽审边界"
      fi
    fi
  done
  if [ "$na_total" -gt 0 ]; then
    if [ "$strict" = "1" ]; then
      warn "变异门禁 N/A 覆盖 ${na_total}/${#pairs[@]} 个文件（无可评变异，已转人工确认；不阻断≠达标）"
    else
      say "变异门禁 N/A 覆盖 ${na_total}/${#pairs[@]} 个文件（无可评变异，push 审计不阻断）"
    fi
  fi
  export PATH="$savedpath"
  return "$overall_rc"
}
run_mutation

# ---------- 汇总 ----------
echo >> "$REPORT"
if [ "$FAIL" -eq 1 ]; then
  say "门禁未通过（等级 ${LEVEL}）→ 废弃/重试（重试上限 MAX_RETRY=3,超限转人工）"
  echo "**结论：❌ 未通过**" >> "$REPORT"
  exit 1
fi
if [ "$WARN" -eq 1 ]; then
  say "门禁通过（等级 ${LEVEL}，含警告，需人工确认警告项）"
  echo "**结论：⚠️ 通过(含警告)**" >> "$REPORT"
  exit 0
fi
say "门禁全部通过（等级 ${LEVEL}）"
echo "**结论：✅ 通过**" >> "$REPORT"
echo "报告: $REPORT"
exit 0
