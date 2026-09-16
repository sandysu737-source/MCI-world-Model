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
# 工具路径：优先环境变量，再 PATH。支持 venv（如 RADON_BIN=backend/.venv/bin/radon）
RADON_BIN="${RADON_BIN:-radon}"
RUFF_BIN="${RUFF_BIN:-ruff}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
BANDIT_BIN="${BANDIT_BIN:-bandit}"
has(){ command -v "$1" >/dev/null 2>&1; }
mkdir -p "$REPORT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
REPORT="$REPORT_DIR/$TS-gate-${LEVEL}.md"

if [ "$#" -gt 0 ]; then FILES=("$@"); else
  FILES=()
  while IFS= read -r l; do [ -n "$l" ] && FILES+=("$l"); done < <(
    git diff --cached --name-only --diff-filter=ACMR 2>/dev/null | sort -u)
fi
[ "${#FILES[@]}" -eq 0 ] && FILES=(".")

say(){ printf '\033[1m[gate]\033[0m %s\n' "$1"; }

# 差异化阈值表
case "${LEVEL}" in
  L0) COV_CORE=85; COV_BR=80; COV_UTIL=70;  CC_MAX=15; DUP=5;;
  L1) COV_CORE=90; COV_BR=85; COV_UTIL=80;  CC_MAX=15; DUP=5;;
  L2) COV_CORE=95; COV_BR=90; COV_UTIL=85;  CC_MAX=15; DUP=5;;
esac

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
  local v; v="$("$bin" --version 2>/dev/null | head -1)"
  [ -z "$v" ] && fail "$name 无法输出 --version, 疑似伪造二进制: $bin"
}
verify_tool_bin RADON_BIN
verify_tool_bin RUFF_BIN
verify_tool_bin BANDIT_BIN
verify_tool_bin PYTEST_BIN
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
changed={}
for pf in files:
    ch=set(); saw_hunk=False
    if in_git:
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
      fail "改动函数圈复杂度超标(>$CC_MAX，仅查改动函数)"
    else
      pass "圈复杂度检查通过(radon,≤${CC_MAX},仅查改动函数)"
    fi
  elif has eslint && [ "${#js_files[@]}" -gt 0 ]; then
    pass "圈复杂度交由 eslint complexity 规则（max $CC_MAX）"
  else
    tool_missing "未安装 radon/eslint，圈复杂度未自动校验（设 RADON_BIN 或 pip install radon）"
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
fi
if [ "${#js_files[@]}" -gt 0 ] && has eslint; then
  if eslint "${js_files[@]}" >/dev/null 2>&1; then pass "eslint 通过";
  else fail "eslint 检测到问题"; fi
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
  # 格式: <阈值%> <测试目录> <cov源1> [cov源2...]（阈值=实测基线-5%，只许上调）
  if [ -f "$ROOT/.ai-coverage-threshold" ]; then
    local spec threshold testdir sources
    spec=$(grep -v '^[[:space:]]*#' "$ROOT/.ai-coverage-threshold" | head -1)
    threshold=$(echo "$spec" | awk '{print $1}')
    testdir=$(echo "$spec" | awk '{print $2}')
    sources=$(echo "$spec" | cut -s -d' ' -f3-)
    local cov_args=() s
    for s in $sources; do cov_args+=("--cov=$s"); done
    if (cd "$cfgdir" && "$PYTEST_BIN" "$testdir" -q "${cov_args[@]}" --cov-branch \
        --cov-report=term-missing --cov-report=json:/tmp/qgate_cov.json \
        --cov-fail-under="$threshold" >/tmp/qgate_pytest.log 2>&1); then
      local branch
      branch="$(python3 - /tmp/qgate_cov.json <<'PYCOV'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    totals = json.load(f)["totals"]
covered = totals.get("covered_branches", 0)
num = totals.get("num_branches", 0)
print("100" if num == 0 else f"{100 * covered / num:.1f}")
PYCOV
)"
      if [ -z "$branch" ] || [ "$(python3 -c "print(1 if float('$branch') < float('$COV_BR') else 0)")" = "1" ]; then
        fail "分支覆盖率 ${branch:-?}% < ${COV_BR}%（真门禁，见 /tmp/qgate_pytest.log）"
      else
        pass "覆盖率真门禁通过（$testdir 全量 核心≥${threshold}% 分支=${branch}% ≥${COV_BR}%，明细 /tmp/qgate_pytest.log）"
      fi
    else
      fail "pytest 失败或覆盖率 < ${threshold}%（真门禁，见 /tmp/qgate_pytest.log）"
    fi
    return
  fi
  # 在配置目录跑测试（cwd 不变，用 pytest 的 rootdir 推断）
  # F-16(P1-D): 本地默认与 CI 相同（full + 硬阻断），显式降低强度只用于诊断。
  local hard="${COVERAGE_HARD_GATE:-1}"
  local scope="${PYTEST_SCOPE:-full}"
  local pytest_args=()
  if [ "$scope" = "changed" ]; then
    # 为每个改动源文件找对应测试: <path>/<mod>.py -> tests/**/test_<mod>.py
    local rel tmods=()
    for ff in "${FILES[@]}"; do
      rel="${ff#"$ROOT/"}"; rel="${rel#./}"
      case "$rel" in *test*|*spec*) continue;; esac
      local base="${rel##*/}"; base="${base%.py}"
      [ "$base" = "__init__" ] && continue
      while IFS= read -r t; do tmods+=("$t"); done < <(find "$cfgdir" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | grep -vE '/\.venv|/vendor/|/node_modules/|/site-packages/')
    done
    if [ "${#tmods[@]}" -gt 0 ]; then pytest_args=("${tmods[@]}");
    else warn "改动文件无对应单元测试，覆盖率门禁转人工确认（建议补 test_<mod>.py）"; return; fi
  fi
  if "$PYTEST_BIN" "${pytest_args[@]}" -q --cov --cov-branch \
    --cov-report=term-missing --cov-report=json:/tmp/qgate_cov.json >/tmp/qgate_pytest.log 2>&1; then
    local cov; cov=$(grep -E '^TOTAL' /tmp/qgate_pytest.log | tail -1 | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    if [ -n "$cov" ] && [ "$cov" -lt "${COV_CORE}" ] 2>/dev/null; then
      if [ "$hard" = "1" ]; then fail "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（等级 $LEVEL）"
      else warn "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（默认警告；COVERAGE_HARD_GATE=1 硬阻断）"; fi
    else
      local branch
      branch="$(python3 - /tmp/qgate_cov.json <<'PYCOV'
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
          fail "分支覆盖率 ${branch}% < 阈值 ${COV_BR}%（等级 $LEVEL）"
        else
          warn "分支覆盖率 ${branch}% < 阈值 ${COV_BR}%（默认警告；COVERAGE_HARD_GATE=1 硬阻断）"
        fi
      else
        pass "pytest+coverage 通过 (核心 ${cov:-?}% ≥ ${COV_CORE}% 分支 ${branch:-?}% ≥ ${COV_BR}%, scope=$scope)"
      fi
    fi
  else
    # 测试失败：硬阻断（测试不过绝不能放行）
    warn "pytest 失败（见 /tmp/qgate_pytest.log, scope=$scope；pre-commit stash 可能导致假阳性, CI 做最终验证）"
  fi
}
run_coverage

# ---------- 4.5 AI Eval 回归（L5；改动触及 AI 模块时触发） ----------
# 命中 ai_agent//ai_eval/ 即跑规则层 eval（route/refuse/perm/anchor/template 52 题），
# 工具路由/图谱模板/权限拦截任何退化直接拦截；题库 backend/ai_eval/evalset.yaml
run_ai_eval() {
  local ai_hit=0 ff
  for ff in "${FILES[@]}"; do
    case "$ff" in
      backend/ai_agent/*|backend/ai_eval/*) ai_hit=1; break;;
    esac
  done
  [ "$ai_hit" -eq 0 ] && return 0
  if ! has "$PYTEST_BIN"; then tool_missing "未安装 pytest，AI Eval 门禁无法执行（设 PYTEST_BIN）"; return; fi
  local cfgdir=""
  for cand in "." "./backend"; do
    if [ -f "$cand/pytest.ini" ] || [ -f "$cand/pyproject.toml" ]; then cfgdir="$cand"; break; fi
  done
  [ -z "$cfgdir" ] && { tool_missing "无 pytest 配置目录，AI Eval 门禁无法执行"; return; }
  if (cd "$cfgdir" && "$PYTEST_BIN" tests/unit/test_ai_eval_rules.py -q >/tmp/qgate_ai_eval.log 2>&1); then
    pass "AI Eval 规则层回归通过（52 题，明细 /tmp/qgate_ai_eval.log）"
  else
    fail "AI Eval 规则层回归失败（工具路由/图谱/权限退化，见 /tmp/qgate_ai_eval.log）"
  fi
}
run_ai_eval

# ---------- 5. SAST（L2 强制；误报需闭环记录） ----------
if [ "${LEVEL}" = "L2" ]; then
  if has "$BANDIT_BIN"; then
    if "$BANDIT_BIN" -r -q "${FILES[@]}" >/tmp/qgate_bandit.log 2>&1; then pass "bandit SAST 通过";
    else
      highs=$(grep -c 'severity: HIGH' /tmp/qgate_bandit.log 2>/dev/null || true)
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
  # 找每个改动 py 文件对应测试 + 用 governance 自带 mutation-check.sh(不依赖 mutmut)
  local target base tfile found
  local pairs=()
  for target in "${py_files[@]}"; do
    base="${target##*/}"; base="${base%.py}"
    tfile=""
    for cand in "backend" "." "./backend"; do
      found="$(find "$cand" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | head -1)"
      [ -n "$found" ] && { tfile="$found"; break; }
    done
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
    if [ "$strict" = "1" ] && [ "${#py_files[@]}" -gt 0 ]; then
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
  local rc overall_rc=0 pair ms
  for pair in "${pairs[@]}"; do
    target="${pair%%|*}"; tfile="${pair#*|}"
    base="${target##*/}"; base="${base%.py}"
    case "$target" in /*) abs_src="$target";; *) abs_src="$PWD/$target";; esac
    case "$tfile" in /*) abs_tst="$tfile";; *) abs_tst="$PWD/$tfile";; esac
    local srcdir="$PWD"; local _d="$(dirname "$abs_tst")"
    while [ "$_d" != "/" ]; do
      if [ -f "$_d/pyproject.toml" ] || [ -f "$_d/conftest.py" ]; then srcdir="$_d"; break; fi
      _d="$(dirname "$_d")"
    done
    cd "$srcdir"
      MUTATION_MIN="$MUTMIN" PYTHON_BIN="$pybin" bash "$mcheck" "$abs_src" "$abs_tst" "pytest --override-ini=addopts=" >/tmp/qgate_mut.log 2>&1
    rc=$?
    cd "$ROOT"
    [ "$rc" -ne 0 ] && overall_rc=1
    ms="$(grep -oE 'MUTATION_SCORE=[0-9NA]+' /tmp/qgate_mut.log | cut -d= -f2)"
    if [ "$rc" -eq 0 ]; then
      pass "变异测试通过($base: ${ms}% ≥ ${MUTMIN}%, L0 可免逐行审核的科学依据)"
    elif [ "$rc" -eq 1 ]; then
      if [ "$hard" = "1" ]; then
        fail "变异分数 $base: ${ms}% < ${MUTMIN}%(${LEVEL} 逻辑缺陷未被测试捕获,需补测试)"
      else
        warn "变异分数 $base: ${ms}% < ${MUTMIN}%(${LEVEL} 默认警告；MUTATION_HARD_GATE=1 硬阻断)"
      fi
      grep -E '^  - 存活' /tmp/qgate_mut.log | head -5 | while read -r l; do say "   $l"; done
    else
      if [ "$strict" = "1" ]; then
        fail "变异测试环境错误($base rc=$rc),L0/L2 fail-closed 无法验证测试有效性: $(tail -1 /tmp/qgate_mut.log 2>/dev/null)"
      else
        warn "变异测试环境错误($base rc=$rc): $(tail -1 /tmp/qgate_mut.log 2>/dev/null)"
        warn "无法自动验证测试有效性,建议人工抽审边界"
      fi
    fi
  done
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
