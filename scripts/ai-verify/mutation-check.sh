#!/usr/bin/env bash
# 轻量变异测试器（governance 自带，跨平台，不依赖 mutmut）
# 原理: 对源文件逐条应用典型变异(运算符翻转/常量变更),跑测试
#   测试仍 PASS = 变异存活 = 测试覆盖有洞; FAIL = 变异被杀 = 测试有效
#   变异分数 = 被杀 / 有效变异总数 × 100%
# F-23(P0-J): 计分口径修正。本仓库副本 = kit 2026-09-30 版 + 两项修复：
#   ① f-string 整段不计分：Python ≥3.12 把 f-string 拆成 FSTRING_START/MIDDLE/END，
#      只匹配 STRING 会漏排除其字面量里的运算符文本 → 同一文件在 3.11 与 3.13/3.14 落点不同。
#   ② 语法破坏型变异不计分：`->` 变 `-<` 之类变异后无法 `ast.parse`，不构成"测试有效性"
#      证据，计为杀死会虚高分数（apply_mut 退出码 3 = 跳过，汇总里单独计数）。
#   历史根因（kit 版已修、本仓库副本此前落后一个版本）：排除区间偏移量重复计换行
#   （`splitlines(True)` 已含 `\n` 又 `+1`），使注释/字符串白名单整体前移而失效 ——
#   与 Python 版本无关，CI 3.11 同样命中。
# 跳过区间口径: 注释 / 字符串 / f-string 整段 / 类型注解区域 / 返回箭头 `->`。
# 用法: mutation-check.sh <源文件> <测试目标> [pytest前缀]
# 退出码: 0=达标 1=不达标 2=环境错误
SRC="${1:?用法: mutation-check.sh <源文件> <测试> [pytest]}"
TESTS="${2:?缺测试目标}"
PYTEST="${3:-pytest}"
PY="${PYTHON_BIN:-python3}"

[ -f "$SRC" ] || { echo "ERROR: 源文件不存在: $SRC"; exit 2; }

# run_test: PYTEST 含空格时用 sh -c
run_test(){ sh -c "PYTEST=\"$PYTEST\" TESTS=\"$TESTS\" \"\$PYTEST\" -x -q --no-header \"\$TESTS\"" >/dev/null 2>&1; }

echo "[mut] 基准测试..."
if ! sh -c "$PYTEST -x -q --no-header \"$TESTS\"" >/tmp/mut_base.log 2>&1; then
  echo "ERROR: 基准测试未通过,先修测试:"; tail -5 /tmp/mut_base.log; exit 2
fi
echo "[mut] 基准通过 ✓"

cp "$SRC" "$SRC.mutorig"
trap 'cp "$SRC.mutorig" "$SRC" 2>/dev/null; rm -f "$SRC.mutorig"' EXIT

# 变异范围只取 staged diff 的改动行，避免历史大文件掩盖本次逻辑盲区；
# 无差异（CI 干净检出）或差异不含代码行时回退全文件，防 0 变异点让门禁空转
MUTATION_SCOPE="${MUTATION_SCOPE:-changed}"
CHANGED_LINES_FILE="$(mktemp)"
trap 'cp "$SRC.mutorig" "$SRC" 2>/dev/null; rm -f "$SRC.mutorig" "$CHANGED_LINES_FILE" "$CHANGED_LINES_FILE.diff"' EXIT
if [ "$MUTATION_SCOPE" = "all" ]; then
  echo "[mut] 变异范围: 全文件（显式 MUTATION_SCOPE=all）"
elif [ "$MUTATION_SCOPE" = "changed" ]; then
  SRC_ROOT="$(git -C "$(dirname "$SRC")" rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -z "$SRC_ROOT" ]; then
    echo "ERROR: changed 模式需要 Git 仓库，或设置 MUTATION_SCOPE=all" >&2
    exit 2
  fi
  SRC_REL="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$SRC" "$SRC_ROOT")"
  git -C "$SRC_ROOT" diff --cached -U0 -- "$SRC_REL" > "$CHANGED_LINES_FILE.diff" || {
    echo "ERROR: 无法读取 staged diff" >&2
    exit 2
  }
  if [ ! -s "$CHANGED_LINES_FILE.diff" ]; then
    git -C "$SRC_ROOT" diff HEAD -U0 -- "$SRC_REL" > "$CHANGED_LINES_FILE.diff"
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
  if [ -s "$CHANGED_LINES_FILE" ]; then
    # 注意: 单行改动集无结尾换行, wc -l 会少算 1 行, 故用非空行计数
    echo "[mut] 变异范围: 改动行 $(tr -s ' \n' '\n' < "$CHANGED_LINES_FILE" | grep -c .) 行"
  else
    # 无差异（如 CI 干净检出）或差异不含代码行：删除白名单文件 →
    # apply_mut 以 allowed=None 走全文件，避免 MUTATION_SCORE=N/A 的静默空转
    rm -f "$CHANGED_LINES_FILE"
    echo "[mut] 变异范围: 无可定位改动行 → 回退全文件"
  fi
else
  echo "ERROR: MUTATION_SCOPE 仅支持 changed/all" >&2
  exit 2
fi

# 变异规则: 描述|正则|替换。用 python 应用(单点替换第1处匹配)
# 跳过区间: 字符串/注释(tokenize) + 类型注解区域(ast) + 返回箭头 ->。
# 普通规则与整数+1 共用同一段跳过/匹配逻辑, 避免两套偏移算法漂移。
apply_mut(){
  "$PY" - "$SRC" "$2" "$3" "$CHANGED_LINES_FILE" <<'PYMUT'
import sys, re, tokenize, io, ast
path, pat, rep, lines_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
code = open(path, encoding="utf-8").read()
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
    while idx != -1 and (in_skip(idx, skips) or not line_allowed(idx)):
        idx = code.find(pat, idx + 1)
    if idx == -1:
        sys.exit(2)
    new = code[:idx] + rep + code[idx + len(pat):]
else:
    m = rx.search(code)
    while m and (in_skip(m.start(), skips) or not line_allowed(m.start())):
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
    new = code[:m.start()] + sub + code[m.end():]
if new == code:
    sys.exit(2)
# 语法破坏型变异（如 -> 变 -<）不构成"测试有效性"的证据，按跳过处理（退出码 3）
try:
    ast.parse(new)
except SyntaxError:
    sys.exit(3)
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
  if sh -c "$PYTEST -x -q --no-header \"$TESTS\"" >/tmp/mut_run.log 2>&1; then
    survived=$((survived+1)); surv_list="$surv_list\n  - 存活: $desc"
  else
    killed=$((killed+1))
  fi
done

if [ "$applied" -eq 0 ]; then
  echo "[mut] 未产生有效变异"; echo "MUTATION_SCORE=N/A"; exit 0
fi
score=$((killed * 100 / applied))
echo "[mut] 杀死 $killed / 存活 $survived / 有效变异 $applied / 语法破坏跳过 $skipped"
echo "[mut] 变异分数: ${score}%"
[ -n "$surv_list" ] && { echo "[mut] 存活变异(测试未捕获):"; printf "$surv_list\n"; }
echo "MUTATION_SCORE=$score"
MIN="${MUTATION_MIN:-80}"
if [ "$score" -ge "$MIN" ]; then echo "[mut] ✅ 达标(≥${MIN}%)"; exit 0
else echo "[mut] ❌ 不达标(<${MIN}%), 测试套件有覆盖盲区"; exit 1; fi
