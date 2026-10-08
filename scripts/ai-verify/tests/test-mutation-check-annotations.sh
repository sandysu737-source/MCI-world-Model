#!/usr/bin/env bash
# 变异器类型注解回归套件 test-mutation-check-annotations.sh
# 用途: 修复 mutation-check 对 Python 类型注解的伪变异——先红灯后修复。
# 缺陷背景（P3 verifier 66% 门禁失败实证）:
#   D1 None→0 首先命中参数/返回注解中的 None, 而非可执行代码;
#   D2 `> 变 <` 把返回箭头 `->` 当成比较操作符, 制造语法错误被虚假计为"杀死";
#   D3 仅有类型注解的 None 也会产生无业务意义的"有效变异"。
#   另: accept.sh 后端 pytest 硬超时 120s 已低于当前全量实测 ~175s → 锁定 300s。
# 测试原则: 检查实际变异后的临时文件内容与工具输出, 不允许仅 grep 实现文字假绿;
#   临时源文件运行后必须逐字恢复, 无 .mutorig 残留。
# 运行要求: pytest 依次由 PYTEST_BIN / 仓库 venv / PATH 解析, 不依赖裸 python3;
#   解析不到则前置报错 fail-closed（避免把"命令不存在"伪造成"变异被杀死"）。
# 兼容: Bash 3.2（macOS 自带）, 算术展开中不得嵌入带引号的命令替换。
# 夹具口径: T9 断言依赖宿主项目仓的 governance/accept.sh（不在 kit 内下发）。
#   夹具缺失 → SKIP（不算失败）；夹具存在但取值不符 → FAIL（真回归不得被跳过掩盖）。
set -u
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MUT="$KIT_DIR/mutation-check.sh"
ACCEPT="$KIT_DIR/accept.sh"
PASS=0; FAIL=0; SKIP=0

say(){ printf '\033[1m[anno-test]\033[0m %s\n' "$1"; }
ok(){ say "✅ PASS  $1"; PASS=$((PASS+1)); }
skip(){ say "⏭️  SKIP  $1"; SKIP=$((SKIP+1)); }
no(){ say "❌ FAIL  $1"; FAIL=$((FAIL+1)); printf '%s\n' "$2" | tail -12; }

t="$(mktemp -d /tmp/gov-anno.XXXXXX)"
cleanup(){ rm -rf "$t" 2>/dev/null; }
trap cleanup EXIT

# ---------- pytest 解释器解析（不依赖 PATH 中的裸 python3） ----------
ROOT_DIR="$(cd "$KIT_DIR/../.." && pwd)"

resolve_pytest_bin(){
  # 1) 调用方显式提供的 PYTEST_BIN
  if [ -n "${PYTEST_BIN:-}" ]; then
    if [ -x "$PYTEST_BIN" ] || command -v "$PYTEST_BIN" >/dev/null 2>&1; then
      printf '%s' "$PYTEST_BIN"; return 0
    fi
    echo "ERROR: PYTEST_BIN 不可执行: $PYTEST_BIN" >&2
    return 1
  fi
  # 2) 仓库内 venv（POSIX bin/ 与 Windows Scripts/ 两种布局）
  for vd in backend/.venv .venv venv; do
    for p in "$ROOT_DIR/$vd/bin/pytest" "$ROOT_DIR/$vd/Scripts/pytest.exe"; do
      [ -x "$p" ] && { printf '%s' "$p"; return 0; }
    done
  done
  # 3) PATH
  if command -v pytest >/dev/null 2>&1; then
    command -v pytest; return 0
  fi
  return 1
}

if ! RESOLVED_PYTEST_BIN="$(resolve_pytest_bin)"; then
  echo "ERROR: 未找到可用 pytest; 请设 PYTEST_BIN（如 backend/.venv/bin/pytest）后重试。" >&2
  exit 2
fi
export RESOLVED_PYTEST_BIN

# 尊重 mutation-check.sh 已有的 PYTHON_BIN（它用该解释器跑 PYMUT）;
# 调用方未设置时, 从 pytest 同目录推导 venv 解释器。
if [ -z "${PYTHON_BIN:-}" ]; then
  PYB_DIR="$(dirname "$RESOLVED_PYTEST_BIN")"
  for cand in "$PYB_DIR/python3" "$PYB_DIR/python" "$PYB_DIR/python3.exe" "$PYB_DIR/python.exe"; do
    [ -x "$cand" ] && { PYTHON_BIN="$cand"; break; }
  done
fi
[ -n "${PYTHON_BIN:-}" ] && export PYTHON_BIN
say "pytest 解释器: $RESOLVED_PYTEST_BIN${PYTHON_BIN:+  (PYTHON_BIN=$PYTHON_BIN)}"

# 记录型 pytest 包装: 每次被调用时快照当前(可能已变异的)源文件与真实 pytest 退出码
cat > "$t/wpytest.sh" <<'EOF'
#!/usr/bin/env bash
# 先读取并校验计数器, 再算 n; 不在算术展开里嵌带引号的命令替换
# （Bash 3.2 会对 $(( "$(cat f)" + 1 )) 报 syntax error, 令快照全部塌缩到 snap-.py）。
counter="$(cat "$SNAP_DIR/counter" 2>/dev/null || echo 0)"
case "$counter" in
  ''|*[!0-9]*) counter=0 ;;
esac
n=$((counter + 1))
echo "$n" > "$SNAP_DIR/counter"
cp "$SRC_PATH" "$SNAP_DIR/snap-$n.py"
# fail-closed: 解析不到 pytest 时必须报错, 不得让"命令不存在"伪装成"变异被杀死"。
if [ -z "${RESOLVED_PYTEST_BIN:-}" ]; then
  echo "wpytest: RESOLVED_PYTEST_BIN 未设置" >&2
  exit 127
fi
"$RESOLVED_PYTEST_BIN" "$@"
rc=$?
echo "$rc" > "$SNAP_DIR/snap-$n.rc"
exit $rc
EOF
chmod +x "$t/wpytest.sh"

run_mut_case(){
  local src="$1" tests="$2"
  # 沙箱源文件不在 git 仓库内：显式声明全文件范围，避免 changed 模式 fail-closed 报错
  ( cd "$t" && SRC_PATH="$src" SNAP_DIR="$t" PYTHONUTF8=1 \
    PYTHON_BIN="${PYTHON_BIN:-}" \
    MUTATION_SCOPE=all \
    bash "$MUT" "$src" "$tests" "$t/wpytest.sh" 2>&1 )
}

snap_grep(){ grep -l "$1" "$t"/snap-*.py 2>/dev/null | head -1; }

# ---------- D1/D2: 注解与箭头混入可执行变异 ----------
say "T8a: 多行注解+UTF-8 注释文件（None→0 应命中可执行区, 箭头不得被变异）"
cat > "$t/mut_anno.py" <<'EOF'
# 中文注释：UTF-8 偏移验证 αβγ
def annotated(
    x: int | None,
) -> None:
    """多行注解示例：中文文本 αβγ"""
    return None

def compare(a: int, b: int) -> bool:
    if a > b + 10:
        return True
    return False
EOF
cat > "$t/test_mut_anno.py" <<'EOF'
from mut_anno import annotated, compare

def test_annotated():
    assert annotated(1) is None

def test_compare():
    assert compare(13, 2) is True
    assert compare(2, 13) is False
EOF
cp "$t/mut_anno.py" "$t/mut_anno.before"
out8a="$(run_mut_case "$t/mut_anno.py" "$t/test_mut_anno.py")"
# RED D1a: None→0 命中参数注解（snap 出现 int | 0）且可执行 return None 未被动过
if [ -n "$(snap_grep 'int | 0')" ] && [ -n "$(snap_grep 'return None')" ]; then
  no "T8a-1 [RED] None→0 首先命中参数注解 x: int | None（可执行 return None 未变异）" "$(snap_grep 'int | 0'; cat "$(snap_grep 'int | 0')" 2>/dev/null)"
else
  ok "T8a-1 [GREEN] 注解 None 不再变异, None→0 命中可执行代码"
fi
# RED D2a: 返回箭头 `->` 被当比较符变异（`[^<>=!]>(?!=)` 连同前置 - 一起替换, -> None 变 < None, 语法错误被虚假计为杀死）
if [ -n "$(snap_grep ') < None')" ]; then
  no "T8a-2 [RED] 返回箭头 -> 被变异为 < （语法错误被虚假计为杀死）" "$(cat "$(snap_grep ') < None')" 2>/dev/null | head -8)"
else
  ok "T8a-2 [GREEN] 返回箭头不再产生变异"
fi
# GREEN: 可执行 None 仍被变异为 0（return 0）且被杀死
if [ -n "$(snap_grep 'return 0')" ]; then
  ok "T8a-3 可执行 return None 仍可变异（出现 return 0 快照）"
else
  no "T8a-3 修复不得把可执行 None 一起屏蔽（应出现 return 0 快照）" "$out8a"
fi
# GREEN: 真实比较仍被变异（> 规则连前置空白一起替换 → 快照形如 a< b / a < b）
if grep -lE 'a< ?b' "$t"/snap-*.py >/dev/null 2>&1; then
  ok "T8a-4 可执行 > 比较仍可变异（出现 a< b 快照）"
else
  no "T8a-4 修复不得屏蔽真实比较变异（应出现 a< b 快照）" "$out8a"
fi
# GREEN: 可执行整数常量仍被 +1 变异（(10+1) 快照）
if [ -n "$(snap_grep '(10+1)')" ]; then
  ok "T8a-5 可执行整数常量仍可 +1 变异（出现 (10+1) 快照）"
else
  no "T8a-5 可执行整数 10 应仍可变异为 (10+1)" "$out8a"
fi
# GREEN: 全部可执行变异被杀死 → 分数 100%
if printf '%s' "$out8a" | grep -q 'MUTATION_SCORE=100'; then
  ok "T8a-6 可执行变异全部被杀死（MUTATION_SCORE=100）"
else
  no "T8a-6 修复后该文件变异分数应为 100%（可执行 None/比较/布尔/整数均被杀死）" "$out8a"
fi
# 恢复与残留
if cmp -s "$t/mut_anno.before" "$t/mut_anno.py"; then
  ok "T8a-7 源文件逐字恢复（trap 恢复）"
else
  no "T8a-7 源文件未恢复" "$(diff "$t/mut_anno.before" "$t/mut_anno.py" | head -12)"
fi
if [ -z "$(find "$t" -maxdepth 1 -name '*.mutorig' -print -quit)" ]; then
  ok "T8a-8 无 .mutorig 残留"
else
  no "T8a-8 存在 .mutorig 残留" "$(find "$t" -maxdepth 1 -name '*.mutorig')"
fi

# ---------- D3: 仅注解文件不得产生无业务意义的有效变异 ----------
say "T8b: 仅类型注解文件（无可执行 None → 不得产生有效变异）"
cat > "$t/anno_only.py" <<'EOF'
def annotated(x: int) -> None:
    pass
EOF
cat > "$t/test_anno_only.py" <<'EOF'
from anno_only import annotated

def test_annotated():
    assert annotated(1) is None
EOF
cp "$t/anno_only.py" "$t/anno_only.before"
out8b="$(run_mut_case "$t/anno_only.py" "$t/test_anno_only.py")"
# RED: 注解 -> None 被变异为 -> 0 且存活（无业务意义的有效变异）
if [ -n "$(snap_grep '\-> 0')" ] && printf '%s' "$out8b" | grep -q '存活: None 变 0'; then
  no "T8b-1 [RED] 仅注解 None 产生无业务意义的有效变异且存活" "$(cat "$(snap_grep '\-> 0')" 2>/dev/null)"
else
  ok "T8b-1 [GREEN] 仅注解文件不再产生 None 有效变异"
fi
if printf '%s' "$out8b" | grep -q '未产生有效变异'; then
  ok "T8b-2 仅注解文件报告未产生有效变异（MUTATION_SCORE=N/A 语义保留）"
else
  no "T8b-2 无可执行变异 token 时应报告「未产生有效变异」" "$out8b"
fi
if cmp -s "$t/anno_only.before" "$t/anno_only.py"; then
  ok "T8b-3 源文件逐字恢复"
else
  no "T8b-3 源文件未恢复" "$(diff "$t/anno_only.before" "$t/anno_only.py" | head -12)"
fi

# ---------- accept.sh 后端 pytest 硬超时 ----------
say "T9: accept.sh 后端 pytest 硬超时（120s → 300s, 配置断言 + run_with_timeout 行为）"
if [ ! -f "$ACCEPT" ]; then
  # kit 独立自检场景：accept.sh 属项目侧产物，kit 目录内本就不存在
  skip "T9 accept.sh 夹具缺失（${ACCEPT}）；该断言针对宿主项目仓，kit 自检跳过"
else
  # T9a 配置断言: 当前(RED)为 120, 目标锁定 300
  if grep -qE 'run_with_timeout 120' "$ACCEPT" || ! grep -qE 'run_with_timeout 300' "$ACCEPT"; then
    no "T9a [RED] accept.sh 后端 pytest 超时应为 300 秒（当前: $(grep -oE 'run_with_timeout [0-9]+' "$ACCEPT" | head -1)）" \
       "$(grep -n 'run_with_timeout' "$ACCEPT")"
  else
    ok "T9a [GREEN] accept.sh 后端 pytest 超时已锁定 300 秒"
  fi
  # T9b 行为验证: 从 accept.sh 提取 run_with_timeout 实定义做短超时行为测试（不等 300s）
  awk '/^run_with_timeout\(\)/{f=1} f{print} f&&/^\}/{exit}' "$ACCEPT" > "$t/rwt_extract.sh"
  if [ -s "$t/rwt_extract.sh" ]; then
    ( . "$t/rwt_extract.sh"
      run_with_timeout 1 sleep 3 >/dev/null 2>&1; r1=$?
      run_with_timeout 2 true >/dev/null 2>&1; r2=$?
      # 超时表现为被 kill 后的非零退出（124 或 128+15, 平台相关）; 正常完成必须 rc=0
      if [ "$r1" -ne 0 ] && [ "$r2" -eq 0 ]; then exit 0; else exit 1; fi
    )
    if [ $? -eq 0 ]; then
      ok "T9b run_with_timeout 行为正确（超时 rc=124, 正常完成 rc=0）"
    else
      no "T9b run_with_timeout 行为异常（期望超时 124 / 正常 0）" "$(cat "$t/rwt_extract.sh")"
    fi
  else
    no "T9b 未能从 accept.sh 提取 run_with_timeout 定义" "$(grep -n 'run_with_timeout' "$ACCEPT")"
  fi
fi


# ---------- M5(OODA-20261007-053): N/A 显式裁决（原因码 + 严格模式人工确认） ----------
# 缺陷背景：`MUTATION_SCORE=N/A`（改动行内 0 个 fitting 变异算子）曾被 quality-gate 的旧正则
#   `[0-9NA]+` 截成 "N"，渲染成假分数「N% ≥ 80%」；「无变异可评」被静默贴上达标证书。
#   量化（2026-10-09）：受约束文件 26/300 任何改动必落 N/A，近 40 提交 15 个改动单元中 5 个落 N/A
#   → 不 fail-closed（只会制造假红），改为显式原因码 + L0/L2 转人工确认。
say "T10: N/A 的显式裁决（原因码 + 人工确认通道，不得静默记达标）"

# T10a 引擎层：无可评变异（T8b 同一样本）必须携带机器可读原因码
if printf '%s' "$out8b" | grep -qx 'MUTATION_NA_REASON=no-operator-in-changed-lines'; then
  ok "T10a 无可评变异输出原因码 no-operator-in-changed-lines"
else
  no "T10a 无可评变异缺机器可读原因码（无法区分合法 N/A 与静默放行）" "$out8b"
fi

# T10b 引擎层：纯删除变更（新文件侧无新增行）→ 专用原因码，且不得回落全文件
# 注意：macOS 的 /tmp 是 /private/tmp 的符号链接，夹具必须取物理路径（pwd -P），
# 否则 mutation-check 的 `relpath` 与 git 的 toplevel 不同源，diff 读不到 → 假 SKIP。
T_REAL="$(cd "$t" && pwd -P)"
DEL="$T_REAL/delrepo"
mkdir -p "$DEL"
git -C "$DEL" init -q
git -C "$DEL" config user.email anno@example.com
git -C "$DEL" config user.name anno
cat > "$DEL/del_only.py" <<'EOF'
def total(a):
    unused = 0
    return a
EOF
cat > "$DEL/test_del_only.py" <<'EOF'
from del_only import total


def test_total():
    assert total(1) == 1
EOF
git -C "$DEL" add -A
git -C "$DEL" commit -qm base
cat > "$DEL/del_only.py" <<'EOF'
def total(a):
    return a
EOF
out10b="$( cd "$DEL" && SRC_PATH="$DEL/del_only.py" SNAP_DIR="$t" PYTHONUTF8=1 \
  PYTHON_BIN="${PYTHON_BIN:-}" MUTATION_SCOPE=changed \
  bash "$MUT" "$DEL/del_only.py" "$DEL/test_del_only.py" "$t/wpytest.sh" 2>&1 )"
if printf '%s' "$out10b" | grep -qx 'MUTATION_NA_REASON=pure-delete'; then
  ok "T10b 纯删除变更输出专用原因码 pure-delete"
else
  no "T10b 纯删除变更未输出 pure-delete（合法 N/A 不得回落全文件）" "$out10b"
fi
if printf '%s' "$out10b" | grep -q '纯删除变更'; then
  ok "T10b-2 纯删除变更保留语义说明"
else
  no "T10b-2 纯删除变更缺语义说明" "$out10b"
fi

# T10c/T10d 门控层：严格（L0/L2 合入拦截点）必须转人工确认，非严格（push 审计）不阻断
GATE="$KIT_DIR/quality-gate.sh"
if [ ! -f "$GATE" ]; then
  skip "T10c 缺 $GATE，跳过门控层行为验证"
elif ! command -v radon >/dev/null 2>&1 || ! command -v ruff >/dev/null 2>&1 \
  || ! command -v bandit >/dev/null 2>&1; then
  skip "T10c 缺 radon/ruff/bandit，跳过门控层行为验证"
else
  GF="$t/gatefix"
  mkdir -p "$GF/tests"
  git -C "$GF" init -q
  git -C "$GF" config user.email gate@example.com
  git -C "$GF" config user.name gate
  cat > "$GF/pyproject.toml" <<'EOF'
[tool.pytest.ini_options]
pythonpath = ["."]
EOF
  cat > "$GF/na_mod.py" <<'EOF'
"""声明类模块：无可评变异（T10c 夹具）。"""

from enum import Enum


class Cat(Enum):
    """分类枚举。"""

    A = "a"
    B = "b"
EOF
  cat > "$GF/tests/test_na_mod.py" <<'EOF'
from na_mod import Cat


def test_cat():
    assert Cat.A.value == "a"
EOF
  git -C "$GF" add -A
  git -C "$GF" commit -qm base
  out10c="$( cd "$GF" && PYTHONUTF8=1 PYTHON_BIN="${PYTHON_BIN:-}" \
    MUTATION_STRICT=1 COVERAGE_HARD_GATE=0 PYTEST_SCOPE=changed \
    bash "$GATE" L2 na_mod.py 2>&1 )"
  rc10c=$?
  if printf '%s' "$out10c" | grep -q '工具完整性校验未通过'; then
    skip "T10c 本机工具完整性校验未通过（环境问题，非回归）"
  elif [ "$rc10c" -ne 0 ]; then
    no "T10c 严格模式 N/A 不应硬阻断（期望 rc=0，转人工确认；fail-closed 会制造假红）" "$out10c"
  elif printf '%s' "$out10c" | grep -q '原因=no-operator-in-changed-lines' \
    && printf '%s' "$out10c" | grep -q '不得记为达标'; then
    ok "T10c 严格模式 N/A 显式转人工确认（原因码可见，非达标）"
  else
    no "T10c 严格模式 N/A 未显式标注（疑似静默记达标）" "$out10c"
  fi
  # T10d 仅在门控实现了 MUTATION_STRICT 降级开关时断言（能力探测，非实现文字断言）。
  #   该开关目前只在仓库副本（F-22/push 合并后审计降级）；kit 源尚未回灌该开关，
  #   其自带 CI 模板也未注入 → kit 自检只覆盖严格口径，降级口径待 M6 统一。
  if ! grep -q 'MUTATION_STRICT' "$GATE"; then
    skip "T10d 该门控未实现 MUTATION_STRICT 降级开关（kit 源待回灌），跳过降级口径断言"
  else
    out10d="$( cd "$GF" && PYTHONUTF8=1 PYTHON_BIN="${PYTHON_BIN:-}" \
      MUTATION_STRICT=0 COVERAGE_HARD_GATE=0 PYTEST_SCOPE=changed \
      bash "$GATE" L2 na_mod.py 2>&1 )"
    rc10d=$?
    if [ "$rc10d" -eq 0 ] && printf '%s' "$out10d" | grep -q 'push 合并后审计不阻断'; then
      ok "T10d 非严格（push 审计）N/A 与 PR 拦截口径区分"
    else
      no "T10d 非严格模式 N/A 口径异常" "$out10d"
    fi
  fi
fi

echo
say "汇总: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
