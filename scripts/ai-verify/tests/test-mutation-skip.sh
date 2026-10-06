#!/usr/bin/env bash
# F-23(P0-J) 回归：变异检查器的"不计分位点"口径
# 目的：锁定 注释 / 字符串 / f-string 整段 / 类型注解 / 返回箭头 位点不计分，
#       代码位点正常计分，"语法破坏型"变异不计分（否则分数虚高）。
# 背景：仓库副本曾因偏移量重复计换行（splitlines(True) 已含 \n 又 +1）+ Python ≥3.12
#       把 f-string 拆成 FSTRING_* token，把注释与 f-string 里的运算符当成可变异位点，
#       使 benchmarks/real_world/p03_metrics.py 的分数被结构性压低（CI main push 常红）。
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CHECK="$ROOT/scripts/ai-verify/mutation-check.sh"
PY="${PYTHON_BIN:-python3}"
PYTEST_CMD="${MUT_PYTEST_CMD:-pytest} --override-ini=addopts="
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail(){ echo "❌ $1" >&2; [ -n "${2:-}" ] && printf '%s\n' "$2" >&2; exit 1; }
ok(){ echo "✅ $1"; }

[ -f "$CHECK" ] || fail "缺少 $CHECK"

cat > "$TMP/test_sample.py" <<'PY'
def test_always_passes() -> None:
    assert True
PY

# 只含"不计分位点"的样本：注释 / 模块文档串 / f-string / 类型注解 / 返回箭头
cat > "$TMP/skip_only.py" <<'PY'
"""模块文档: 0 <= 1 且 2 >= 1。"""


def f(value: int) -> int:
    # 注释: value <= 1 且 value >= 0
    message = f"界限: {value} <= 1 且 >= 0"
    del message
    return value + 1
PY

# 代码位点样本：比较 / 布尔 都应正常计分
cat > "$TMP/code_sites.py" <<'PY'
def g(a: int, b: int) -> bool:
    if a <= b:
        return True
    return b > a
PY

# 语法破坏型样本：`a << 2` 的 `<` 变异成 `a >< 2` 无法 ast.parse
cat > "$TMP/syntax_break.py" <<'PY'
def s(a: int) -> int:
    return a << 2
PY

run_check(){  # $1=源文件
  MUTATION_SCOPE=all MUTATION_MIN=0 PYTHON_BIN="$PY" \
    bash "$CHECK" "$1" "$TMP/test_sample.py" "$PYTEST_CMD" 2>&1
}
field(){  # $1=输出 $2=字段
  printf '%s' "$1" | grep -oE "[0-9]+ / 存活 [0-9]+ / 有效变异 [0-9]+ / 语法破坏跳过 [0-9]+" \
    | grep -oE "(有效变异|语法破坏跳过) [0-9]+" | grep "$2" | grep -oE '[0-9]+$'
}

out="$(run_check "$TMP/skip_only.py")"
applied="$(field "$out" '有效变异')"
[ "${applied:-x}" = "1" ] || fail "注释/字符串/f-string/注解位点未被排除：有效变异=${applied:-?}（应为 1）" "$out"
skipped="$(field "$out" '语法破坏跳过')"
[ "${skipped:-x}" = "0" ] || fail "skip_only 样本不应产生语法破坏型变异：${skipped:-?}" "$out"
ok "注释/字符串/f-string/注解位点不计分（有效变异=1）"

out="$(run_check "$TMP/code_sites.py")"
applied="$(field "$out" '有效变异')"
[ "${applied:-x}" = "3" ] || fail "代码位点计分口径异常：有效变异=${applied:-?}（应为 3）" "$out"
ok "代码位点正常计分（有效变异=3）"

out="$(run_check "$TMP/syntax_break.py")"
applied="$(field "$out" '有效变异')"
skipped="$(field "$out" '语法破坏跳过')"
[ "${applied:-x}" = "1" ] || fail "语法破坏型样本的有效变异数异常：${applied:-?}（应为 1）" "$out"
[ "${skipped:-x}" = "1" ] || fail "语法破坏型变异未被单独计数：${skipped:-?}（应为 1）" "$out"
ok "语法破坏型变异不计分（有效变异=1，跳过=1）"

echo "🎉 test-mutation-skip.sh 全部通过"
