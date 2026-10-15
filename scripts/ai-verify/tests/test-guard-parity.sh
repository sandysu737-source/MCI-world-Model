#!/usr/bin/env bash
# 门禁引擎双载体一致性回归（OODA-20260930-002 收口）
#
# 背景：mutation-check.sh 曾在 kit 与仓库副本各修一半（kit 有改动行范围无注解跳过，
# repo 有注解跳过无范围限定），导致"本地门禁绿 / CI 红"。本套件把"两载体一致"钉成断言：
#   1) 仓库副本 hash == 期望值 —— CI 无 kit 也能拦住副本被静默改动；
#   2) kit 存在时字节级比对 kit 与仓库副本 —— 双载体不得再分叉；
#   3) 行为回归：无差异（CI 干净检出）→ 回退全文件并产出真实分数，不得空转成 N/A；
#      有 staged 差异 → 只变异改动行；注释改动不产生假变异；
#      索引过期（staged=旧快照、工作区=新内容）→ 按工作区取行并告警，不得空转成 N/A
#      （OODA-20261007-103：曾用 `git diff --cached`，错位的行号使变异点全部落空）。
# 用法: bash scripts/ai-verify/tests/test-guard-parity.sh
# 退出码: 0=全部通过 1=有失败 2=环境错误
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
KIT_DIR="${AI_ENG_KIT_GUARD_DIR:-$HOME/qoder m5pro/_ai-eng-kit/governance}"

# ── 载体定位（K-10 修复，2026-10-15）──────────────────────────────────────────
# 原实现写死 $ROOT/scripts/ai-verify/mutation-check.sh ⇒ 在 kit 内必报「缺少仓库副本」
# 并 exit 2，使纪律「改 governance 脚本先跑本套件」在**权威源不可执行**（套件是仓锚定的）。
# 现口径：仓副本优先；无 scripts/ai-verify/ 但存在 governance/ ⇒ kit 自检模式。
KIT_SELF=0
if [ -f "$ROOT/scripts/ai-verify/mutation-check.sh" ]; then
  MUT_REPO="$ROOT/scripts/ai-verify/mutation-check.sh"
elif [ -f "$ROOT/governance/mutation-check.sh" ]; then
  MUT_REPO="$ROOT/governance/mutation-check.sh"
  KIT_SELF=1
  [ -n "${AI_ENG_KIT_GUARD_DIR:-}" ] || KIT_DIR="$ROOT/governance"
fi
MUT_KIT="$KIT_DIR/mutation-check.sh"
# 期望值：以 kit 为源同步后的引擎 hash 字面量（历史遗产，见下方断言③的口径订正）
# 2026-10-07（OODA-20261007-124）：纯删除变更显式判 N/A，不再回退全文件（消假红）
# 2026-10-08（kit 7e02432）：回灌 mci-world-model 三项修复（F-23 f-string/语法破坏跳过、
#   RISK_BRANCH、F-20 阈值 env），引擎 hash 随之升级
# 2026-10-07（OODA-20261007-103）：行号口径由 `git diff --cached` 改为 `git diff HEAD`
# （工作区 vs HEAD），并新增「索引与工作区不一致」告警
# 2026-10-09（M5 / OODA-20261007-053）：mutation-check 输出 N/A 机器可读原因码
#   （MUTATION_NA_REASON=pure-delete|no-operator-in-changed-lines），供 quality-gate
#   区分「合法无可评变异」与「静默放行」并在严格模式转人工确认。引擎 hash 同步升级。
# 2026-10-09（M6 / OODA-20261009-WAVE1-348）：门禁认波次基线（GATE_DIFF_BASE /
#   MUTATION_DIFF_BASE）+ 变异存活点带行号（kit 381345f）。该提交漏更新本期望值，
#   致 parity 套件假红（期望 7712c58e ≠ 引擎 8a15d9f4），此处补钉为回灌后引擎 hash。
# 2026-10-15（U6 订正）：本常量**不再是无条件硬断言**。它只在外层锚缺席（CI 无 kit）时兜底；
#   历史问题＝它与用例名「双载体一致性（kit↔repo）」不符，kit 每次合法演进都造成各仓假红，
#   已被人工补钉至少 3 次（a6c41e3 / 002cf84 / 337f9cc 一类）。真正的锚见断言①②。
EXPECTED_MUTATION_ENGINE_SHA="f5b33b22f08d25d4ff8e619068548ea0e8863570bb9469456851838f6d23eb91"
# 维护规则：本常量 = kit 单源**当前**引擎 hash（与 kit-baseline.tsv 同源同义），只服务
# 「CI 无 kit」这一条通道；有 kit 时判定锚是 kit 本身（断言②），字面量陈旧仅告警（④）。

PASS=0; FAIL=0; SKIP=0; WARN=0
say(){ printf '\033[1m[parity]\033[0m %s\n' "$1"; }
sha(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
pin_from_baseline(){
  [ -f "$KIT_DIR/kit-baseline.tsv" ] || return 1
  awk -F'\t' '$1=="mutation-check.sh"{print $2; exit}' "$KIT_DIR/kit-baseline.tsv"
}

if [ ! -f "$MUT_REPO" ]; then
  say "❌ ERROR 缺少载体（scripts/ai-verify/ 与 governance/ 均无 mutation-check.sh）: $ROOT"
  exit 2
fi
repo_sha="$(sha "$MUT_REPO")"

# ① kit 自检模式的锚：kit 基线表。kit 载体的任何改动都必须**显式 re-anchor**，不得静默。
if [ "$KIT_SELF" -eq 1 ]; then
  pin="$(pin_from_baseline || true)"
  if [ -z "$pin" ]; then
    say "❌ FAIL  kit 自检模式缺基线锚: $KIT_DIR/kit-baseline.tsv（先跑 check-kit-parity.py --write-baseline）"
    FAIL=$((FAIL+1))
  elif [ "$repo_sha" = "$pin" ]; then
    say "✅ PASS  kit 载体与基线一致（$(printf '%s' "$repo_sha" | cut -c1-16)）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  kit 载体与基线不一致：$repo_sha ≠ $pin"
    say "         若为有意升级：先改 kit 源 → 复算 → check-kit-parity.py --write-baseline → 再回灌各仓"
    FAIL=$((FAIL+1))
  fi
fi

# ② kit↔repo 双载体一致（本地有 kit 才比对）。这是用例名的本义，也是硬断言。
if [ -f "$MUT_KIT" ]; then
  kit_sha="$(sha "$MUT_KIT")"
  if [ "$kit_sha" = "$repo_sha" ]; then
    say "✅ PASS  kit 与仓库副本字节一致"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  双载体已分叉：kit=$(printf '%s' "$kit_sha" | cut -c1-16) repo=$(printf '%s' "$repo_sha" | cut -c1-16)（重跑 sync-governance.sh）"
    FAIL=$((FAIL+1))
  fi
else
  # ③ 无 kit（CI）：字面量是唯一外部锚，保留硬断言——CI 仍能拦住「副本被静默改动」。
  if [ "$repo_sha" = "$EXPECTED_MUTATION_ENGINE_SHA" ]; then
    say "✅ PASS  仓库副本 hash 与记录一致（$(printf '%s' "$repo_sha" | cut -c1-16)）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  仓库副本与记录 hash 不一致：$repo_sha ≠ $EXPECTED_MUTATION_ENGINE_SHA"
    say "         引擎升级：先改 kit 源 → bash _ai-eng-kit/sync-governance.sh <项目> → 更新本常量"
    FAIL=$((FAIL+1))
  fi
fi

# ④ 钉值滞后只告警：有 kit 时锚已是 kit 本身，字面量陈旧不再制造假红（U6）。
if [ -f "$MUT_KIT" ] && [ "$repo_sha" = "$(sha "$MUT_KIT")" ] \
   && [ "$repo_sha" != "$EXPECTED_MUTATION_ENGINE_SHA" ]; then
  say "⚠️  WARN  字面量钉值滞后（repo=$(printf '%s' "$repo_sha" | cut -c1-16) vs 常量=$(printf '%s' "$EXPECTED_MUTATION_ENGINE_SHA" | cut -c1-16)）；有 kit 时不影响判定，建议顺手更新"
  WARN=$((WARN+1))
fi

# 3) 行为回归（沙箱 git 仓库，不触碰真实仓库）
PYBIN="${PYTHON_BIN:-$(command -v python3 || true)}"
# `command -v` 结果可能含空格（本工作区所有项目路径形如 `/Users/mac/qoder m5pro/<仓>/…`）。
# 该值随后被**拼进一个字符串**交给 mutation-check（其内 shlex 二次切分）⇒ 必须显式单引号化，
# 否则会被切成 `/Users/mac/qoder` 并当作命令执行（2026-10-16 实证：6 条行为判据连坐 FAIL）。
PYTESTBIN="$(command -v pytest || true)"
PYTEST_PREFIX="'$PYTESTBIN' --override-ini=addopts= -p no:cacheprovider"
if [ -z "$PYBIN" ] || [ -z "$PYTESTBIN" ]; then
  say "ℹ️  SKIP  缺少 python3/pytest，跳过行为回归"
else
  SANDBOX="$(mktemp -d /tmp/guard-parity.XXXXXX)" || { say "❌ ERROR 无法创建沙箱"; exit 2; }
  trap 'rm -rf "${SANDBOX:-}"' EXIT
  git -C "$SANDBOX" init -q
  git -C "$SANDBOX" config user.email parity@example.com
  git -C "$SANDBOX" config user.name parity
  cat > "$SANDBOX/sample.py" <<'EOF'
"""沙箱样本：两个变异点（比较符 + 数值常量）。"""


def limit():
    return 2


def sign(x):
    if x > 0:
        return 1
    return -1
EOF
  cat > "$SANDBOX/test_sample.py" <<'EOF'
from sample import limit, sign


def test_limit():
    assert limit() == 2


def test_sign():
    assert sign(2) == 1
    assert sign(-2) == -1
EOF
  git -C "$SANDBOX" add sample.py test_sample.py
  git -C "$SANDBOX" -c commit.gpgsign=false commit -qm init

  # 引擎固定走仓库副本；pytest 前缀显式清空 addopts，避免继承真实仓库配置
  run_engine(){ ( cd "$SANDBOX" && PYTHON_BIN="$PYBIN" MUTATION_MIN=80 \
    bash "$MUT_REPO" sample.py test_sample.py "$PYTEST_PREFIX" 2>&1 ); }

  # 用例A: 无任何差异（等价 CI 干净检出）→ 回退全文件，产出真实分数而非 N/A
  out_a="$(run_engine)"
  if printf '%s' "$out_a" | grep -q '回退全文件' \
    && printf '%s' "$out_a" | grep -q 'MUTATION_SCORE=100'; then
    say "✅ PASS  无差异 → 回退全文件并产出真实分数（100%）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  无差异时未回退全文件或分数异常（门禁空转风险）"
    printf '%s\n' "$out_a" | tail -8
    FAIL=$((FAIL+1))
  fi

  # 用例D: 显式 MUTATION_SCOPE=all → 全文件变异，不得因空白名单文件空转
  out_d="$( cd "$SANDBOX" && MUTATION_SCOPE=all PYTHON_BIN="$PYBIN" MUTATION_MIN=80 \
    bash "$MUT_REPO" sample.py test_sample.py "$PYTEST_PREFIX" 2>&1 )"
  if printf '%s' "$out_d" | grep -q '全文件（显式 MUTATION_SCOPE=all）' \
    && printf '%s' "$out_d" | grep -q 'MUTATION_SCORE=100'; then
    say "✅ PASS  显式 all 模式全文件变异（未空转成 N/A）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  all 模式空转：空白名单被误当空集合，所有变异点被跳过"
    printf '%s\n' "$out_d" | tail -8
    FAIL=$((FAIL+1))
  fi

  # 用例B: staged 差异只是注释 → 只变异该行且被注释跳过 → 不产生假变异
  printf '\n# 无害注释 123\n' >> "$SANDBOX/sample.py"
  git -C "$SANDBOX" add sample.py
  out_b="$(run_engine)"
  if printf '%s' "$out_b" | grep -q '未产生有效变异' \
    && printf '%s' "$out_b" | grep -q 'MUTATION_SCORE=N/A'; then
    say "✅ PASS  注释改动行不产生假变异"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  注释改动行产生假变异或未按改动行限定"
    printf '%s\n' "$out_b" | tail -8
    FAIL=$((FAIL+1))
  fi

  # 用例C: staged 差异命中数值常量行 → 只变异改动行（1 个有效变异）且被杀
  rm -f "$SANDBOX/sample.py"
  cat > "$SANDBOX/sample.py" <<'EOF'
"""沙箱样本：两个变异点（比较符 + 数值常量）。"""


def limit():
    return 2  # 支付阈值


def sign(x):
    if x > 0:
        return 1
    return -1
EOF
  git -C "$SANDBOX" add sample.py
  out_c="$(run_engine)"
  if printf '%s' "$out_c" | grep -q '改动行 1 行' \
    && printf '%s' "$out_c" | grep -q '杀死 1 / 存活 0 / 有效变异 1' \
    && printf '%s' "$out_c" | grep -q 'MUTATION_SCORE=100'; then
    say "✅ PASS  staged 差异只变异改动行（1 个有效变异，已杀）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  改动行范围限定失效（可能变异了未改动行）"
    printf '%s\n' "$out_c" | tail -8
    FAIL=$((FAIL+1))
  fi

  # 用例E: 索引过期（staged=旧快照，工作区=新内容）→ 必须按工作区取行并告警，不得空转成 N/A
  # 场景来源：OODA-20261007-103 —— pre-commit 失败后改了代码未重新 `git add`，引擎按旧索引
  # 行号定位变异点，全部落空 → 输出 MUTATION_SCORE=N/A 且被 quality-gate 记 pass（静默空转）。
  cat > "$SANDBOX/sample.py" <<'EOF'
# 备注 A
# 备注 B
"""沙箱样本：两个变异点（比较符 + 数值常量）。"""


def limit():
    return 2


def sign(x):
    if x > 0:
        return 1
    return -1
EOF
  git -C "$SANDBOX" add sample.py
  # 故意不再 add：工作区把常量改成 `1 + 1`（基准测试仍通过，且该行 1 个变异必被杀），
  # 索引仍停留在「只有注释改动」的旧快照 → 旧实现按 {1,2} 取行会得到 N/A
  cat > "$SANDBOX/sample.py" <<'EOF'
# 备注 A
# 备注 B
"""沙箱样本：两个变异点（比较符 + 数值常量）。"""


def limit():
    return 1 + 1


def sign(x):
    if x > 0:
        return 1
    return -1
EOF
  out_e="$(run_engine)"
  if printf '%s' "$out_e" | grep -q '索引与工作区不一致' \
    && printf '%s' "$out_e" | grep -q 'MUTATION_SCORE=100'; then
    say "✅ PASS  索引过期时按工作区取行并告警（未空转成 N/A）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  索引过期场景空转或未告警（旧实现输出 MUTATION_SCORE=N/A）"
    printf '%s\n' "$out_e" | tail -8
    FAIL=$((FAIL+1))
  fi

  # 用例F: 纯删除变更（新文件侧无新增行）→ 必须显式判 N/A，不得回退全文件
  # 场景来源：OODA-20261007-124 —— 删除冗余声明（如模型里重复的 Index(...)）时，
  # 旧实现「无可定位改动行 → 回退全文件」，把未改动历史代码的覆盖盲区记成本次分数，
  # 基准测试明明通过却报「测试套件有覆盖盲区」＝治理报告里的假结论。
  git -C "$SANDBOX" add sample.py
  git -C "$SANDBOX" -c commit.gpgsign=false commit -qm base
  python3 - "$SANDBOX/sample.py" <<'PYDEL'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
assert "# 备注 A\n# 备注 B\n" in text
p.write_text(text.replace("# 备注 A\n# 备注 B\n", "", 1), encoding="utf-8")
PYDEL
  out_f="$(run_engine)"
  if printf '%s' "$out_f" | grep -q '纯删除变更（新文件侧无可变异行）' \
    && printf '%s' "$out_f" | grep -q 'MUTATION_SCORE=N/A' \
    && ! printf '%s' "$out_f" | grep -q '回退全文件'; then
    say "✅ PASS  纯删除变更显式判 N/A（未回退全文件）"
    PASS=$((PASS+1))
  else
    say "❌ FAIL  纯删除变更被判成假红或空转（应显式 N/A）"
    printf '%s\n' "$out_f" | tail -8
    FAIL=$((FAIL+1))
  fi
fi

echo
say "汇总: PASS=$PASS FAIL=$FAIL WARN=$WARN"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
