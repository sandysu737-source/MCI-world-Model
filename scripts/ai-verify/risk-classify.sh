#!/usr/bin/env bash
# AI 代码风险自动定级器（保守兜底：默认 L2，仅显式特征可降级）
# 用法: risk-classify.sh [file1 file2 ...]   不传参则扫 git 暂存+工作区改动
# 输出: 每文件一行 "LEVEL<TAB>path<TAB>reason"，末行 "---" 后一行 MAX_LEVEL
# 退出码: 0=L0  1=L1  2=L2
set -o pipefail
CALLER_CWD="$PWD"                                   # 保存调用方 cwd（路径 resolve 用）
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# 路径硬域：词边界清晰，误报低
L2_PATH_RE='(payment|refund|settle|auth|login|webhook|alipay|wechatpay|wxpay|credential)'
# 硬词：任意出现即 L2（词边界，误报低）
L2_SYM_HARD_RE='\b(payment|refund|settle|auth|login|webhook|alipay|wechatpay|wxpay)\b'
# 文档类真实密钥模式（H-05-13 v4 C8）：只拦"疑似真实密钥"，不拦行文提及。
# 通用硬词集面向代码语义（支付/鉴权处理），不含 secret/key——文档粘贴真实
# 密钥会漏报；此处按密钥形态识别（sk-/ghp_/AKIA/JWT/私钥/长值赋值），
# 占位符（<token>/xxx/示例）长度不足不命中，避免 8-27 EVAL-RUNBOOK 误报。
L2_DOC_SECRET_RE='(BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|(SECRET|PASSWORD|API[_-]?KEY|TOKEN|PASSWD)[[:space:]]*[=:][[:space:]]*[A-Za-z0-9/+=_-]{16,})'
# 软词：需定义性出现(def/class/import/from 行)，排除 order_by 等通用用法
# dsh_poc 危险 import 模式（C1/H-05-16）：仅命中 import/from 行，避免普通变量名
# （calibration.py 的 order 排序变量）误伤；白名单=目录 + 无危险 import + 无硬词。
DSH_POC_DANGER_IMPORT_RE='^[[:space:]]*(import|from)[[:space:]]+[A-Za-z0-9_.]*(payment|order|auth|user|permission|crypto|requests|httpx|aiohttp|axios|redis)'
L2_SYM_SOFT_RE='(^|[[:space:]])(def|class|import|from)[[:space:]].{0,40}\b(order|permission|token|crypto|encrypt|decrypt)[a-zA-Z_]*\b'
L2_CONFIG_RE='(\.env$|docker-compose|\.github/workflows/|/migrations/.*\.py$|alembic/versions/.*\.py$|\.ai-coverage-threshold$|\.ai-governance/|/scripts/ai-verify/|_ai-eng-kit/)'
L0_PATH_RE='(^|/)(utils|helpers|scripts|tests|test|__tests__|spec)/|^.*(_test|\.test|\.spec)\.(py|ts|tsx|js)$'
# 测试路径专判（H-05-17）：删除测试文件=直接触碰「禁止删测试以通过门禁」红线，
# 不能与 utils/helpers/scripts 的 L0 同待遇，故单独一支用于删除类定级。
TEST_PATH_RE='(^|/)(tests?|__tests__|spec)/|(_test|\.test|\.spec)\.(py|ts|tsx|js)$'
DANGER_IMPORT_RE='\b(payment|order|auth|user|permission|crypto)\b|db\.session|db\.sessionmaker|\b(redis|httpx|requests|aiohttp|axios)\b|fetch\('

# F-25(治理 L) 回灌（2026-10-08，源自 mci-world-model 窗口）：CI 的 PR 事件 checkout
# 出的是 detached HEAD（分支名 "HEAD"），只看 HEAD 名称会把业务代码按非生产分支定级（L1），
# 使 PR 侧拦截弱于 push 侧。故允许显式传入目标分支（CI: RISK_BRANCH=${github.base_ref}），
# 并在此处留警告保证可追溯。
if [ -n "${RISK_BRANCH:-}" ]; then
  BRANCH="$RISK_BRANCH"
else
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  case "$BRANCH" in
    HEAD|unknown)
      echo "risk-classify: 警告: detached HEAD 且未设置 RISK_BRANCH，按非生产分支口径定级" >&2;;
  esac
fi
case "$BRANCH" in
  main|master|prod|production|release/*) PROD_BRANCH=1;;
  *) PROD_BRANCH=0;;
esac

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=()
  while IFS= read -r line; do [ -n "$line" ] && FILES+=("$line"); done < <(
    { git diff --cached --name-only --diff-filter=ACMRD 2>/dev/null; \
      git diff --name-only --diff-filter=ACMRD 2>/dev/null; } | sort -u)
fi

[ "${#FILES[@]}" -eq 0 ] && { echo "No files to classify"; exit 0; }

classify_one() {
  local f="$1"
  # 路径 resolve: 相对路径按调用方 cwd 解析成绝对路径，再判存在性
  case "$f" in
    /*) : ;;                                  # 已是绝对路径
    *)  if [ -f "$CALLER_CWD/$f" ]; then f="$CALLER_CWD/$f";  # 相对调用方 cwd
        elif [ -f "$f" ]; then :                           # 相对 git 根(cd 后)
        elif [ -f "$ROOT/$f" ]; then f="$ROOT/$f";         # 相对 git 根显式
        fi ;;
  esac

  # 删除类改动「删除即放行」盲区修复（H-05-17）：文件不存在时不再直接 SKIP，
  # 改为从 HEAD 取原内容参与符号判定，路径规则照常生效。否则删除迁移/测试/
  # CI/生产文件可完全绕过定级与门禁（历史上 --diff-filter=ACMR 还会把 D 滤掉）。
  local DELETED=0 REL=""
  if [ ! -f "$f" ]; then
    REL="${f#"$ROOT"/}"
    case "$f" in "$CALLER_CWD"/*) REL="${f#"$CALLER_CWD"/}" ;; esac
    if git -C "$ROOT" cat-file -e "HEAD:$REL" 2>/dev/null; then
      DELETED=1
      f="$ROOT/$REL"
    else
      printf 'SKIP\t%s\tnot-a-file\n' "$f"; return
    fi
  fi

  # 内容判定统一入口：已删除文件读 HEAD 版本，未删除读工作区版本
  _match() {
    if [ "$DELETED" = 1 ]; then
      git -C "$ROOT" show "HEAD:$REL" 2>/dev/null | grep -qiE "$1"
    else
      grep -qiE "$1" "$f" 2>/dev/null
    fi
  }

  if [[ "$f" =~ $L2_CONFIG_RE ]]; then
    printf 'L2\t%s\tconfig/migration/CI-变更强制高级审\n' "$f"; return; fi

  local low; low="$(printf '%s' "$f" | tr 'A-Z' 'a-z')"
  if [[ "$low" =~ $L2_PATH_RE ]]; then
    printf 'L2\t%s\t路径命中高风险域: %s\n' "$f" "${BASH_REMATCH[1]}"; return; fi

  # 文档类（.md/.txt/.rst/.docx）：描述性文本，无执行面；但真实密钥检测
  # 保留拦截位（H-05-13 v4 C8 修正）：文档粘贴真实密钥/私钥仍须 L2 人工
  # 确认，不得静默放行；行文提及 token/auth 流程（非密钥形态）降 L1（避免
  # 8-27 EVAL-RUNBOOK/H-05-12 被硬词误定 L2 阻塞提交）。人工确认通道 =
  # L2 token 签发或 CI PR 双人复核（ai-guard.sh L2 分支）。
  case "$f" in
    *.md|*.txt|*.rst|*.docx|LICENSE*|CHANGELOG*)
      if _match "$L2_DOC_SECRET_RE"; then
        printf 'L2\t%s\t文档疑似包含真实密钥(如为占位示例，人工确认后 token 放行)\n' "$f"
        return
      fi
      printf 'L1\t%s\t文档/说明类(描述性文本，L1)\n' "$f"; return;;
  esac

  # 硬词任意出现即 L2（即便在测试目录，含真实支付/鉴权符号也要审）
  if _match "$L2_SYM_HARD_RE"; then
    printf 'L2\t%s\t符号命中高风险域(硬词)\n' "$f"; return; fi

  # 删除类兜底定级（H-05-17）：路径/密钥规则未覆盖的删除也不得静默放行。
  if [ "$DELETED" = 1 ]; then
    if [[ "$f" =~ $TEST_PATH_RE ]]; then
      printf 'L2\t%s\t删除测试文件(红线「禁止删测试以通过门禁」，需 L2 授权)\n' "$f"; return
    fi
    printf 'L1\t%s\t删除生产文件(不可逆动作，需人工评审确认)\n' "$f"; return
  fi

  # L0 路径优先放行（测试/工具），仅当无危险 import
  if [[ "$f" =~ $L0_PATH_RE ]] && ! _match "$DANGER_IMPORT_RE"; then
    printf 'L0\t%s\t纯工具/测试路径且无危险依赖\n' "$f"; return; fi

  # 软词需定义性出现（def/class/import/from），排除 order_by 等通用用法
  if _match "$L2_SYM_SOFT_RE"; then
    printf 'L2\t%s\t符号命中高风险域(软词定义)\n' "$f"; return; fi

  # dsh_poc 纯评测工具白名单（C1/H-05-16 第一性原理）：纯统计/重放/盲评工具被
  # "生产分支保守"误定 L2（无支付/鉴权/用户数据/网络出口面）。双重保障：上方
  # 硬词/软词规则优先（runner.py 含 auth 语义→L2 不回退）；此处再拦危险 import
  # 与 db.session 用法（白名单仅放行真正无风险面的工具）。
  if [[ "$f" == *"/services/ai_engine/dsh_poc/"* ]] && \
     ! _match "$DSH_POC_DANGER_IMPORT_RE" && \
     ! _match 'db\.session|db\.sessionmaker'; then
    printf 'L1\t%s\tdsh_poc 纯评测工具白名单(无危险 import/凭据面)\n' "$f"; return; fi

  if _match '(@router|@app\.route|@command|@click|app\.(get|post|put|delete)\()'; then
    printf 'L1\t%s\t含路由/命令入口\n' "$f"; return; fi

  # 文档/说明类: 即便生产分支也只到 L1(不强制双人Review文档)
  case "$f" in
    *.md|*.txt|*.rst|*.json|*.yaml|*.yml|*.toml|*.docx|LICENSE*|CHANGELOG*)
      printf 'L1\t%s\t配置/文档类(生产分支L1)\n' "$f"; return;;
  esac
  if [ "$PROD_BRANCH" = "1" ]; then
    printf 'L2\t%s\t生产分支默认保守定级\n' "$f"; return; fi
  printf 'L1\t%s\t普通业务代码默认定级\n' "$f"
}

MAX_LEVEL=0
for f in "${FILES[@]}"; do
  out="$(classify_one "$f")"
  printf '%s\n' "$out"
  lvl="${out%%	*}"
  case "$lvl" in
    L0) n=0;; L1) n=1;; L2) n=2;; *) continue;;
  esac
  [ "$n" -gt "$MAX_LEVEL" ] && MAX_LEVEL=$n
done

printf -- '---\n'
printf 'MAX_LEVEL\tL%d\n' "$MAX_LEVEL"
exit "$MAX_LEVEL"
