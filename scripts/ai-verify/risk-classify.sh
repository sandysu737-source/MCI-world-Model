#!/usr/bin/env bash
# AI 代码风险自动定级器（保守兜底：默认 L2，仅显式特征可降级）
# 用法: risk-classify.sh [file1 file2 ...]   不传参则扫 git 暂存+工作区改动
# 输出: 每文件一行 "LEVEL<TAB>path<TAB>reason"，末行 "---" 后一行 MAX_LEVEL
# 退出码: 0=L0  1=L1  2=L2
# 逻辑/物理路径归一（2026-10-16 对抗审计实测）：macOS 的 `/tmp`→`/private/tmp`、
# `TMPDIR=/var/...`→`/private/var/...` 是软链；子进程若继承**逻辑** PWD，而 git 返回
# **物理** toplevel，则 `f` 与 `"$ROOT"/` 前缀对不上 ⇒ 依赖「f 在 ROOT 下」的判据静默失效
# （实测：用例19 的 7 项正控全灭，全部退化为默认档）。故两侧都取物理路径。
set -o pipefail
CALLER_CWD="$(cd "$PWD" 2>/dev/null && pwd -P)" || CALLER_CWD="$PWD"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ROOT="$(cd "$ROOT" 2>/dev/null && pwd -P)" || :
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
# 运维脚本生产写护栏（2026-10-17，OODA-20261015-RISKL0GAP-452 二次复发）：
# `scripts/` 曾整体按「纯工具/测试」放行 L0，但 `backend/scripts/*` 是一次性运维脚本族，
# 会直写生产库。L0 的语义是「无执行面」，故只要「DB 入口 + 写动作」或「--execute + psql」
# 成立，执行面即成立 ⇒ 不得 L0。判据只看路径 + 内容，不看文件名（改名不能绕过）。
OPS_SCRIPT_PATH_RE='(^|/)scripts/'
# DB 入口与写动词分开判定：只有「确实拿到数据库会话」且「确实存在写动作」才升级。
# 不能泛化匹配 `.add(`，否则 set/list 的 add 会把只读导出脚本误判为生产写脚本。
OPS_DB_ENTRY_RE='(SessionLocal[[:space:]]*\(|sessionmaker[[:space:]]*\(|create_engine[[:space:]]*\(|from[[:space:]]+database[[:space:]]+import[[:space:]]+get_db|get_db[[:space:]]*\(|psycopg2\.connect[[:space:]]*\(|sqlite3\.connect[[:space:]]*\()'
OPS_DB_WRITE_RE='(\.commit[[:space:]]*\(|\.flush[[:space:]]*\(|\b(db|session|conn|cursor|cur)\.(add|add_all|delete|merge|bulk_save_objects|bulk_insert_mappings|bulk_update_mappings)[[:space:]]*\()'
# 证据件放宽（2026-10-17，owner 裁决 A：放宽面收窄为「仅读数件」；台账 -594，
# 覆盖 -575 的 B 口径）。动机不变：证据件此前一律落「生产分支默认保守」= L2，
# 与仓内 `.gitignore:155-157`「完整日志不入库、只收策展摘要」的策展口径相反 ——
# 被策展入库的读数件反而把每次提交抬到双人 Review 门槛（实证 `-565`／`-567`）。
# 证据域＝**仓根锚定的权威证据根**（不是任意深度的同名目录：审计实测
# `backend/evidence/evil.py`、`frontend/src/evidence/evil2.sh` 曾被判 L1 =
# 把代码挪进同名目录即可降级）：
#   ① **仓根** `.ai-artifacts/`  ② `docs/<任意>/evidence/`
#   ③ `.ai-ooda/evidence/`       ④ 仓根 `evidence/`
# A 口径：**只有读数件放宽**。工具件（.sh/.bash/.py/.js/.mjs/.cjs/.mk）一律不放宽，
# 因为可执行面无法用内容正则证明安全（引号／env 前缀／解释器中介／变量拼路径均可绕过；
# 台账 OODA-20261016-575 的 6 条实测通道）。故本支不设任何工具件护栏正则。
EVIDENCE_ROOT_RE='(^|/)docs/([^/]+/)*evidence/|(^|/)\.ai-ooda/evidence/'
# 读数件：无执行面的日志／校验和／数据导出。**压缩件（.gz/.bz2/.xz/.zip/.zst）已排除**：
# 内容不可扫 ⇒ 放宽等于撤掉「疑似密钥」这道唯一兜底（审计实测：gzip 内含
# `PASSWORD=<19 字符>` 的 `.log.gz` 判 L1）。要恢复压缩件的放宽，前置条件是先有
# 解压内容扫描（`zcat` ＋ 同一密钥形态正则），见台账 -570 待办。
EVIDENCE_DATA_RE='\.(log|out|err|jsonl|csv|tsv|xml|sha256|sha512|sums)$|(^|/)(sha256sums|sha512sums|sha1sums)$'

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
  # F-26（2026-10-09）路径枚举必须 NUL 分隔：`git diff --name-only` 默认
  # core.quotePath=true，会把非 ASCII 路径写成带引号的八进制转义
  # （如 "docs/\346\262\273…md"），随后所有 `[ -f "$f" ]` 落空 → 该类文件被判
  # `SKIP not-a-file`；而 SKIP 在末段走 `*) continue` **不计入 MAX_LEVEL**，
  # 于是「只改中文名文件的提交」会被判 L0（实测：clinical-nutrition-finance-prod
  # 的 docs/ 中文名文档）。这既是漏判也是绕过面，故改 `-z` + `sort -zu`
  # （`read -d ''` 兼容 bash 3.2，不用 mapfile）。
  while IFS= read -r -d '' line; do FILES+=("$line"); done < <(
    { git diff --cached --name-only -z --diff-filter=ACMRD 2>/dev/null; \
      git diff --name-only -z --diff-filter=ACMRD 2>/dev/null; } | sort -zu)
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
  # 改为取原内容参与符号判定，路径规则照常生效。否则删除迁移/测试/CI/生产
  # 文件可完全绕过定级与门禁（历史上 --diff-filter=ACMR 还会把 D 滤掉）。
  # H-05-17b（2026-10-10 · OODA-20261010-012 路径 B）：**已提交删除**场景下 HEAD
  # 侧本就不存在该路径（CI 的 PR/push diff 即此形态），只回看 `HEAD:$REL` 仍落
  # SKIP ⇒ 删除类改动在 CI 恒判 L0（实测夹具 A1~A4 全 L0）。故追加回退：取
  # 「最后一次改动该路径的提交」的**父提交**版本（＝被删前的内容）参与判定。
  local DELETED=0 REL="" SRC_REF=""
  if [ ! -f "$f" ]; then
    REL="${f#"$ROOT"/}"
    case "$f" in "$CALLER_CWD"/*) REL="${f#"$CALLER_CWD"/}" ;; esac
    if git -C "$ROOT" cat-file -e "HEAD:$REL" 2>/dev/null; then
      DELETED=1; SRC_REF="HEAD"
      f="$ROOT/$REL"
    else
      local _dsha=""
      _dsha="$(git -C "$ROOT" log -1 --format=%H -- "$REL" 2>/dev/null || true)"
      if [ -n "$_dsha" ] && git -C "$ROOT" cat-file -e "${_dsha}^:$REL" 2>/dev/null; then
        DELETED=1; SRC_REF="${_dsha}^"
        f="$ROOT/$REL"
      else
        printf 'SKIP\t%s\tnot-a-file\n' "$f"; return
      fi
    fi
  fi

  # 内容判定统一入口：已删除文件读原版本（$SRC_REF），未删除读工作区版本
  # `-e` 不可省：以 `-` 开头的模式（如 `--no-verify`）会被
  # grep 当**选项**解析 ⇒ 报错返回 2 ⇒ 该护栏静默失效（2026-10-16 对抗审计实测）。
  _match() {
    if [ "$DELETED" = 1 ]; then
      git -C "$ROOT" show "$SRC_REF:$REL" 2>/dev/null | grep -qiE -e "$1"
    else
      grep -qiE -e "$1" "$f" 2>/dev/null
    fi
  }

  # 证据域＝权威证据根（审计 F3：任意 `/evidence/` 目录可自造 ⇒ 不能只按目录名判）。
  # A 口径把 `.ai-artifacts/` 锚到**仓根**：任意深度的同名目录不再算证据域。
  _is_evidence_domain() {
    local rel="$f"
    case "$rel" in "$ROOT"/*) rel="${rel#"$ROOT"/}";; esac
    case "$rel" in
      evidence/*|.ai-artifacts/*|.ai-ooda/evidence/*) return 0;;
    esac
    [[ "$rel" =~ $EVIDENCE_ROOT_RE ]]
  }

  # 放宽限定为**已在 index 中的件**（`git ls-files`）。确切语义（2026-10-16 对抗审计实测，
  # 台账 OODA-20261016-575）：**`git add` 过的新建件同样受益**，只有未 `git add` 的未跟踪件
  # 不受益；CI／PR 形态下新增件必然已入库 ⇒ 本限定**不构成**对「新增文件」的屏障。
  # 惰性构建一次全仓清单（bash 3.2 无关联数组 ⇒ 排序清单 ＋ grep -Fxq）。
  _is_tracked_evidence() {
    local rel="${f#"$ROOT"/}"
    [ "$rel" != "$f" ] || return 1
    if [ -z "${TRACKED_LIST:-}" ]; then
      TRACKED_LIST="${TMPDIR:-/tmp}/risk-classify-tracked.$$"
      git -C "$ROOT" ls-files -z 2>/dev/null | tr '\0' '\n' | LC_ALL=C sort -u > "$TRACKED_LIST" \
        || : > "$TRACKED_LIST"
      trap 'rm -f "$TRACKED_LIST"' EXIT
    fi
    grep -Fxq -- "$rel" "$TRACKED_LIST" 2>/dev/null
  }

  if [[ "$f" =~ $L2_CONFIG_RE ]]; then
    printf 'L2\t%s\tconfig/migration/CI-变更强制高级审\n' "$f"; return; fi

  local low; low="$(printf '%s' "$f" | tr 'A-Z' 'a-z')"
  if [[ "$low" =~ $L2_PATH_RE ]]; then
    printf 'L2\t%s\t路径命中高风险域: %s\n' "$f" "${BASH_REMATCH[1]}"; return; fi

  # 文档/数据类（.md/.txt/.rst/.docx/.xlsx/.csv）：描述性文本或无执行面数据，均无执行面；
  # 数据文件（.xlsx/.csv）与已列入的 .docx 同性质，此前漏列 ⇒ 生产分支下被「默认保守」
  # 兜到 L2，而同类 .md 为 L1（口径不一致，且会随 -z 修复放大成「中文名数据文件提交即需
  # 留痕」的噪声）。注意 .sql 不列入：含可执行语句，仍走 L2。但真实密钥检测
  # 保留拦截位（H-05-13 v4 C8 修正）：文档粘贴真实密钥/私钥仍须 L2 人工
  # 确认，不得静默放行；行文提及 token/auth 流程（非密钥形态）降 L1（避免
  # 8-27 EVAL-RUNBOOK/H-05-12 被硬词误定 L2 阻塞提交）。人工确认通道 =
  # L2 token 签发或 CI PR 双人复核（ai-guard.sh L2 分支）。
  case "$f" in
    *.md|*.txt|*.rst|*.docx|*.xlsx|*.csv|LICENSE*|CHANGELOG*)
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
  if [[ "$f" =~ $OPS_SCRIPT_PATH_RE ]]; then
    if { _match "$OPS_DB_ENTRY_RE" && _match "$OPS_DB_WRITE_RE"; } \
       || { _match '(^|[^A-Za-z])--execute([^A-Za-z]|$)' \
            && _match '(psql|SessionLocal\()'; }; then
      printf 'L2\t%s\t运维脚本含生产写入口(DB会话+写动词/--execute+psql)\n' "$f"; return
    fi
  fi
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
    *.md|*.txt|*.rst|*.json|*.yaml|*.yml|*.toml|*.docx|*.xlsx|*.csv|LICENSE*|CHANGELOG*)
      printf 'L1\t%s\t配置/文档类(生产分支L1)\n' "$f"; return;;
  esac

  # 证据域读数件白名单（2026-10-17 owner 裁决 A；台账 -594，覆盖 -575 的 B 口径）。
  # 位置经刻意选择：**置于全部硬词/软词/路径/文档规则之后**（同 dsh_poc 白名单先例），
  # 因此本支只能改变「原本落 `生产分支默认保守定级`」的那批件——命中密钥形态、凭据硬词、
  # 危险依赖者仍按上游规则判 L2，不会被本支回退。
  # 保障：① 上游硬词/软词规则优先；② **仅生产分支**生效；③ **仓根锚定的权威证据根**；
  # ④ **限定已在 index 的件**（`git ls-files`；未跟踪件不受益，CI／PR 形态下新增件已入库）；
  # ⑤ **仅读数件**（`.log/.out/.err/.jsonl/.csv/.tsv/.xml/.sha256/.sha512/.sums`／
  # `SHA256SUMS` 等）放宽；工具件与压缩件一律不放宽 —— 放宽面本身无执行面，不依赖任何
  # 「命令位/import 位」正则，故不存在 B 口径的 6 条间接构造通道。
  # 未放宽：.patch/.diff（内含可落地生产代码）、**压缩件**、无扩展名件、.b64/.sig/.bak-*、
  # 证据域外同名扩展名、未跟踪新增件、全部工具件。
  if [ "$PROD_BRANCH" = "1" ] && _is_evidence_domain && _is_tracked_evidence; then
    if _match "$L2_DOC_SECRET_RE"; then
      printf 'L2\t%s\t证据域件疑似包含真实密钥(放宽被拦；如为占位示例，人工确认后 token 放行)\n' "$f"; return
    fi
    if [[ "$low" =~ $EVIDENCE_DATA_RE ]]; then
      printf 'L1\t%s\t证据域读数件(无执行面/非生产行为)\n' "$f"; return
    fi
  fi
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
