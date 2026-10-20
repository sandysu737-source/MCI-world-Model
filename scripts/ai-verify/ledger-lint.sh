#!/usr/bin/env bash
# OODA 台账结构化校验（L-13）：列数一致 / ID 唯一 / 枚举合法 / 未转义竖线
# 用法: ledger-lint.sh [台账文件.md ...]     （可传 pre-commit 的暂存文件列表）
#   无参时只扫本仓两个载体: OODA回写台账.md、.ai-ooda/ledger.md
# 退出码: 0=通过 1=有异常
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

say(){ printf '\033[1m[ledger-lint]\033[0m %s\n' "$1"; }
FILES=("$@")
if [ "${#FILES[@]}" -eq 0 ]; then
  [ -f "OODA回写台账.md" ] && FILES+=("OODA回写台账.md")
  [ -f ".ai-ooda/ledger.md" ] && FILES+=(".ai-ooda/ledger.md")
fi
[ "${#FILES[@]}" -eq 0 ] && { say "无台账文件可校验"; exit 0; }

python3 - "${FILES[@]}" <<'PY'
import re
import sys

# 状态按 OODA §4 枚举 + 实际在用的 blocked；风险等级/状态允许尾随括号注解
# （如 "L2（...）"、"written（待二审）"），故用前缀匹配而非全等。
LEVELS = ("小环", "中环", "大环", "内环", "紧急")
STATUS_RE = re.compile(r"^(open|writing|written|verified|overdue|cancelled|superseded|blocked)\b")
RISK_RE = re.compile(r"^(L0|L1|L2|紧急停损)")
errors = 0

for path in sys.argv[1:]:
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        print(f"⚠️  {path}: 跳过（不存在/不可读）")
        continue

    lines = text.splitlines()
    header_idx = None
    expected = None
    for i, line in enumerate(lines):
        if line.startswith("|") and re.search(r"\|\s*ID\s*\|", line):
            expected = len(re.split(r"(?<!\\)\|", line)) - 2
            header_idx = i
            break
    if header_idx is None:
        print(f"⚠️  {path}: 未发现台账表头（跳过，非台账文件）")
        continue

    # 列索引按表头名定位（根台账 15 列 / 项目台账 14 列，列序不同）
    headers = [c.strip() for c in re.split(r"(?<!\\)\|", lines[header_idx])[1:-1]]
    def col_of(pred):
        for idx, name in enumerate(headers):
            if pred(name):
                return idx
        return None

    i_level = col_of(lambda n: "层级" in n)
    i_status = col_of(lambda n: n == "状态")
    i_risk = col_of(lambda n: "风险等级" in n)

    ids = {}
    bad = []
    for i, line in enumerate(lines[header_idx + 2:], start=header_idx + 3):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        n = len(cells)
        if n != expected:
            extra = "（疑似内容含未转义竖线，请写 \\|）" if n > expected else ""
            bad.append(f"  行 {i}: 列数 {n} ≠ 表头 {expected}{extra}")
            continue
        row_id = cells[0]
        if row_id in ids:
            bad.append(f"  行 {i}: ID 重复（首见行 {ids[row_id]}）: {row_id}")
        else:
            ids[row_id] = i
        if i_level is not None and i_level < n and not any(lv in cells[i_level] for lv in LEVELS):
            bad.append(f"  行 {i}: 层级非枚举值: {cells[i_level]!r}")
        if i_risk is not None and i_risk < n and cells[i_risk] and not RISK_RE.match(cells[i_risk]):
            bad.append(f"  行 {i}: 风险等级非枚举值: {cells[i_risk]!r}")
        if i_status is not None and i_status < n and not STATUS_RE.match(cells[i_status]):
            bad.append(f"  行 {i}: 状态非枚举值: {cells[i_status]!r}")

    if bad:
        print(f"❌ {path}: {len(bad)} 处异常（表头 {expected} 列，数据行 {len(ids)} 条）")
        for b in bad[:20]:
            print(b)
        if len(bad) > 20:
            print(f"  …另有 {len(bad) - 20} 处")
        errors += len(bad)
    else:
        print(f"✅ {path}: {len(ids)} 条台账条目结构合法")

sys.exit(1 if errors else 0)
PY
