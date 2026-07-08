# 每日开发标准操作 SOP

1. 拉最新代码，确认 `.ai-rules.md` 未变。
2. 填 `.ai-templates/structured-requirement.md`（无此文档禁止开工）。
3. 架构 Agent 出方案 + 人工审核（`.ai-templates/architecture-output-template.md`）。
4. 匹配 `.ai-skills/` 对应 Skill。
5. 编码 Agent 实现。
6. 测试 Agent 生成用例。
7. 质检 Agent 自动校验。
8. 人工终审（`.ai-checklist/ai-code-review-checklist.md`）。
9. 提交触发自动校验（pre-commit + `scripts/ai-verify/ai-guard.sh`）。
10. 合并、归档本次 AI 工作记录。
