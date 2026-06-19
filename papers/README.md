# Papers — MCI World Model

## 论文列表

| 文件 | 版本 | 状态 |
|------|------|------|
| `cognitive_enhanced_world_model.tex` | 英文版 | ✅ arXiv 就绪 |
| `cognitive_enhanced_world_model_cn.tex` | 中文版 | ✅ 完成 |
| `cognitive_enhanced_world_model.pdf` | 英文 PDF | ✅ 已编译 |
| `cognitive_enhanced_world_model_cn.pdf` | 中文 PDF | ✅ 已编译 |

## 论文信息

- **标题**: Cognitive-Enhanced World Model: A Paradigm Shift from Physical Simulation to Causal Understanding
- **中文标题**: 认知增强世界模型：从物理仿真到因果理解的范式跃迁
- **作者**: 苏强, 念淞元
- **理论框架**: Kant-Ashby-Lakatos 三维理论基础
- **对应版本**: MCI World Model v4.3.3

## arXiv 投稿检查清单

- [x] LaTeX 格式满足 arXiv 要求（标准 `article` 文档类）
- [x] 使用内联 `thebibliography`（避免 `.bib` 外部依赖）
- [x] 无 Mermaid/外部图表依赖（全部 TikZ 内联）
- [ ] 作者邮箱脱敏后方可投稿
- [ ] 建议添加 `\usepackage{arxiv}` 以获得标准 arXiv 样式
- [ ] 首次投稿需注册 arXiv 账号：https://arxiv.org/register
- [ ] 投稿入口：https://arxiv.org/submit

## 编译指令

```bash
# 英文版
pdflatex cognitive_enhanced_world_model.tex
pdflatex cognitive_enhanced_world_model.tex  # 两次以生成目录/引用

# 中文版（需 xelatex + ctex）
xelatex cognitive_enhanced_world_model_cn.tex
xelatex cognitive_enhanced_world_model_cn.tex
```
