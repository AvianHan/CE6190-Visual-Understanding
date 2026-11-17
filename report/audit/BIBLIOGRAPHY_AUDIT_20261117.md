# Reference 与使用位置核验记录 — 2026-11-17

本次检查正文实际使用的 **36篇文献、46处 citation-key 对应关系**。检查包括论文身份、作者、题目、发表年份/场所，以及引用附近的具体论断是否由该来源支持。未发现未定义 citation key 或仅为凑数而未被引用的条目。具体来源、原文位置和访问限制在下列三份逐项记录中。

## 本轮实际修改

| 项目 | 修改 | 理由 |
|---|---|---|
| DINOv2 | 引用正式 TMLR 2024 版本，补全 Huy V. Vo | 优先采用正式发表记录；原2023 arXiv记录并非虚假 |
| SAM 2 | 引用正式 ICLR 2025 版本 | 原2024为preprint年份 |
| INSID3 | 补页码21638–21648 | 官方CVPR最终版元数据 |
| SegIC | 补LNCS15096、publisher，并采用geometric/visual/meta instructions表述 | 与原论文术语一致 |
| 数据集协议 | COCO/LVIS dataset引用与Matcher fold协议引用分开 | 避免把后来的ICS split归给原始dataset论文 |
| INSID3 aggregation | 增加“In the evaluated implementation” | 本项目执行代码的candidate coverage不能完全归为原论文公式 |
| 状态措辞 | “is being evaluated” | 第二批仍在进行，不暗示AGS已完成验证 |

这些更改已同步到印刷版与完整作者版bibliography；原数值结果没有改变。Bibliography历史核查保留于literature/source_audit.json，本轮记录为当前依据。

## 逐篇检查

- [ICS与few-shot方法：12篇、18处引用](bib_ics.md)
- [Visual foundation models与相关ICS：14篇、18处引用](bib_foundations.md)
- [数据集、adaptive inference与统计：10篇、10处引用](bib_stats_data.md)
- [当前机器可读引用位置](citation_inventory.json)

## 数学与归因边界

1. search–prompt interaction是匹配输入cohort上的factorial contrast，不是跨域普遍定律。
2. AGS cost–accuracy decomposition是共享prefix下两个分支的解析恒等式，尚非实测AGS加速。
3. Jaccard bound约束两张probe mask准确率之差；两者一致仍可能一起错。
4. IoU pooling covariance是加权恒等式，不能称为因果贡献。直接查询第一批ledger确认LVIS standard920类均有25个episode，因此55.19%→47.44%的当前数值在相同class/fold权重下成立（metric_weight_verification.json）。
5. HAM10000与ISIC2018成对引用遵从官方dataset归属说明；不能把HAM10000分类集独自写成Task1分割mask来源。

## 核查方式的限制

部分出版商页面直接访问受限，使用同一官方页面的检索结果、作者论文或学术机构托管的原论文补充核查；各条记录如实注明。CG-ICS方法逐项对照已保存的作者v2原文，出版元数据另行在线复核。未用其他论文的reference列表代替逐篇核验。
