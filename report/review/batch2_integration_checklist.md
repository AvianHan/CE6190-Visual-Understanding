# 第二批结果整合清单（仅供编辑，不进入 final report）

这份清单记录：若今晚取得可验收的新结果，报告哪些位置需要改。它不要求等待全部1040任务完成才开始检查，也不允许把task PASS或collector PASS直接当作性能证据。

## 先核对可用范围

按 **scope × configuration × query/reference** 核对已完成episode、缺失、error、重复与身份binding；数据范围必须与所写结论一致。新query与第一批隔离、SUIM排除尺寸异常样本的资格范围、ISIC public training-pool协议分别保留。新结果不能拼入第一批均值，形成不透明的cohort变化。

## 按章节更新

| 位置 | 新结果可支持时的改动 | 保留或删除的条件 |
|---|---|---|
| Abstract | 加入最有价值的一项validated AGS accuracy–latency结果或transfer发现，替换当前仅方法定义的末句 | 只用已核对的matching cohort与metric；不要以task完成百分比代替结果 |
| Introduction末段 | 从“假设与proposal”改为“hypothesis + 实证发现” | AGS结果不理想也可成为结论；不能预设其优越 |
| Contribution 3 | 若有完整matched controls，可写提出并评估AGS；若仅部分配置有结果，明确贡献是对应范围的evaluation | 不把缺失random/staged control情况下的平均加速写成gate signal有效 |
| §4.4 Component Ablations | 增加fixed-pool评分结果，区分reference fidelity与query presence | 只有候选池和最终visual prompt固定时才能归因ranking规则 |
| §4.6 Parameter Sensitivity | 可补充fresh-query效率验证，检验single-concept与one-initial-candidate发现 | 独立新query结果与原componentcohort分开列 |
| §5 AGS | 保留Decision Rule与Cost–Accuracy Analysis；将Evaluation Design收紧，增加Results段/表 | 主要对照full search、staged always-search、score gate、random routing；对应controls未完成则不做超范围结论 |
| AGS结果表 | 同表呈现paired IoU差、latency/speedup、search率、样本数；有条件再补requests/memory | 必须报告probe overhead；acceptance rate不等于speedup |
| Transfer结果 | COCO/LVISfresh core、SUIM、ISIC分别汇报；可新建Domain Transfer小节 | 不把SUIM/ISIC完成当成协议完全复现原benchmark |
| Discussion | 用真实结果回答agreement能否预测低价值search、是否跨域稳定 | 效果无优势时分析gate失效模式，而不保留积极预设 |
| Conclusion | 将最后的AGS hypothesis结论替换为经数据支持的明确答案 | 保持accuracy、cost、stability结论各自对应证据 |
| Appendix | 更新实际query/reference资格数、成功/失败/缺失、附加阈值结果与成本 | 服务器进程、jobIDs、collector过程不写进论文；保留在维护记录 |
| References | 若实际采用新增方法/数据源，补引用并核对使用；否则保持已核对36篇 | 不为了新结果数量增加无关文献 |

## 若只有部分结果能在截止前验收

先使用足以构成完整对照的问题单元，例如一个dataset上的AGS与所有对应routing controls。报告精确的已评估范围；未完成的跨域或阈值分析保留为extension design。新结果表有缺失时不能把缺失当作0，也不能从方便的成功子集推断整个计划。

## 必须重做的文稿一致性检查

Abstract、Introduction的contribution、§5、Discussion和Conclusion要同时更新；表格caption、样本数、GPU匹配规则、figurecallouts和Appendix协议同步。保留完整历史研究矩阵，但删除过时的“预计/进行中/等待收取”叙述。编译后逐页检查分页，不靠增加空泛句子填版面。
