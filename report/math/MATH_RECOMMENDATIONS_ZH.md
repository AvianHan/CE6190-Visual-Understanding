# 数学分析建议与核对结果

本次没有运行模型，所有新数值计算只读现有第一批 `episodes.sqlite`。重算脚本为 `analyze_mathematical_structure.py`；结果为 `mathematical_analysis.json`。已完整复现原四项交互效应及2000次image-cluster bootstrap区间，所有身份和原图GT总像素配对检查通过；四组IoU聚合恒等式误差均小于1e-12。没有使用任何第二批成绩。

## 主文建议一：组件收益不能简单相加

最有内容的数学化不是再写一遍方法公式，而是把已有二维消融写成一个interaction contrast。令s=是否完整搜索，v=是否visual boxes，J(s,v)为相同query在reference上平均后的IoU，则

Γ = J(1,1) − J(1,0) − J(0,1) + J(0,0)。

它回答：**加入搜索以后，visual boxes还能带来多少额外收益？** 如果Γ=0，两项收益在该度量上相加；Γ<0表示联合收益小于两项单独收益之和，不表示其中一项没有用，也不表示两套模块本质完全重复。

| 结果 | COCO | LVIS |
|---|---:|---:|
| 一次concept时visual增益 | +5.390 pp | +4.781 pp |
| 完整搜索时visual增益 | +2.541 pp | +1.776 pp |
| 搜索×visual交互Γ | −2.849523 pp | −3.004788 pp |
| image-cluster 95% CI | [−4.291943, −1.475129] | [−4.088406, −1.947408] |
| query数／image cluster数 | 400／398 | 1000／956 |

因此可写：在完整配对输入cohort上，观察到search与visual prompting的边际收益递减。不要写“数学证明搜索和视觉提示冗余”。数学上定义/估计交互；负值及区间是实际数据给出的经验结论。

附加实查：要求四种配置全部在同一记录GPU型号上运行，只剩COCO的300episode/100query和LVIS281episode/100query。该H200子集中交互为COCO+0.664456 pp（区间含0），LVIS−3.469693 pp（区间低于0）；精确区间在JSON。子集的fold和query构成也发生变化，不能把它解释成GPU造成总体效应。主文不必铺开这些审计细节，但应使用“observed/full matched input cohort”的措辞，避免泛化为设备无关普遍规律。

INSID3还有debias×aggregation的交互：COCO+1.172643 pp，CI[0.184024,2.236020]；LVIS+0.346227 pp，CI[−0.232827,0.928164]。适合一句补充，不建议再扩成另一套数学章节。

## 主文建议二：门控何时真正节省计算

这是可以严格推导的计算分解，但**没有第二批性能结论**。A=1表示接受初始concept；C_p为初始concept、visual和agreement probes共同前缀；C_e为直接用初始concept做最终分割的后续成本；C_s为完整搜索再做最终分割的后续成本。最终分割的成本可能随concept变化，因此不能只将saved cost粗略写成search函数的耗时。

C_gate = C_p + A C_e + (1−A) C_s。

对照staged always-search具有相同前缀，C_staged=C_p+C_s。两者相比，平均节省为E[A(C_s−C_e)]。令H=C_staged−C_original_full表示实际staging overhead，则门控相对原始full更快的条件恰好是

**E[A(C_s−C_e)] > E[H]。**

用户容易理解的说法：**跳过搜索省下的钱，必须大于做这次检查花的钱。** 仅仅“少搜索了30%”不能推出“总运行变快了30%”；被跳过的输入可能本来就便宜，probes也可能很贵。该式不假定接受事件与搜索成本独立。

在相同共享前缀的分支模型下，准确率也可写成E[J_gate−J_staged]=E[A(J_e−J_s)]。J_e、J_s是最终joint-prompt输出的IoU，不是text/visual probe之间的IoU。它告诉我们准确率变化来自被接受、不继续搜索的那些输入。因此staged always-search测开销，random routing比较路由信号价值，第二批的实际输入条件分析才验证agreement是否有用。不能从第一批均值或这个恒等式宣称门控已有提升。

## 可放附录：两个简短命题

**Agreement bound。** Jaccard距离满足三角不等式，故|IoU(M_t,GT)−IoU(M_v,GT)|≤1−IoU(M_t,M_v)。如果两路agreement≥0.7，它只保证两路probe的准确率差≤0.3，不能保证任一准确率高。两者都预测同一块错误区域就是反例。这个命题解释agreement的含义和不足；不需要在主文重复大量防御性说明。它也不直接约束最终joint-prompt输出。证明和引用key `kosub2019` 已写进 `appendix_insertions.tex`，基础文献agent在核实原始期刊记录。

**IoU aggregation identity。** 类内pooled IoU是按union像素数加权的episode IoU。因此J_pool−mean(J)=Cov(U,J)/mean(U)。又因U=GTarea+FP，可以把差值精确分成target-area项和FP项。CG-ICS LVIS的55.19% mean episode与47.44% pooled相差−7.75 pp，其中target-area项+3.82 pp、FP项−11.57 pp。这解释“为什么同一套预测两个均值差那么大”：面积很大的误报还会放大坏预测在pooled度量中的权重。这里是精确代数分解，不是对错误产生原因的因果证明。

## 新增诊断，建议不必塞进主文

把同query的reference先平均，比较single-concept+visual与full search：COCO400个query中194个搜索更好、196个single更好、10个相同；LVIS1000个中535/403/62。相差超过5 pp的搜索获益/受损query分别是COCO46/41、LVIS209/131。它支持“平均增益掩盖输入间差异”的描述。

用GT事后为每个query选择两分支中较好者，得到73.33%/58.57%的hindsight上包络。这个选择需要知道答案，**不是可部署方法，也不是agreement gate的预测成绩**。因此建议只留在分析材料，不把它作为主表新方法或主文承诺。实测设备构成也混合H200/L40S，不能把该诊断当成完全隔离搜索随机性的因果效应。

## 放置与写作控制

- Results / Component Ablations：一个interaction公式、一段数值解释即可。
- Agreement-Gated Search：共享前缀定义、两条成本/精度恒等式、break-even一句，连接既定控制实验。
- Appendix：agreement bound+三行证明，pooled IoU恒等式+LVIS数值解释。
- 不需要标题“Theoretical proof of our method”；“Search--Prompt Interaction”“Cost and Accuracy”“Agreement Bound”“IoU Aggregation”足够专业简洁。
- caption不塞审计清单。范围、统计单位、旧四个error的null处理统一放Experimental Setup一次。

