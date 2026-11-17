from pathlib import Path
import json, re, hashlib
D=Path(__file__).resolve().parent
# Each entry maps one source narrative sentence / equation lead-in in reading order.
M={
'introduction':[
['定义 ICS：把 annotated reference 中指定的目标迁移到 query。｜任务定义','交代比较对象 INSID3 / CG-ICS，以及 accuracy、reference sensitivity、inference cost 三条评价轴。｜研究问题','给出主结果：CG-ICS 的 COCO/LVIS mean-fold class-pooled IoU 为 72.17/47.44，INSID3 为 57.65/40.77。｜第一批主表','说明 accuracy 的代价：匹配 runner GPU model 后 mean inference latency 为 7.37–10.63 倍，且另占 MLLM GPU。｜第一批成本','归纳两系统的有效成分：INSID3 的 debiasing / aggregation，CG-ICS 的 semantic concept + visual prompts。｜第一批 ablation','指出额外 concept search 的平均收益不一致：COCO 0.22 pp，LVIS 2.30 pp。｜第一批 ablation','概括 search–prompt interaction：加入 visual prompting 后 search 的边际收益降低。｜第一批配对 interaction','补充 reference variation 与 small-target failure，避免只看均值。｜第一批 robustness / failure','从证据导向 AGS proposal，并引出 cost–accuracy decomposition 与 controlled evaluation。｜方法动机＋数学分析','界定当前贡献状态：比较实验完成，extension 的 empirical results 尚待验收。｜状态边界'],
['给出 ICS 的三项输入和目标。｜任务定义','解释这种接口的价值：用户给 visual example，无需逐张 query 标注或重新训练 task-specific model。｜应用动机','列出 transfer 的困难：appearance change、partial reference、distractors。｜问题来源','把困难收束为一个工程目标：在可接受 inference cost 下消歧。｜引出 accuracy–cost trade-off'],
['引入 foundation model 提供的两条解决路线。｜方法分叉','INSID3 通过 frozen DINOv3 的 debiasing、clustering、similarity aggregation 构造 mask。｜引用主方法','CG-ICS 通过 reference-derived concepts、search 和 SAM 3 visual prompts 构造 mask。｜引用主方法','总结比较轴：reference 如何转化为可用于 query 的 evidence。｜概念对照'],
['指出 headline accuracy comparison 本身不足。｜研究缺口','列出待解释因素：reference、processing stage、不同 input 的 search 收益。｜拆分问题','因此需要把 whole-system evaluation 与 component intervention、measured cost 连接起来。｜研究设计','进一步导出效率目标：减少低 marginal value 的 search，保留初始预测不足时的 search。｜proposal 的一般原则'],
['限定本研究的数据与协议：COCO/LVIS，common inputs，original-resolution evaluation。｜实验范围','先给总体发现：CG-ICS 更准，INSID3 更快。｜第一批主结论','再给机制发现：COCO 的 single concept + visual prompts 已获得多数收益，LVIS 更受益于 extra search。｜第一批 component 结论','指出 visual prompting 在两个数据集都降低 search 的 observed marginal gain。｜第一批 interaction','从此提出可观察 signal：text-only / visual-only predictions 的 agreement，用于 input-dependent search。｜有动机的假说；不是已验证 gate 性能'],
['把前三段研究问题压缩为三项 contributions 的引导语；用户已要求并入前段末尾。｜结构连接'],
['Contribution 1：common inputs 上比较 accuracy、reference sensitivity、computational cost。｜已完成经验贡献'],
['Contribution 2：通过 controlled interventions 区分 feature processing、semantic search、visual prompting、backbone choice。｜已完成经验贡献'],
['Contribution 3：提出 AGS，给出 explicit cost–accuracy formulation 和 predefined controls。｜proposal＋分析贡献','声明 empirical conclusions 依据第一批；fresh query / transfer evaluation 属于 extension。当前“is evaluated”易读成已完成，建议改为 ongoing evaluation 的明确表述。｜状态边界；建议微调']
],
'related_work':[
['从 visual understanding 的两项需求开场：semantic recognition 和 spatial localization。｜建立上位概念','ViT 的 patch tokens 与 DINO 的 self-supervised object structure 提供 feature 路线背景。｜文献：ViT / DINO','DINOv2 / DINOv3 将这一 representation 路线扩展到更大规模。｜文献：DINOv2 / DINOv3','说明这些 dense features 如何服务跨图 correspondence，从而服务 reference→query transfer。｜与本文任务连接'],
['转向 promptable foundation models，把 semantic/spatial information 变成 segmentation interface。｜第二条技术路线','按 SAM→SAM 2→SAM 3 串起 spatial prompts、video、concept prompts。｜文献定位','CLIP 的 image–text alignment 与 Qwen3-VL 的 multimodal reasoning 补足 language 一侧。｜文献定位','把文献能力重新收束为本文的 visual correspondence vs semantic grounding 两条路线。｜回到研究轴'],
['从 one-shot、prototypes、dense correspondence 追溯 reference-conditioned segmentation。｜历史背景','用 visual prompting、Painter、SegGPT 说明 example 本身可以表达 task。｜In-context learning 脉络','归纳 annotated example 是 desired output 的 flexible specification。｜本节共同概念'],
['指出近期方法的区别在于 example 如何变成 query mask。｜分类原则','SegIC、SINE、VRP-SAM 分别展示 correspondence guidance、interaction、reference prompt encoding。｜学习式方法位置','PerSAM 和 Matcher 展示 frozen representations 与 segmentation priors 的组合。｜冻结模型方法位置','将 INSID3 的直接 feature processing 与 CG-ICS 的 concepts/search/visual exemplars 对照。｜定位本研究两主方法','明确本文选取这两个 full inference systems，并继续拆解其 stages。｜收束 Related Work'],
['提醒 reference 本身会引入 variability。｜引入 robustness 文献','Good Examples、UNICL-SAM、CG-ICS 各自讨论 example selection、uncertainty、reference robustness。｜文献与研究缺口','本文固定 query、改变 demonstration，量化 reference sensitivity。｜对应 E2','RePRI 的 transductive optimization 与 ABCDFSS 的 test-time adaptation 提供 cross-domain 参照。｜transfer 背景','将这些 adaptation mechanisms 与当前 frozen inference 区分，解释为何 domain transfer 需要单独评价。｜范围连接']
],
'methods':[
['给出形式化接口：输入 reference image、reference mask、query image，输出 query mask。｜公式 M1'],
['限定输出为所示 semantic target 的 binary segmentation。｜任务范围','query GT 仅用于 evaluation；inference 不接收 dataset class label。｜防止 label leakage 的核心约束','两系统均使用 pretrained models，不做 task-specific parameter updates。｜training-free 范围','说明比较顺序：full pipelines→within-system components→common interface 下的 trunks。｜实验设计预告'],
['INSID3 的信息来源是 frozen DINOv3 的 semantic correspondence 与 spatial grouping。｜主方法解释','Figure 1 将 original / debiased 两条 feature paths 分开。｜图文索引','original query features 用于 coherent clustering 和 within-query similarity。｜分支职责','debiased reference/query features 用于 cross-image matching，弱化 absolute position。｜分支职责'],
['定义 dense-feature matrix F 与 positional-subspace orthonormal basis B。｜公式变量','通过投影到 positional subspace 的 orthogonal complement 再 normalize 实现 debiasing。｜公式 M2'],
['在 reference foreground 内平均 debiased features，构造 foreground prototype。｜算法步骤','forward similarity / backward correspondence 找 candidates，prototype 选择 seed cluster。｜算法步骤','aggregation 组合 cross-image alignment、original-space seed similarity、candidate coverage。｜算法步骤','超过 merge threshold 的 clusters 组成最终 mask。｜算法输出'],
['INSID3 ablation：debiasing on/off × 三种 aggregation；seed-only 仍含 clustering。｜控制变量','simple-similarity control 保留 debiasing，但不做 clustered region processing。｜控制组定义','列出主设置：1024 input、rank 500、clustering 0.6、merge 0.2、bilinear restoration。｜可复现设置','implementation-specific details 放入 Appendix。｜正文与附录分工'],
['CG-ICS 包含 concept search 与 parallel visual transfer。｜结构定义','MLLM 从 reference 与 highlighted target 生成 candidate concepts。｜concept generation','按 reference fidelity 和 query presence 给 candidate concept 打分。｜公式 M3'],
['query score s_q 为 SAM 3 presence score。｜变量解释','expansion、pruning、restarts、reranking 寻找同时匹配 reference 且存在于 query 的 concept。｜search 机制','默认 score exponents α=β=1。｜可复现设置'],
['visual branch 将 reference 和 query 拼接。｜算法步骤','reference-mask components 产生 spatial prompts，query-side outputs 转为 box prompts。｜算法步骤','SAM 3 结合这些 boxes 与 selected concept 输出 query segmentation。｜融合位置；Figure 2','比较 single concept、scored candidates、full search，各配 with/without visual boxes。｜factorial design','visual-only 与 direct concatenation 是额外 controls。｜排除替代解释'],
['实现使用 Qwen3-VL-4B-Instruct、working size 512、5 initial candidates、3 search loops。｜具体运行配置','SAM 3 内部另做 size-1008 preprocessing。｜解释尺度接口','MLLM 与 segmenter 分占 GPUs；system cost 需要计入额外 MLLM allocation。｜成本口径']
],
'experiments':[
['数据使用 author-prepared class memberships：COCO 4 folds×20 classes，LVIS 10 folds×92 classes。｜dataset / split','standard comparison 每系统 4,000 COCO、23,000 LVIS episodes。｜样本量','reference cohort 固定 query target，改变 demonstration。｜E2 设计','component cohort 为 400 COCO、1,000 LVIS query targets，各最多3 references，对应1,200/2,757 episodes。｜component 样本量','同一 comparison 内各方法使用 matched inputs。｜配对设计'],
['每 episode 在 original query resolution 计算 foreground IoU。｜公式 E1'],
['headline metric：class 内 pool TP/FP/FN，fold 内平均 classes，再平均 folds。｜公式 E2'],
['paired metric：先对同一 query target 的 planned episodes 平均，再对 method difference 等权平均。｜公式 E3'],
['bootstrap 按完整 query-image clusters 重采样，concepts 和 references 一起保留。｜统计独立单位','主/component contrasts 用2,000次，parameter/area用1,000次，给 exploratory 95% CIs。｜不确定性设置','reference SD 先合并 identical inputs 的 repeated executions，再在不同 references 间计算。｜robustness 指标','failed predictions 保留 missing accuracy，paired contrasts 使用 common successful inputs。｜失败处理口径'],
['第一批有36 configurations、615,976 records，其中4 preprocessing failures。｜覆盖范围','已核对 expected episode identities 和 shard assignments。｜审计证据','各 dataset 分开报告，在实际执行的 common protocol 上比较。｜解释范围','完整 matrix 与 protocol 放 Appendix。｜正文导航'],
['CG-ICS 在两个 dataset 的 headline accuracy 更高。｜Table 1','paired equal-query 优势为 COCO14.01 pp、LVIS16.22 pp。｜第一批配对效应','区分指标：class pooling 对 episode 按 union size 加权，equal-query 对 query target 等权。｜避免把两列当同一均值','LVIS CG-ICS episode mean55.19%，mean-fold class pooling47.44%。｜指标差异实例','Appendix 用 exact weighting identity 解释差异。｜连接数学分析'],
['accuracy advantage 带来 substantial cost。｜引出 trade-off','same-input/same-runner-GPU 比较中，CG latency 为 INSID3 的7.37–10.63倍。｜Table 2','这些 times 仅含 inference，不含 startup、evaluation、writing。｜成本定义','CG 额外 MLLM GPU 意味着资源差异不止 latency ratio。｜system-level cost'],
['CG 每 standard episode 约3.43–3.50 MLLM requests、29.89–32.76 SAM calls，解释主要开销。｜operation counts','MLLM startup median58–71s，小 batch 时明显。｜启动成本','startup 与 owned-process memory 另放 Appendix。｜附录导航'],
['accuracy ranking 不自动决定 variability ranking。｜新的研究问题','CG 在 COCO 降低 reference SD，在 LVIS 增加。｜Table 3','但 CG 在两个数据集的 mean worst-reference IoU 均更高。｜Table 3','因此更高 worst-case-level accuracy 可以与更大 reference sensitivity 同时出现。｜解释现象'],
['H200-only subsets 中 reference SD difference 的区间仍一负一正。｜硬件分层稳健性检查','reference robustness 是 dataset-dependent，variability 必须结合 accuracy 一起看。｜本节结论'],
['three-seed 固定 component inputs，CG mean per-input SD为2.23/3.65 pp。｜run sensitivity','263/1,200与694/2,757 inputs 的 prediction hashes 跨3 seeds完全一致。｜离散输出稳定性','部分运行混合 GPU model，因此这是 run sensitivity，不是 isolated seed effect。｜归因边界'],
['INSID3 的收益不只来自 feature trunk。｜component 问题','移除 debiasing 使 COCO/LVIS equal-query IoU下降2.38/2.26 pp。｜paired ablation','self-plus-cross 改为seed-only下降9.24/4.06 pp。｜paired ablation','full processing 相比debiased similarity提高19.23/11.94 pp。｜paired ablation','representation adjustment 与 clustered spatial structure 均有贡献。｜机制归纳'],
['CG 的 language 与 visual prompts 提供互补信息。｜component 问题','single concept 加boxes提高5.39/4.78 pp；visual-only加concept提高13.38/16.54 pp。｜paired interventions','extra search 的收益随 dataset 和已有visual route而变。｜从互补转向边际收益'],
['带boxes时full search比single concept在COCO只提高0.22 pp且CI跨0，LVIS提高2.30 pp。｜Table 4','不带boxes时search收益为3.07/5.30 pp。｜factorial comparison','search贡献依赖dataset与visual route。｜引出interaction公式'],
['single-concept route成本显著更低。｜引出conditional search','matched subsets中full search耗时为single的4.17倍或4.37–4.56倍。｜相同inputs / GPU成本','同一timing subsets的accuracy changes为COCO−0.54、LVIS/L40S+1.53、LVIS/H200+3.67 pp。｜同口径trade-off','因此值得在fresh inputs测试conditional search。｜动机，不是gate性能证明'],
['现有candidate-stage变化同时改变generation/selection，full search还包含restarts。｜说明当前干预并非纯score ablation','第二批fixed-pool实验将固定candidate evidence，单独测试score terms。｜明确剩余问题'],
['INSID3 processing提高三种trunks的表现。｜Table 5','两种processing下DINOv3均最好；equal-query processing gain在COCO15.49–19.39、LVIS11.17–11.95 pp。｜representation/processing matrix','representation与mask construction各有贡献。｜分离解释','SAM 3这里只是common feature interface中的trunk；不是full promptable decoder比较，也未逐backbone优化。｜关键比较边界'],
['single-factor sweeps显示更多compute不保证更高accuracy。｜参数研究结论','clustering threshold0.6→0.7对COCO−1.53、LVIS+1.85 pp。｜dataset-dependent sensitivity','input1024→768对accuracy−0.88/−2.25 pp，latency缩为默认0.39–0.41。｜效率操作点'],
['CG initial candidates5→1对accuracy−0.16/−0.82 pp，matched latency缩为0.63–0.66。｜效率操作点','search loops3→1对COCO观察到很小影响，LVIS−0.75 pp。｜参数收益','accuracy来自full component cohort，timing来自same-GPU subsets。｜配对口径','完整CIs见Appendix。｜正文压缩'],
['CG在small query targets上的优势最大。｜尺度规律','foreground<1%时paired gain为21.88/24.03 pp。｜area-bin结果','更大两档仍为正，但幅度更小。｜趋势完整性','size指query semantic foreground proportion。｜变量定义'],
['Figure4展示不同方法互补的成功与失败。｜定性分析','CG找到INS漏掉的book和monitor。｜具体案例','INS找到food，而CG在salmon medallion概念下选到distractor。｜反方向案例','两者都能分stop sign，也都在另一个food target失败。｜共同成功/失败','bowl案例显示过度覆盖background。｜FP失败类型'],
['floor unit、buttered bread、vehicle三个概念分别造成目标范围错误。｜semantic specificity案例','因此除了mask quality，也应检验concept quality。｜引出score/generation研究'],
['四次execution errors发生在preprocessing。｜区分模型错误与执行失败','3个不同reference masks仅1或5 pixels，legacy nearest resize到512²后foreground消失。｜直接失败证据','所以CG在inference前停止。｜因果链','建议foreground-preserving resize，并在原失败与independent small-reference examples上测试。｜具体future work','20case中的36个available masks已重算并复现savedcounts/IoU，正文展示6例。｜定性证据审计']
],
'interaction_analysis':[
['定义J_u(s,v)：query-level reference-averaged IoU；s=single/full search，v=without/with boxes。｜2×2变量','定义difference-in-differences Γ。｜公式 E4'],
['Γ衡量search如何改变visual prompting的marginal benefit；等价地也衡量visual prompting如何改变search收益。｜interaction含义','Γ为COCO−2.85 pp、LVIS−3.00 pp，95%CIs均不跨0。｜第一批配对统计','在full matched cohort，两种addition呈diminishing joint gains；boxes两种search level都有效，但single concept时更有效。｜可支持的结论']
],
'extension':[
['component结果提示按input分配search。｜proposal动机','visual prompting降低observed marginal search benefit，但剩余收益随dataset/query变化。｜证据桥梁','AGS用初始text-only与visual-only predictions的一致程度决定继续search。｜方法概念','将input-dependent computation allocation思想用到CG concept-search stage。｜与adaptive inference文献连接'],
['定义initial concept c0、text-only mask Mt和visual-only mask Mv。｜decision变量','任一probe为空则不接受；否则计算两mask IoU，超过τ则A=1。｜公式 P1'],
['A=1保留c0并产生joint text-plus-visual output。｜early branch','A=0先concept search，再输出final joint prediction。｜search branch；公式 P2'],
['primaryτ=0.7，0.5/0.9用于sensitivity。｜预设超参','decision不使用query annotation。｜可部署性','agreement衡量probe consistency；与target accuracy的关系在Appendix形式化。｜信号边界'],
['第二批包含四个相连部分。｜evaluation roadmap','fixed-pool ablation固定candidate evidence和visual prompts，比较s_rs_q、s_r、s_q。｜isolating scoring','fresh-query比较full、single-concept、one-initial-candidate。｜独立泛化复查','SUIM和ISIC2018 Task1测试domain transfer。｜新domain','AGS对照original full、staged always-search、initial-score gate、random routing。｜排除只要少search就会快的解释'],
['core请求400fresh COCO、1,000fresh LVIS queries，每个最多3references。｜预设样本量','SUIM/ISIC各请求1,000queries，pool定义在Appendix。｜预设transfer规模','initial-score threshold0.8，random search概率0.25/0.5/0.75。｜controls设置','这些controls检验agreement是否识别适合停止的inputs，超越单纯减少search数量。｜核心可证伪问题'],
['评价paired accuracy、latency distribution、reference sensitivity、requests、startup、memory。｜完整metric设计','第二批仍在运行；collect并verify后才能加入performance results。｜明确结果状态']
],
'gate_analysis':[
['令A=1表示接受initial concept。｜变量','shared prefix cost为Cp；其后early/search branches余下cost分别Ce/Cs，均含final segmentation。｜cost边界','定义gated与staged always-search的cost。｜公式 P3'],
['令D=Cs−Ce，Je/Js为相应final-mask IoUs，可得expected cost和accuracy差。｜公式 P4'],
['两个identity分别量化accepted inputs节省的compute，以及同一accepted subset的accuracy变化。｜identity解释','若H为staging相对original full的extra cost，则真正比original更快当且仅当E[AD]>E[H]。｜速度成立条件','因此acceptance rate本身不能证明speedup，branch savings必须覆盖probe overhead。｜排除常见误推','staged always-search与random routing分别识别overhead和agreement signal的价值。｜由分析导出control必要性']
],
'discussion':[
['两系统提供不同operating points。｜整体收束','CG平均更准且small-target更强，INS更便宜。｜主结果综合','reference维度补充：CG两dataset worst-reference更高，但LVIS variation更大。｜robustness综合','系统选择需同时权衡accuracy、latency、resources和reference consistency。｜实际含义'],
['encoder matrix与within-system ablation说明full-system ranking不能简化为backbone ranking。｜归因结论','DINOv3在feature matrix领先，但CG通过language/prompting/decoding组合取得更高system accuracy。｜解释表面矛盾','INS processing对每个trunk都有帮助。｜补充证据','所以representation与mask construction procedure应共同评价。｜原则'],
['selective search动机来自average gain与input-specific gain的区别。｜新方法逻辑核心','full search在194/400COCO与535/1,000LVIS queries改善，在196/403个下降。｜query-level异质性','positive dataset mean不意味着每query受益。｜由计数得出的结论','search–prompt interaction进一步表明component价值取决于其他component是否已存在。｜连接factorial证据'],
['前述interactions描述执行过的full paired cohort。｜适用样本','要求四config同GPU后，仅剩每dataset100H200queries。｜稳健性subset','该subset interaction为COCO+0.66CI跨0，LVIS−3.47CI负。｜分层结果','LVIS方向保留，COCO证据不确定。｜谨慎但必要的结论','subset同时改变fold/input composition，不能据此归因hardware effect。｜避免伪因果'],
['下一步检验observable signal能否在fresh inputs有效分配search。｜从描述到预测','gate将要求写明：saved branch compute覆盖probe cost，accepted inputs决定accuracy变化。｜数学分析回扣'],
['本研究结合system、paired component与pixel-level failure分析。｜scope概括','query-level aggregation与image-cluster uncertainty处理reference dependence。｜统计原则','transfer泛化与foreground-preserving preprocessing是两条具体扩展。｜未来工作','当前关键未决问题是agreement能否跨domain可靠预测low-value search，第二批专门检验。｜明确knowledge gap'],
['两条路线形成不同accuracy–cost trade-offs。｜Conclusion总句','CG在初始benchmark更准，INS更快。｜结论1','components显示feature processing、concept、visual prompts贡献，并揭示dataset-dependent search收益。｜结论2','reference sensitivity与small-target failures补充mean accuracy之外的区别。｜结论3','这些证据motivate AGS；formulation写清必要compute savings与accuracy trade-off。｜proposal贡献','ongoing controlled evaluation将决定它是否成为实际有用的adaptive system。｜不提前宣称成功']
],
'appendix':[
['Table6记录第一批研究范围。｜matrix导航','task count解释seed/sweep复用controls，避免重复计数。｜覆盖口径'],
['第一批1,480successful formal tasks＋4failed reference tasks。｜执行覆盖','615,976expected records含4errors均已匹配，并核对532outputbindings。｜审计完整性','standard cohorts含3,975/13,750 distinct targets，来自3,820/9,051 distinct image hashes。｜区分episode/query/image','COCO/LVIS共享462standard image hashes，所以dataset分别报告。｜避免误当独立复现'],
['common manifests＋original GT保留class memberships，但sampling/aggregation可能不同于author protocol。｜复现范围','sampling前排除empty LVIS masks，仍保留all classes/folds。｜数据有效性','未控制pretraining overlap，也未做multiple-comparison adjustment。｜剩余局限'],
['执行版INS positional basis用ImageNet-normalized all-zero image；论文描述Gaussian noise。｜实际实现差异','original features用于clustering/self-similarity，debiased用于cross-image matching。｜分支实现','mainroute为bilinear restoration，无CRF refinement。｜后处理','CG working size512，之后SAM internal preprocessing，nearest restoration回原尺寸。｜预处理/恢复','direct concatenation control采用实现中的query-side instance union。｜control定义'],
['fresh core queries排除以往formal query image hashes以及跨初始dataset的canonical COCO IDs。｜freshness规则','transfer使用author-prepared merged SUIM pool和reorganized ISIC2018 Task1 training images。｜数据来源','SUIM sampling前检查full source pool，剔除image/mask尺寸不一致pairs而不改原raster。｜已修复data validation','要求7foreground classes和1,000unique queries；eligible counts随结果报告。｜计划覆盖','ISIC references用supplied diagnostic strata；不是patient-disjoint clinical test或official challenge-test。｜transfer解释边界'],
['fixed-pool每input只生成一次candidates，各rankingrule用同一evidence。｜公平scoring比较','pool最大combined score<1e−6时，各rule共享visual fallback。｜特殊情况一致性','random routing使用预设probabilities，不依据gate已观测search rate选概率。｜避免事后选control','staged always-search量化initial probing的extra cost。｜必要control'],
['owned memory在同一采样时刻汇总runner与其MLLM processes。｜memory定义','MLLM0.85utilization含reservation/cache，故数值不是minimum model memory，也不是runner-only CUDA peak。｜防止误读memory']
],
'mathematical_appendix':[
['把非空binary masks看作pixel sets，定义J为IoU，d=1−J为Jaccard distance。｜数学对象','用reverse triangle inequality约束两probe对GT的IoU差。｜公式 A1'],
['证明思路：双向用triangle inequality得到distance difference上界。｜推导','high agreement只限制两probe accuracies之间的差，不保证它们的absolute accuracy。｜准确解释','反例：两probe完全一样却都选在GT外，则mutual IoU=1、target IoUs=0。｜反驳错误充分条件','该bound仅约束probe masks，不直接约束后续joint-prompt output。｜定理适用范围'],
['在每个class内定义union size Ue及episode IoU Je。｜加权变量','使用episode population covariance展开pooled IoU与episode mean的差。｜公式 A2'],
['G表示GT target area，因此U=G+FP。｜第二层分解变量','由TP=UJ及mean product=covariance+product of means推出identity。｜推导','large false-positive region既降低IoU又增大union weight，所以pooled score可能受更大影响。｜数学解释与经验现象连接'],
['LVIS CG episode mean55.19%，class-pooled47.44%。｜待解释观测','按相同class/fold weights平均identity得到−7.75pp，分为target-area+3.82与FP−11.57pp。｜经验数值代入identity；已核对920类各25episodes，因此这里global episode mean与class/fold-weighted mean一致','这两个是exact weighting components，不是causal effect estimates。｜数学结论边界']
]
}
# Synchronize the explicit edits requested and applied during this review.
M['introduction'][0].append('提供可点击GitHub repo link，与Group report的abstract末尾做法一致。｜代码入口')
M['introduction'][4].extend(M['introduction'][5])
del M['introduction'][5]
M['introduction'][4][-1]='将Our contributions are threefold接在前段末尾，随后列出三项贡献。｜结构连接'
M['introduction'][-1][-1]='声明empirical conclusions来自completed first batch，extension正在fresh queries与transfer datasets上独立评价。｜已改为being evaluated，避免暗示完成'
M['experiments'][0][0:1]=['说明COCO-20i与LVIS-92i分别派生于COCO与LVIS，并分别引用dataset来源。｜dataset来源', '说明使用author-prepared class memberships：COCO4folds×20classes，LVIS10folds×92classes；foldprotocol单独引用Matcher。｜split来源']
M['related_work'][3][1]='SegIC把foundation-model correspondences转化为geometric、visual、meta instructions；SINE与VRP-SAM分别展示interaction与reference prompt encoding。｜文献定位'
M['methods'][4][2]='在本次evaluated implementation中，aggregation组合cross-image alignment、original-space seed similarity、candidate coverage。｜实际实现步骤；不混同论文全部细节'
# Re-extract current active prose, keeping escaped percent signs and equation lead-ins.
for name in M:
 t=(D.parent/'paper/sections'/f'{name}.tex').read_text()
 t=re.sub(r'(?m)(?<!\\)%.*','',t)
 for env in ['table\\*?','figure\\*?','equation','minipage']:
  t=re.sub(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}','',t,flags=re.S)
 t=re.sub(r'\\(?:begin|end)\{(?:abstract|enumerate)\}(?:\[[^\]]*\])?','',t)
 t=re.sub(r'\\(?:section|subsection|paragraph)\*?\{([^}]*)\}',r'\n\nHEADING: \1\n\n',t)
 t=re.sub(r'\\label\{[^}]*\}|\\input\{[^}]*\}|\\balance|\\appendix|\\twocolumn\[\{|\\vspace\{[^}]*\}|\}\]','',t)
 t=t.replace('\\item','\n\n')
 paras=[]
 for p in re.split(r'\n\s*\n',t):
  p=' '.join(p.split()).strip()
  if not p or p.startswith('HEADING'):continue
  p=p.replace('Eq.','Eq§')
  paras.append([x.replace('Eq§','Eq.') for x in re.split(r'(?<=[.!?])\s+(?=[A-Z\\])',p)])
 (D/(name+'_sentences.json')).write_text(json.dumps(paras,ensure_ascii=False,indent=2)+'\n')
TITLES={'introduction':'Abstract 与 1. Introduction','related_work':'2. Related Work','methods':'3. Methods','experiments':'4. Experimental Results','interaction_analysis':'4.4 内的 Search–Prompt Interaction','extension':'5. Proposed Method: Agreement-Gated Search (AGS)','gate_analysis':'5.2 内的 Cost–Accuracy Analysis','discussion':'6. Discussion 与 7. Conclusion','appendix':'Appendix A. Protocol Details / B. Additional Results','mathematical_appendix':'Appendix C. Mathematical Analysis'}
PREFIX={'introduction':'I','related_work':'R','methods':'M','experiments':'E','interaction_analysis':'X','extension':'P','gate_analysis':'G','discussion':'D','appendix':'A','mathematical_appendix':'C'}

def locator(s):
 s=re.sub(r'\\(?:cite|ref)\{[^}]*\}','',s)
 s=re.sub(r'\\(?:textbf|emph)\{([^}]*)\}',r'\1',s)
 s=s.replace('~',' ').replace('\\%','%')
 return ' '.join(s.split()[:9])+'…'

head='''# CE6190 Individual Report：逻辑审阅稿（中文）

日期：2026-11-17。对应当前报告 **Visual Correspondence and Concept Guidance for In-Context Segmentation**。这是内容逻辑地图，不是重新写一篇论文。以下把“已完成实验支持的结论”“数学恒等式／界”“待检验的 AGS proposal”分开，便于今晚确定最终表达。

## 1. 全文一句话

**先比较 visual correspondence 与 concept guidance 两套完整 ICS systems 的 accuracy–cost–reference trade-off；再用 component interventions 解释差异，发现 extra search 的边际收益取决于 visual prompts 与 input；据此提出 AGS，并用 cost–accuracy analysis 明确它何时可能有价值、还需要什么实验验证。**

全文主线可压缩为：

1. **任务**：同一 reference image＋mask，怎样在 query 中找到目标？
2. **两条路线**：INSID3 用 visual correspondence；CG-ICS 用 concept guidance＋visual transfer。
3. **第一层答案（system）**：CG-ICS 更准，INSID3 更快；reference robustness 并非一边倒。
4. **第二层答案（mechanism）**：representation 不等于完整系统，feature processing、semantic concepts、visual prompts 都有独立可观察贡献。
5. **关键发现（marginal value）**：额外 search 昂贵，收益随 dataset/query 改变；已有 visual prompts 时边际收益更小。
6. **研究推进（proposal）**：能否用 observable agreement 将 search 分配给更需要它的 inputs？这就是 AGS。
7. **严格可说的结论**：第一批已支持前五点；AGS 的 cost identities 和 agreement bound 已可推导，但其 empirical effectiveness 要由独立控制实验回答。

## 2. 每一部分如何推进主线

| 部分 | 回答的问题 | 本节结束后读者应知道什么 | 与下一节的连接 |
|---|---|---|---|
| Abstract | 做了什么、发现什么、提出什么？ | 完整比较＋component发现＋AGS状态 | 给出阅读路线 |
| 1 Introduction | 为什么不能只比一个 mean IoU？ | 要同时研究system、component、cost、reference | 需要技术背景与方法定义 |
| 2 Related Work | 两条路线从哪里来？ | foundation representations / promptable segmentation → ICS | 解释为何选INSID3与CG-ICS |
| 3 Methods | 两个系统怎样用reference？ | 两条feature path；concept/search与visual boxes；controls的意义 | 知道每个实验实际移除了什么 |
| 4 Experimental Results | 效果、代价、稳定性与贡献分别是什么？ | 高accuracy有cost；components各有作用；search收益异质 | 为conditional search提供具体动机 |
| 5 Proposed Method: AGS | 如何尝试更有选择地使用search？ | agreement rule；cost savings条件；公平controls | 不把proposal当作已成功结果 |
| 6 Discussion | 这些结果共同说明什么？ | system≠backbone；average gain≠每query都受益 | 汇总适用场景与剩余问题 |
| 7 Conclusion | 已回答什么，还剩什么？ | 准确压缩已验收发现与AGS待验证点 | 收尾，不新增结论 |
| Appendix | 哪些细节支持正文但不打断主线？ | protocol/matrix、startup/memory/sweeps、数学推导 | 可复现与可审查 |

### Section 4 的内部推进

**Setup → Accuracy and Efficiency → Reference Sensitivity → Component Ablations（含 interaction）→ Feature Backbones → Parameter Sensitivity → Scale and Failure Analysis。**

先回答“谁更准、成本多少”，再回答“换reference是否稳定”，再解释“哪些parts有效”，最后说明“何时有效、哪里失败”。其中 **Component Ablations → AGS** 是全文最关键的转折；Backbones / Parameters / Failure Analysis 补全研究深度，但不应把这条转折冲散。

## 3. 逐句逻辑地图

编号是**内容定位 ID**，不随PDF换页变化。`I.1.1`表示introduction源文件第1个正文块第1句；`E`为experiments，`X`为插入4.4的interaction，`P/G`为第5节，`D`同时包含Discussion和Conclusion，`A/C`为Appendix。公式前导句和contribution条目也逐项列出；公式本身见第4部分。英文短语用于回到源码定位。每项末尾说明其推理功能或证据类型。

'''
parts=[head]
index=[]
for name,paras in M.items():
 src=json.loads((D/(name+'_sentences.json')).read_text())
 assert len(paras)==len(src),(name,len(paras),len(src))
 parts.append('### '+TITLES[name]+'\n\n')
 for pi,(rows,sents) in enumerate(zip(paras,src),1):
  assert len(rows)==len(sents),(name,pi,len(rows),len(sents))
  parts.append(f'**正文块 {pi}**\n\n')
  for si,(desc,sent) in enumerate(zip(rows,sents),1):
   ident=f'{PREFIX[name]}.{pi}.{si}'
   meaning,role=desc.split('｜',1)
   parts.append(f'- **{ident}** · *{locator(sent)}*\n  {meaning} **作用／依据：**{role}。\n')
   index.append({'id':ident,'source':f'paper/sections/{name}.tex','block':pi,'sentence':si,'english':sent,'chinese':meaning,'role_evidence':role})
  parts.append('\n')
(D/'sentence_logic_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
parts.append('''## 4. 公式、图和表分别承担什么论证任务

### 公式：从定义 → 机制 → 效应 → proposal 条件

| 逻辑ID | 公式／对象 | 它回答什么 | 可以得出什么，不能越过什么 |
|---|---|---|---|
| M1 | M̂q=f(Ir,Mr,Iq) | 输入输出是什么？ | 任务定义，不是新模型推导 |
| M2 | F̃=norm[F(I−BBᵀ)] | INSID3怎样去除positional subspace？ | 解释debiasing intervention；不证明它必然提高IoU |
| M3 | s(c)=sr(c)^α sq(c)^β | CG怎样选择concept？ | reference fidelity和query presence的组合；为fixed-pool ablation提供变量 |
| E1 | episode foreground IoU | 每次预测怎样打分？ | 原分辨率TP/FP/FN，失败不伪装成成功0分 |
| E2 | mean-fold class-pooled IoU | headline benchmark metric是什么？ | class/fold平衡且class内按union加权；与query mean不同 |
| E3 | paired equal-query difference | 同一query的method change是什么？ | references先在query内平均，不把全部reference当独立样本 |
| E4 | 2×2 interaction Γ | search和visual prompts是怎样组合的？ | 当前cohort的negative interaction；不证明agreement能识别低收益queries |
| P1 | A=1[IoU(Mt,Mv)≥τ] | 何时接受initial concept？ | 可观察、无query GT的routing rule |
| P2 | early joint mask / searched joint mask | gate改变哪一步？ | 改search allocation，最终仍做joint segmentation |
| P3 | Cg=Cp+ACe+(1−A)Cs | 每条branch怎么算cost？ | probe/shared prefix必须计算，不能只数省掉几轮 |
| P4 | ΔC=−E[AD]；ΔJ=E[A(Je−Js)] | 节省与accuracy change来自哪些inputs？ | 两项均由accepted subset决定；identity不是经验speedup保证 |
| A1 | |J(Mt,GT)−J(Mv,GT)|≤1−J(Mt,Mv) | 高agreement数学上意味着什么？ | 只保证两probe accuracy彼此接近；两者可以一起错 |
| A2 | Jpool−Jmean=Cov(U,J)/mean(U) | 为什么两种IoU aggregate差很大？ | exact weighting decomposition；FP项不是因果效应 |

### 图：必须展示与文字不同的信息

| 图 | 当前内容 | 在主线里的职责 |
|---|---|---|
| Figure 1 | INSID3独立pipeline，reference/query/mask与original/debiased paths | 让读者看懂feature如何变成mask，支撑E3 ablation的解释 |
| Figure 2 | CG-ICS独立pipeline，concept search与visual transfer汇合 | 让读者看清concept/score/search/box controls分别切在哪里 |
| Figure 3 | component paired effects＋CIs | 一眼对照feature processing、visual prompts、extra search的收益量级；不是新增数据 |
| Figure 4 | 6个paired qualitative cases | 展示方法互补和concept specificity/FP问题；不能据selected cases估计普遍失败率 |

### 表：对应具体问题，避免同一数字多次堆叠

| 顺序／label | 问题 | 证据单位 |
|---|---|---|
| main | 完整系统谁更准？ | headline mean-fold class-pooled IoU＋paired query effect |
| cost | higher accuracy多花多少cost？ | identical inputs、matched runner GPU |
| reference | 换reference后如何变化？ | fixed query targets、distinct references |
| components | 哪些parts有效？ | paired within-system contrasts |
| matrix | trunk与processing能否分开观察？ | 3trunks×2processing，common1008input |
| matrix-scope | 既定E1–E7是否被覆盖？ | uniqueconfig/formaltask与sharedcontrol |
| memory | 除inference latency外还有哪些cost？ | task-equal startup medians与sampledownedmemory |
| sweeps | 参数改动的accuracy sensitivity是什么？ | matchedcohort，image-clusterCI |

## 5. 当前逻辑中值得你决定的几个点

### 5.1 最需要收紧的桥梁：negative interaction → agreement gate

现有证据可靠支持 **“search的平均收益有限且因input而异，因此值得研究conditional search”**。但 **“text/visual agreement恰好能预测search是否有用”** 还是hypothesis。当前正文没有把它写成已经成功，不过Intro/Abstract多次连续出现这两者，容易让读者误以为前者已经验证后者。

建议保留一句更直接的桥梁：**“We hypothesize that agreement between independently prompted masks identifies inputs for which further concept search has limited value.”** 之后立刻用staged always-search、random routing和fresh queries说明如何检验。这里“independently prompted”只能表示不同prompt条件，不能表示statistical independence；可更稳妥改成 **“text-only and visual-only masks”**。

### 5.2 本文主贡献应是已完成的comparative study，AGS是由结果导出的proposal

今晚即使第二批来不及完整验收，论文主线仍完整：system trade-off → reference dependence → component effects → search heterogeneity。标题当前没有将AGS写成已成功主方法，符合现有证据。Contribution 3宜明确写proposal＋analysis，Abstract/Conclusion各保留一处ongoing evaluation即可，不必每页重复提醒pending。

### 5.3 Section 4内容充实，但到Section 5的转折稍远

现在最关键的search发现出现在4.4，随后还经过backbone、parameters、scale/failure才到AGS。可选方案：

- **保持当前结构**：在4.7末补一条短收束句，把cost、variable search gains、semantic failures重新接回input-dependent search；改动最小。
- **更聚焦的结构**：4.4改为Feature Processing，先讲INSID3与backbone；4.5讲Concepts and Search，把CGablation、interaction、searchcost、parameters放一起；4.6讲Failure Analysis，然后直接到AGS。科学范围不减少，但阅读顺序更像一个论证。

不要今晚在未确认内容前大规模搬节。两种结构都能成立，第一种最稳。

### 5.4 Introduction与Discussion/Conclusion有合理重复，但可减少细节重复

Intro需要预告“CG更准、INS更快、search收益异质”；Discussion应解释“为什么system≠backbone，为什么mean gain≠每input受益”；Conclusion只压缩最终答案。现在三处都重复较完整的reference与small-target列表，可把Discussion留解释、Conclusion留一行，不影响研究矩阵。

### 5.5 Mathematical analysis的价值是解释与条件，不是强行prove empirical success

- **interaction**：量化已观察的non-additivity。
- **cost–accuracy identity**：说明AGS节省何时成立、由哪些inputs决定。
- **Jaccard bound**：澄清agreement信号的理论边界，防止错误保证。
- **IoU covariance decomposition**：解释两种aggregate结果为何差异明显。

这四部分各有作用，避免把它们统称为“理论证明AGS有效”。Agreement bound可以短留Appendix；正文costidentity最贴合主线。

### 5.6 数据/措辞需核对的细处

1. Intro末尾已改为 **“the extension is being evaluated separately”**，避免将正在进行的评价写成已完成。
2. Related Work引用的是PerSAM的training-free路线；不要把PerSAM-F的轻量微调与本文frozen inference混为一谈。
3. Appendix的 **55.19% episode mean → 47.44% class-pooled** covariance identity已通过只读SQL核验：LVIS standard的920classes均为25episodes，因此本例global episode mean与class/fold-weighted episode mean相等；−7.7504665pp差值成立。一般情形仍须使用一致aggregationweights，不能随意混用两个均值。核验记录见 `audit/metric_weight_verification.json`。
4. “operation counts explain much of this cost”是合理机制解释，但没有独立profiling分摊时，不写成精确因果归因。
5. Figure4应靠近Section4.7的具体案例，而不是因float跨到AGS/Discussion之后。
6. 第一批4个preprocessing failures与small-query优势不是同一问题：前者发生在**reference resize**，后者按**query foreground area**分组；两者目前同节，需要保持这两个词明确。

## 6. 当前建议保留的最终故事

**本研究的核心不是宣称一种路线全面胜出，而是建立一条从system comparison到component evidence再到adaptive computation proposal的完整链。CG-ICS用更多language/segmentation computation取得更高accuracy；INSID3用更轻的visual correspondence获得更低latency。Ablations揭示visual prompts与search的组合存在diminishing returns，而query-level结果显示search并非对每个input都有效。因此，AGS提出一个具体可检验的问题：能否用text–visual agreement保留有价值的search、减少低价值search？当前报告已经给出比较证据、形式化决策和验证所需controls；最终有效性由第二批已验收结果决定。**

这段是用于讨论的中文storyline，不是建议原样塞进report。
''')
(D/'CE6190_ICS_Logic_Review_ZH.md').write_text(''.join(parts))
validation={'narrative_units':len(index),'per_source':{name:sum(map(len,paras)) for name,paras in M.items()},'excluded':['old_evaluation.tex','old_extension.tex'],'scope':'All active narrative sentences and equation lead-ins; figures/tables/equations separately mapped. Reference bibliography entries are audited separately.','source_sha256':{name:hashlib.sha256((D.parent/'paper/sections'/f'{name}.tex').read_bytes()).hexdigest() for name in M}}
(D/'logic_coverage.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(validation)
