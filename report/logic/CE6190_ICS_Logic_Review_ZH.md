# CE6190 Individual Report：逻辑审阅稿（中文）

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

### Abstract 与 1. Introduction

**正文块 1**

- **I.1.1** · *In-context segmentation transfers a target from an annotated reference…*
  定义 ICS：把 annotated reference 中指定的目标迁移到 query。 **作用／依据：**任务定义。
- **I.1.2** · *We compare two training-free systems, INSID3 and CG-ICS, to…*
  交代比较对象 INSID3 / CG-ICS，以及 accuracy、reference sensitivity、inference cost 三条评价轴。 **作用／依据：**研究问题。
- **I.1.3** · *On COCO-$20^i$ and LVIS-$92^i$, CG-ICS achieves mean-fold class-pooled IoU…*
  给出主结果：CG-ICS 的 COCO/LVIS mean-fold class-pooled IoU 为 72.17/47.44，INSID3 为 57.65/40.77。 **作用／依据：**第一批主表。
- **I.1.4** · *This improvement comes with 7.37--10.63 times greater mean inference…*
  说明 accuracy 的代价：匹配 runner GPU model 后 mean inference latency 为 7.37–10.63 倍，且另占 MLLM GPU。 **作用／依据：**第一批成本。
- **I.1.5** · *Component experiments identify distinct sources of performance: INSID3 benefits…*
  归纳两系统的有效成分：INSID3 的 debiasing / aggregation，CG-ICS 的 semantic concept + visual prompts。 **作用／依据：**第一批 ablation。
- **I.1.6** · *Additional concept search yields only a 0.22-point equal-query IoU…*
  指出额外 concept search 的平均收益不一致：COCO 0.22 pp，LVIS 2.30 pp。 **作用／依据：**第一批 ablation。
- **I.1.7** · *In the paired-input cohort, visual prompting reduces the marginal…*
  概括 search–prompt interaction：加入 visual prompting 后 search 的边际收益降低。 **作用／依据：**第一批配对 interaction。
- **I.1.8** · *Reference variation and small-target failures further distinguish the systems.…*
  补充 reference variation 与 small-target failure，避免只看均值。 **作用／依据：**第一批 robustness / failure。
- **I.1.9** · *These findings motivate Agreement-Gated Search (AGS), for which we…*
  从证据导向 AGS proposal，并引出 cost–accuracy decomposition 与 controlled evaluation。 **作用／依据：**方法动机＋数学分析。
- **I.1.10** · *The report combines completed comparative evidence with this extension,…*
  界定当前贡献状态：比较实验完成，extension 的 empirical results 尚待验收。 **作用／依据：**状态边界。
- **I.1.11** · *Code is available at \url{https://github.com/AvianHan/CE6190-Visual-Understanding}.…*
  提供可点击GitHub repo link，与Group report的abstract末尾做法一致。 **作用／依据：**代码入口。

**正文块 2**

- **I.2.1** · *In-context segmentation (ICS) identifies a target in a query…*
  给出 ICS 的三项输入和目标。 **作用／依据：**任务定义。
- **I.2.2** · *This interface allows a user to specify the target…*
  解释这种接口的价值：用户给 visual example，无需逐张 query 标注或重新训练 task-specific model。 **作用／依据：**应用动机。
- **I.2.3** · *Reliable transfer remains difficult: the same category can change…*
  列出 transfer 的困难：appearance change、partial reference、distractors。 **作用／依据：**问题来源。
- **I.2.4** · *A practical system must resolve these ambiguities at an…*
  把困难收束为一个工程目标：在可接受 inference cost 下消歧。 **作用／依据：**引出 accuracy–cost trade-off。

**正文块 3**

- **I.3.1** · *Recent foundation models support two distinct solutions.…*
  引入 foundation model 提供的两条解决路线。 **作用／依据：**方法分叉。
- **I.3.2** · *INSID3 constructs a mask from frozen DINOv3 features through…*
  INSID3 通过 frozen DINOv3 的 debiasing、clustering、similarity aggregation 构造 mask。 **作用／依据：**引用主方法。
- **I.3.3** · *CG-ICS generates language concepts from the reference, evaluates and…*
  CG-ICS 通过 reference-derived concepts、search 和 SAM 3 visual prompts 构造 mask。 **作用／依据：**引用主方法。
- **I.3.4** · *Their central difference is how the reference becomes actionable…*
  总结比较轴：reference 如何转化为可用于 query 的 evidence。 **作用／依据：**概念对照。

**正文块 4**

- **I.4.1** · *A headline accuracy comparison leaves important behavior unresolved.…*
  指出 headline accuracy comparison 本身不足。 **作用／依据：**研究缺口。
- **I.4.2** · *An improvement may depend on the reference, arise from…*
  列出待解释因素：reference、processing stage、不同 input 的 search 收益。 **作用／依据：**拆分问题。
- **I.4.3** · *Complete-system evaluation therefore needs to be connected to component…*
  因此需要把 whole-system evaluation 与 component intervention、measured cost 连接起来。 **作用／依据：**研究设计。
- **I.4.4** · *This connection is also necessary for designing an efficient…*
  进一步导出效率目标：减少低 marginal value 的 search，保留初始预测不足时的 search。 **作用／依据：**proposal 的一般原则。

**正文块 5**

- **I.5.1** · *We study these questions on COCO-$20^i$ and LVIS-$92^i$ using…*
  限定本研究的数据与协议：COCO/LVIS，common inputs，original-resolution evaluation。 **作用／依据：**实验范围。
- **I.5.2** · *CG-ICS is more accurate overall, while INSID3 is substantially…*
  先给总体发现：CG-ICS 更准，INSID3 更快。 **作用／依据：**第一批主结论。
- **I.5.3** · *The component results sharpen this comparison: a single concept…*
  再给机制发现：COCO 的 single concept + visual prompts 已获得多数收益，LVIS 更受益于 extra search。 **作用／依据：**第一批 component 结论。
- **I.5.4** · *Moreover, visual prompting reduces the marginal search gain on…*
  指出 visual prompting 在两个数据集都降低 search 的 observed marginal gain。 **作用／依据：**第一批 interaction。
- **I.5.5** · *This measured interaction motivates an input-dependent search rule based…*
  从此提出可观察 signal：text-only / visual-only predictions 的 agreement，用于 input-dependent search。 **作用／依据：**有动机的假说；不是已验证 gate 性能。
- **I.5.6** · *Our contributions are threefold:…*
  将Our contributions are threefold接在前段末尾，随后列出三项贡献。 **作用／依据：**结构连接。

**正文块 6**

- **I.6.1** · *Paired system evaluation. We compare accuracy, reference sensitivity, and…*
  Contribution 1：common inputs 上比较 accuracy、reference sensitivity、computational cost。 **作用／依据：**已完成经验贡献。

**正文块 7**

- **I.7.1** · *Component analysis. We separate feature processing, semantic search, visual…*
  Contribution 2：通过 controlled interventions 区分 feature processing、semantic search、visual prompting、backbone choice。 **作用／依据：**已完成经验贡献。

**正文块 8**

- **I.8.1** · *Agreement-Gated Search. We propose input-dependent search with an explicit…*
  Contribution 3：提出 AGS，给出 explicit cost–accuracy formulation 和 predefined controls。 **作用／依据：**proposal＋分析贡献。
- **I.8.2** · *The empirical conclusions use the completed first batch; the…*
  声明empirical conclusions来自completed first batch，extension正在fresh queries与transfer datasets上独立评价。 **作用／依据：**已改为being evaluated，避免暗示完成。

### 2. Related Work

**正文块 1**

- **R.1.1** · *Visual understanding requires both semantic recognition and spatial localization.…*
  从 visual understanding 的两项需求开场：semantic recognition 和 spatial localization。 **作用／依据：**建立上位概念。
- **R.1.2** · *Vision Transformers represent images as patch tokens , while…*
  ViT 的 patch tokens 与 DINO 的 self-supervised object structure 提供 feature 路线背景。 **作用／依据：**文献：ViT / DINO。
- **R.1.3** · *DINOv2 and DINOv3 extend this representation approach at scale…*
  DINOv2 / DINOv3 将这一 representation 路线扩展到更大规模。 **作用／依据：**文献：DINOv2 / DINOv3。
- **R.1.4** · *Their dense features support correspondence across images, providing a…*
  说明这些 dense features 如何服务跨图 correspondence，从而服务 reference→query transfer。 **作用／依据：**与本文任务连接。

**正文块 2**

- **R.2.1** · *Promptable foundation models expose spatial and semantic information through…*
  转向 promptable foundation models，把 semantic/spatial information 变成 segmentation interface。 **作用／依据：**第二条技术路线。
- **R.2.2** · *SAM predicts masks from spatial prompts ; SAM 2…*
  按 SAM→SAM 2→SAM 3 串起 spatial prompts、video、concept prompts。 **作用／依据：**文献定位。
- **R.2.3** · *CLIP connects visual representations with language through image--text alignment…*
  CLIP 的 image–text alignment 与 Qwen3-VL 的 multimodal reasoning 补足 language 一侧。 **作用／依据：**文献定位。
- **R.2.4** · *Together, these capabilities enable the two routes studied here:…*
  把文献能力重新收束为本文的 visual correspondence vs semantic grounding 两条路线。 **作用／依据：**回到研究轴。

**正文块 3**

- **R.3.1** · *Reference-conditioned segmentation builds on one-shot prediction , support prototypes…*
  从 one-shot、prototypes、dense correspondence 追溯 reference-conditioned segmentation。 **作用／依据：**历史背景。
- **R.3.2** · *Visual in-context learning broadens this idea by expressing tasks…*
  用 visual prompting、Painter、SegGPT 说明 example 本身可以表达 task。 **作用／依据：**In-context learning 脉络。
- **R.3.3** · *These approaches establish the annotated example as a flexible…*
  归纳 annotated example 是 desired output 的 flexible specification。 **作用／依据：**本节共同概念。

**正文块 4**

- **R.4.1** · *Recent methods differ in how they turn that example…*
  指出近期方法的区别在于 example 如何变成 query mask。 **作用／依据：**分类原则。
- **R.4.2** · *SegIC turns foundation-model correspondences into geometric, visual, and meta…*
  SegIC把foundation-model correspondences转化为geometric、visual、meta instructions；SINE与VRP-SAM分别展示interaction与reference prompt encoding。 **作用／依据：**文献定位。
- **R.4.3** · *Frozen-model pipelines instead reuse pretrained representations and segmentation priors:…*
  PerSAM 和 Matcher 展示 frozen representations 与 segmentation priors 的组合。 **作用／依据：**冻结模型方法位置。
- **R.4.4** · *INSID3 operates directly on DINOv3 features , whereas CG-ICS…*
  将 INSID3 的直接 feature processing 与 CG-ICS 的 concepts/search/visual exemplars 对照。 **作用／依据：**定位本研究两主方法。
- **R.4.5** · *We compare these last two complete inference systems and…*
  明确本文选取这两个 full inference systems，并继续拆解其 stages。 **作用／依据：**收束 Related Work。

**正文块 5**

- **R.5.1** · *The chosen reference is itself a source of variation.…*
  提醒 reference 本身会引入 variability。 **作用／依据：**引入 robustness 文献。
- **R.5.2** · *Good Examples studies example selection , UNICL-SAM addresses uncertainty…*
  Good Examples、UNICL-SAM、CG-ICS 各自讨论 example selection、uncertainty、reference robustness。 **作用／依据：**文献与研究缺口。
- **R.5.3** · *Our reference study measures performance when the query is…*
  本文固定 query、改变 demonstration，量化 reference sensitivity。 **作用／依据：**对应 E2。
- **R.5.4** · *Cross-domain work provides a complementary perspective: RePRI uses transductive…*
  RePRI 的 transductive optimization 与 ABCDFSS 的 test-time adaptation 提供 cross-domain 参照。 **作用／依据：**transfer 背景。
- **R.5.5** · *Their adaptation mechanisms differ from the frozen inference evaluated…*
  将这些 adaptation mechanisms 与当前 frozen inference 区分，解释为何 domain transfer 需要单独评价。 **作用／依据：**范围连接。

### 3. Methods

**正文块 1**

- **M.1.1** · *Given a reference image $I_r$, its foreground mask $M_r$,…*
  给出形式化接口：输入 reference image、reference mask、query image，输出 query mask。 **作用／依据：**公式 M1。

**正文块 2**

- **M.2.1** · *The task is binary segmentation of the demonstrated semantic…*
  限定输出为所示 semantic target 的 binary segmentation。 **作用／依据：**任务范围。
- **M.2.2** · *Query ground truth is used only for evaluation; the…*
  query GT 仅用于 evaluation；inference 不接收 dataset class label。 **作用／依据：**防止 label leakage 的核心约束。
- **M.2.3** · *Both systems use pretrained models without task-specific parameter updates.…*
  两系统均使用 pretrained models，不做 task-specific parameter updates。 **作用／依据：**training-free 范围。
- **M.2.4** · *We first compare their complete pipelines, then vary components…*
  说明比较顺序：full pipelines→within-system components→common interface 下的 trunks。 **作用／依据：**实验设计预告。

**正文块 3**

- **M.3.1** · *INSID3 uses the semantic correspondence and spatial grouping structure…*
  INSID3 的信息来源是 frozen DINOv3 的 semantic correspondence 与 spatial grouping。 **作用／依据：**主方法解释。
- **M.3.2** · *Figure separates two feature paths.…*
  Figure 1 将 original / debiased 两条 feature paths 分开。 **作用／依据：**图文索引。
- **M.3.3** · *Original query features provide coherent clusters and within-query similarity.…*
  original query features 用于 coherent clustering 和 within-query similarity。 **作用／依据：**分支职责。
- **M.3.4** · *Debiased reference and query features provide cross-image matching, reducing…*
  debiased reference/query features 用于 cross-image matching，弱化 absolute position。 **作用／依据：**分支职责。

**正文块 4**

- **M.4.1** · *Let $F$ be a matrix of dense features and…*
  定义 dense-feature matrix F 与 positional-subspace orthonormal basis B。 **作用／依据：**公式变量。
- **M.4.2** · *Debiasing projects onto its orthogonal complement and normalizes each…*
  通过投影到 positional subspace 的 orthogonal complement 再 normalize 实现 debiasing。 **作用／依据：**公式 M2。

**正文块 5**

- **M.5.1** · *A foreground prototype averages debiased reference features inside $M_r$.…*
  在 reference foreground 内平均 debiased features，构造 foreground prototype。 **作用／依据：**算法步骤。
- **M.5.2** · *Forward similarity and backward correspondence identify query candidates, and…*
  forward similarity / backward correspondence 找 candidates，prototype 选择 seed cluster。 **作用／依据：**算法步骤。
- **M.5.3** · *In the evaluated implementation, final aggregation combines each cluster's…*
  在本次evaluated implementation中，aggregation组合cross-image alignment、original-space seed similarity、candidate coverage。 **作用／依据：**实际实现步骤；不混同论文全部细节。
- **M.5.4** · *Clusters above the merge threshold form the prediction.…*
  超过 merge threshold 的 clusters 组成最终 mask。 **作用／依据：**算法输出。

**正文块 6**

- **M.6.1** · *Our component study crosses debiasing on/off with self-plus-cross, cross-only,…*
  INSID3 ablation：debiasing on/off × 三种 aggregation；seed-only 仍含 clustering。 **作用／依据：**控制变量。
- **M.6.2** · *The simple-similarity control retains debiasing but omits clustered region…*
  simple-similarity control 保留 debiasing，但不做 clustered region processing。 **作用／依据：**控制组定义。
- **M.6.3** · *The main configuration uses input size 1024, positional rank…*
  列出主设置：1024 input、rank 500、clustering 0.6、merge 0.2、bilinear restoration。 **作用／依据：**可复现设置。
- **M.6.4** · *Implementation-specific details are recorded in Appendix .…*
  implementation-specific details 放入 Appendix。 **作用／依据：**正文与附录分工。

**正文块 7**

- **M.7.1** · *CG-ICS combines concept search with a parallel visual transfer…*
  CG-ICS 包含 concept search 与 parallel visual transfer。 **作用／依据：**结构定义。
- **M.7.2** · *A multimodal language model generates candidate concepts from the…*
  MLLM 从 reference 与 highlighted target 生成 candidate concepts。 **作用／依据：**concept generation。
- **M.7.3** · *Each concept $c$ is scored by reference fidelity and…*
  按 reference fidelity 和 query presence 给 candidate concept 打分。 **作用／依据：**公式 M3。

**正文块 8**

- **M.8.1** · *The query term $s_q$ is SAM 3's presence score.…*
  query score s_q 为 SAM 3 presence score。 **作用／依据：**变量解释。
- **M.8.2** · *Candidate expansion, pruning, restarts, and reranking seek a concept…*
  expansion、pruning、restarts、reranking 寻找同时匹配 reference 且存在于 query 的 concept。 **作用／依据：**search 机制。
- **M.8.3** · *The default uses $\alpha=\beta=1$.…*
  默认 score exponents α=β=1。 **作用／依据：**可复现设置。

**正文块 9**

- **M.9.1** · *The visual branch concatenates the reference and query.…*
  visual branch 将 reference 和 query 拼接。 **作用／依据：**算法步骤。
- **M.9.2** · *Reference-mask components provide spatial prompts on this combined image;…*
  reference-mask components 产生 spatial prompts，query-side outputs 转为 box prompts。 **作用／依据：**算法步骤。
- **M.9.3** · *SAM 3 then combines these boxes with the selected…*
  SAM 3 结合这些 boxes 与 selected concept 输出 query segmentation。 **作用／依据：**融合位置；Figure 2。
- **M.9.4** · *We compare a single concept, scored candidates, and full…*
  比较 single concept、scored candidates、full search，各配 with/without visual boxes。 **作用／依据：**factorial design。
- **M.9.5** · *Visual-only and direct-concatenation outputs provide additional controls.…*
  visual-only 与 direct concatenation 是额外 controls。 **作用／依据：**排除替代解释。

**正文块 10**

- **M.10.1** · *The implemented system uses Qwen3-VL-4B-Instruct for concept generation ,…*
  实现使用 Qwen3-VL-4B-Instruct、working size 512、5 initial candidates、3 search loops。 **作用／依据：**具体运行配置。
- **M.10.2** · *SAM 3 applies its own size-1008 preprocessing.…*
  SAM 3 内部另做 size-1008 preprocessing。 **作用／依据：**解释尺度接口。
- **M.10.3** · *The language model and segmenter occupy separate GPUs; this…*
  MLLM 与 segmenter 分占 GPUs；system cost 需要计入额外 MLLM allocation。 **作用／依据：**成本口径。

### 4. Experimental Results

**正文块 1**

- **E.1.1** · *We evaluate COCO-$20^i$ and LVIS-$92^i$, derived from COCO and…*
  说明COCO-20i与LVIS-92i分别派生于COCO与LVIS，并分别引用dataset来源。 **作用／依据：**dataset来源。
- **E.1.2** · *We use the author-prepared class memberships, with four folds…*
  说明使用author-prepared class memberships：COCO4folds×20classes，LVIS10folds×92classes；foldprotocol单独引用Matcher。 **作用／依据：**split来源。
- **E.1.3** · *The standard comparison contains 4,000 COCO and 23,000 LVIS…*
  standard comparison 每系统 4,000 COCO、23,000 LVIS episodes。 **作用／依据：**样本量。
- **E.1.4** · *A reference cohort varies the demonstration for fixed query…*
  reference cohort 固定 query target，改变 demonstration。 **作用／依据：**E2 设计。
- **E.1.5** · *A separate component cohort contains 400 COCO and 1,000…*
  component cohort 为 400 COCO、1,000 LVIS query targets，各最多3 references，对应1,200/2,757 episodes。 **作用／依据：**component 样本量。
- **E.1.6** · *All methods within a comparison receive matched inputs.…*
  同一 comparison 内各方法使用 matched inputs。 **作用／依据：**配对设计。

**正文块 2**

- **E.2.1** · *For each episode, foreground IoU is evaluated at the…*
  每 episode 在 original query resolution 计算 foreground IoU。 **作用／依据：**公式 E1。

**正文块 3**

- **E.3.1** · *The headline metric pools counts within each class, averages…*
  headline metric：class 内 pool TP/FP/FN，fold 内平均 classes，再平均 folds。 **作用／依据：**公式 E2。

**正文块 4**

- **E.4.1** · *For paired comparisons, we first average planned episodes within…*
  paired metric：先对同一 query target 的 planned episodes 平均，再对 method difference 等权平均。 **作用／依据：**公式 E3。

**正文块 5**

- **E.5.1** · *Confidence intervals resample whole query-image clusters, keeping their concepts…*
  bootstrap 按完整 query-image clusters 重采样，concepts 和 references 一起保留。 **作用／依据：**统计独立单位。
- **E.5.2** · *We report exploratory 95% intervals using 2,000 bootstrap replicates…*
  主/component contrasts 用2,000次，parameter/area用1,000次，给 exploratory 95% CIs。 **作用／依据：**不确定性设置。
- **E.5.3** · *Reference sensitivity is the standard deviation across distinct references…*
  reference SD 先合并 identical inputs 的 repeated executions，再在不同 references 间计算。 **作用／依据：**robustness 指标。
- **E.5.4** · *Failed predictions retain missing accuracy; paired contrasts use common…*
  failed predictions 保留 missing accuracy，paired contrasts 使用 common successful inputs。 **作用／依据：**失败处理口径。

**正文块 6**

- **E.6.1** · *The completed first batch covers 36 configurations and 615,976…*
  第一批有36 configurations、615,976 records，其中4 preprocessing failures。 **作用／依据：**覆盖范围。
- **E.6.2** · *All expected episode identities and shard assignments have been…*
  已核对 expected episode identities 和 shard assignments。 **作用／依据：**审计证据。
- **E.6.3** · *We report each dataset separately and use the executed…*
  各 dataset 分开报告，在实际执行的 common protocol 上比较。 **作用／依据：**解释范围。
- **E.6.4** · *Protocol details and the full research matrix are in…*
  完整 matrix 与 protocol 放 Appendix。 **作用／依据：**正文导航。

**正文块 7**

- **E.7.1** · *CG-ICS obtains higher accuracy on both datasets (Table ).…*
  CG-ICS 在两个 dataset 的 headline accuracy 更高。 **作用／依据：**Table 1。
- **E.7.2** · *Its paired equal-query advantage is 14.01 points on COCO…*
  paired equal-query 优势为 COCO14.01 pp、LVIS16.22 pp。 **作用／依据：**第一批配对效应。
- **E.7.3** · *The headline class-pooled and paired metrics answer different questions:…*
  区分指标：class pooling 对 episode 按 union size 加权，equal-query 对 query target 等权。 **作用／依据：**避免把两列当同一均值。
- **E.7.4** · *On LVIS, CG-ICS's episode-mean IoU is 55.19%, compared with…*
  LVIS CG-ICS episode mean55.19%，mean-fold class pooling47.44%。 **作用／依据：**指标差异实例。
- **E.7.5** · *Appendix gives the exact weighting identity behind this distinction.…*
  Appendix 用 exact weighting identity 解释差异。 **作用／依据：**连接数学分析。

**正文块 8**

- **E.8.1** · *This accuracy advantage has a substantial computational cost.…*
  accuracy advantage 带来 substantial cost。 **作用／依据：**引出 trade-off。
- **E.8.2** · *On exact inputs matched by recorded runner GPU, CG-ICS…*
  same-input/same-runner-GPU 比较中，CG latency 为 INSID3 的7.37–10.63倍。 **作用／依据：**Table 2。
- **E.8.3** · *These times cover inference and exclude model/server startup, evaluation,…*
  这些 times 仅含 inference，不含 startup、evaluation、writing。 **作用／依据：**成本定义。
- **E.8.4** · *The extra language-model GPU makes the resource difference larger…*
  CG 额外 MLLM GPU 意味着资源差异不止 latency ratio。 **作用／依据：**system-level cost。

**正文块 9**

- **E.9.1** · *The operation counts explain much of this cost: CG-ICS…*
  CG 每 standard episode 约3.43–3.50 MLLM requests、29.89–32.76 SAM calls，解释主要开销。 **作用／依据：**operation counts。
- **E.9.2** · *Language-model startup adds a median 58--71 seconds, which matters…*
  MLLM startup median58–71s，小 batch 时明显。 **作用／依据：**启动成本。
- **E.9.3** · *Startup and owned-process memory are reported separately in Appendix…*
  startup 与 owned-process memory 另放 Appendix。 **作用／依据：**附录导航。

**正文块 10**

- **E.10.1** · *The complete-system accuracy ranking does not determine the variability…*
  accuracy ranking 不自动决定 variability ranking。 **作用／依据：**新的研究问题。
- **E.10.2** · *CG-ICS reduces reference SD on COCO but increases it…*
  CG 在 COCO 降低 reference SD，在 LVIS 增加。 **作用／依据：**Table 3。
- **E.10.3** · *Nevertheless, its mean worst-reference IoU is higher on both…*
  但 CG 在两个数据集的 mean worst-reference IoU 均更高。 **作用／依据：**Table 3。
- **E.10.4** · *A system can therefore improve the lower end of…*
  因此更高 worst-case-level accuracy 可以与更大 reference sensitivity 同时出现。 **作用／依据：**解释现象。

**正文块 11**

- **E.11.1** · *The same direction appears in H200-only subsets: the CG-ICS…*
  H200-only subsets 中 reference SD difference 的区间仍一负一正。 **作用／依据：**硬件分层稳健性检查。
- **E.11.2** · *Reference robustness is therefore dataset-dependent, and reduced variability should…*
  reference robustness 是 dataset-dependent，variability 必须结合 accuracy 一起看。 **作用／依据：**本节结论。

**正文块 12**

- **E.12.1** · *A three-seed study fixes the component inputs and yields…*
  three-seed 固定 component inputs，CG mean per-input SD为2.23/3.65 pp。 **作用／依据：**run sensitivity。
- **E.12.2** · *Prediction hashes are identical across all three seeds for…*
  263/1,200与694/2,757 inputs 的 prediction hashes 跨3 seeds完全一致。 **作用／依据：**离散输出稳定性。
- **E.12.3** · *Because some runs use different GPU models, we interpret…*
  部分运行混合 GPU model，因此这是 run sensitivity，不是 isolated seed effect。 **作用／依据：**归因边界。

**正文块 13**

- **E.13.1** · *INSID3's gains are not explained by the feature trunk…*
  INSID3 的收益不只来自 feature trunk。 **作用／依据：**component 问题。
- **E.13.2** · *Removing debiasing from the full processing route reduces equal-query…*
  移除 debiasing 使 COCO/LVIS equal-query IoU下降2.38/2.26 pp。 **作用／依据：**paired ablation。
- **E.13.3** · *Replacing self-plus-cross aggregation with seed-only output reduces it by…*
  self-plus-cross 改为seed-only下降9.24/4.06 pp。 **作用／依据：**paired ablation。
- **E.13.4** · *The full processing route improves over debiased simple similarity…*
  full processing 相比debiased similarity提高19.23/11.94 pp。 **作用／依据：**paired ablation。
- **E.13.5** · *These contrasts indicate that both the representation adjustment and…*
  representation adjustment 与 clustered spatial structure 均有贡献。 **作用／依据：**机制归纳。

**正文块 14**

- **E.14.1** · *For CG-ICS, language and visual prompts provide complementary information.…*
  CG 的 language 与 visual prompts 提供互补信息。 **作用／依据：**component 问题。
- **E.14.2** · *Adding visual boxes to a single concept yields 5.39…*
  single concept 加boxes提高5.39/4.78 pp；visual-only加concept提高13.38/16.54 pp。 **作用／依据：**paired interventions。
- **E.14.3** · *By contrast, the incremental return from more search depends…*
  extra search 的收益随 dataset 和已有visual route而变。 **作用／依据：**从互补转向边际收益。

**正文块 15**

- **E.15.1** · *With visual boxes, full search adds 0.22 points beyond…*
  带boxes时full search比single concept在COCO只提高0.22 pp且CI跨0，LVIS提高2.30 pp。 **作用／依据：**Table 4。
- **E.15.2** · *Without boxes, the gains rise to 3.07 and 5.30…*
  不带boxes时search收益为3.07/5.30 pp。 **作用／依据：**factorial comparison。
- **E.15.3** · *Thus, the observed contribution of search depends on both…*
  search贡献依赖dataset与visual route。 **作用／依据：**引出interaction公式。

**正文块 16**

- **E.16.1** · *The single-concept route is also much cheaper.…*
  single-concept route成本显著更低。 **作用／依据：**引出conditional search。
- **E.16.2** · *Full search takes 4.17 times as long on 300…*
  matched subsets中full search耗时为single的4.17倍或4.37–4.56倍。 **作用／依据：**相同inputs / GPU成本。
- **E.16.3** · *The accuracy changes on those same timing subsets are…*
  同一timing subsets的accuracy changes为COCO−0.54、LVIS/L40S+1.53、LVIS/H200+3.67 pp。 **作用／依据：**同口径trade-off。
- **E.16.4** · *These paired observations motivate testing conditional search on fresh…*
  因此值得在fresh inputs测试conditional search。 **作用／依据：**动机，不是gate性能证明。

**正文块 17**

- **E.17.1** · *The current candidate-stage comparison changes both candidate generation and…*
  现有candidate-stage变化同时改变generation/selection，full search还包含restarts。 **作用／依据：**说明当前干预并非纯score ablation。
- **E.17.2** · *The next batch therefore adds a fixed-pool experiment to…*
  第二批fixed-pool实验将固定candidate evidence，单独测试score terms。 **作用／依据：**明确剩余问题。

**正文块 18**

- **E.18.1** · *INSID3 processing improves all three feature trunks (Table ).…*
  INSID3 processing提高三种trunks的表现。 **作用／依据：**Table 5。
- **E.18.2** · *DINOv3 remains strongest under both processing routes, while equal-query…*
  两种processing下DINOv3均最好；equal-query processing gain在COCO15.49–19.39、LVIS11.17–11.95 pp。 **作用／依据：**representation/processing matrix。
- **E.18.3** · *Representation quality and mask-construction strategy therefore make distinct contributions.…*
  representation与mask construction各有贡献。 **作用／依据：**分离解释。
- **E.18.4** · *The SAM 3 entry is a trunk adapted to…*
  SAM 3这里只是common feature interface中的trunk；不是full promptable decoder比较，也未逐backbone优化。 **作用／依据：**关键比较边界。

**正文块 19**

- **E.19.1** · *Single-factor sweeps show that increasing computation does not consistently…*
  single-factor sweeps显示更多compute不保证更高accuracy。 **作用／依据：**参数研究结论。
- **E.19.2** · *Raising the INSID3 clustering threshold from 0.6 to 0.7…*
  clustering threshold0.6→0.7对COCO−1.53、LVIS+1.85 pp。 **作用／依据：**dataset-dependent sensitivity。
- **E.19.3** · *Reducing input size to 768 changes accuracy by $-0.88$…*
  input1024→768对accuracy−0.88/−2.25 pp，latency缩为默认0.39–0.41。 **作用／依据：**效率操作点。

**正文块 20**

- **E.20.1** · *For CG-ICS, using one initial candidate changes equal-query IoU…*
  CG initial candidates5→1对accuracy−0.16/−0.82 pp，matched latency缩为0.63–0.66。 **作用／依据：**效率操作点。
- **E.20.2** · *One search loop has little observed COCO penalty but…*
  search loops3→1对COCO观察到很小影响，LVIS−0.75 pp。 **作用／依据：**参数收益。
- **E.20.3** · *Accuracy effects use the full component cohort, while timing…*
  accuracy来自full component cohort，timing来自same-GPU subsets。 **作用／依据：**配对口径。
- **E.20.4** · *Full intervals are in Appendix .…*
  完整CIs见Appendix。 **作用／依据：**正文压缩。

**正文块 21**

- **E.21.1** · *CG-ICS's accuracy advantage is largest on small query targets.…*
  CG在small query targets上的优势最大。 **作用／依据：**尺度规律。
- **E.21.2** · *For foreground area below 1% of the image, its…*
  foreground<1%时paired gain为21.88/24.03 pp。 **作用／依据：**area-bin结果。
- **E.21.3** · *The gain remains positive but smaller in the 1--10%…*
  更大两档仍为正，但幅度更小。 **作用／依据：**趋势完整性。
- **E.21.4** · *Here size means semantic foreground proportion, measured on the…*
  size指query semantic foreground proportion。 **作用／依据：**变量定义。

**正文块 22**

- **E.22.1** · *The examples in Figure reveal complementary failure modes.…*
  Figure4展示不同方法互补的成功与失败。 **作用／依据：**定性分析。
- **E.22.2** · *CG-ICS recovers a book and monitor missed by INSID3.…*
  CG找到INS漏掉的book和monitor。 **作用／依据：**具体案例。
- **E.22.3** · *INSID3 succeeds on a food target for which CG-ICS…*
  INS找到food，而CG在salmon medallion概念下选到distractor。 **作用／依据：**反方向案例。
- **E.22.4** · *Both methods accurately trace a stop sign, and both…*
  两者都能分stop sign，也都在另一个food target失败。 **作用／依据：**共同成功/失败。
- **E.22.5** · *A bowl example illustrates excessive background coverage.…*
  bowl案例显示过度覆盖background。 **作用／依据：**FP失败类型。

**正文块 23**

- **E.23.1** · *Other inspected cases connect semantic specificity to the predicted…*
  floor unit、buttered bread、vehicle三个概念分别造成目标范围错误。 **作用／依据：**semantic specificity案例。
- **E.23.2** · *These cases motivate testing concept quality as well as…*
  因此除了mask quality，也应检验concept quality。 **作用／依据：**引出score/generation研究。

**正文块 24**

- **E.24.1** · *The four execution errors arise earlier, during preprocessing.…*
  四次execution errors发生在preprocessing。 **作用／依据：**区分模型错误与执行失败。
- **E.24.2** · *Three distinct reference masks contain only one or five…*
  3个不同reference masks仅1或5 pixels，legacy nearest resize到512²后foreground消失。 **作用／依据：**直接失败证据。
- **E.24.3** · *CG-ICS then stops before model inference.…*
  所以CG在inference前停止。 **作用／依据：**因果链。
- **E.24.4** · *A foreground-preserving resize is a concrete future improvement, to…*
  建议foreground-preserving resize，并在原失败与independent small-reference examples上测试。 **作用／依据：**具体future work。
- **E.24.5** · *The 20-case inspection recomputed all 36 available prediction masks…*
  20case中的36个available masks已重算并复现savedcounts/IoU，正文展示6例。 **作用／依据：**定性证据审计。

### 4.4 内的 Search–Prompt Interaction

**正文块 1**

- **X.1.1** · *Let $J_u(s,v)$ denote reference-averaged IoU for query target $u$,…*
  定义J_u(s,v)：query-level reference-averaged IoU；s=single/full search，v=without/with boxes。 **作用／依据：**2×2变量。
- **X.1.2** · *The interaction contrast is…*
  定义difference-in-differences Γ。 **作用／依据：**公式 E4。

**正文块 2**

- **X.2.1** · *This measures how much search changes the marginal benefit…*
  Γ衡量search如何改变visual prompting的marginal benefit；等价地也衡量visual prompting如何改变search收益。 **作用／依据：**interaction含义。
- **X.2.2** · *The observed interaction is $-2.85$ percentage points on COCO…*
  Γ为COCO−2.85 pp、LVIS−3.00 pp，95%CIs均不跨0。 **作用／依据：**第一批配对统计。
- **X.2.3** · *Thus, on the full matched input cohort, the two…*
  在full matched cohort，两种addition呈diminishing joint gains；boxes两种search level都有效，但single concept时更有效。 **作用／依据：**可支持的结论。

### 5. Proposed Method: Agreement-Gated Search (AGS)

**正文块 1**

- **P.1.1** · *The component results motivate allocating search according to the…*
  component结果提示按input分配search。 **作用／依据：**proposal动机。
- **P.1.2** · *Visual prompting reduces the observed marginal benefit of concept…*
  visual prompting降低observed marginal search benefit，但剩余收益随dataset/query变化。 **作用／依据：**证据桥梁。
- **P.1.3** · *We propose Agreement-Gated Search (AGS), which uses consistency between…*
  AGS用初始text-only与visual-only predictions的一致程度决定继续search。 **作用／依据：**方法概念。
- **P.1.4** · *AGS applies input-dependent computation allocation to the concept-search stage…*
  将input-dependent computation allocation思想用到CG concept-search stage。 **作用／依据：**与adaptive inference文献连接。

**正文块 2**

- **P.2.1** · *For the first reference-derived concept $c_0$, let $M_t(c_0)$ denote…*
  定义initial concept c0、text-only mask Mt和visual-only mask Mv。 **作用／依据：**decision变量。
- **P.2.2** · *If either probe is empty, set $A=0$; otherwise define…*
  任一probe为空则不接受；否则计算两mask IoU，超过τ则A=1。 **作用／依据：**公式 P1。

**正文块 3**

- **P.3.1** · *If $A=1$, the system retains $c_0$ and produces the…*
  A=1保留c0并产生joint text-plus-visual output。 **作用／依据：**early branch。
- **P.3.2** · *Otherwise, it performs concept search before the final joint…*
  A=0先concept search，再输出final joint prediction。 **作用／依据：**search branch；公式 P2。

**正文块 4**

- **P.4.1** · *The primary threshold is fixed at $\tau=0.7$, with 0.5…*
  primaryτ=0.7，0.5/0.9用于sensitivity。 **作用／依据：**预设超参。
- **P.4.2** · *No query annotation enters this decision.…*
  decision不使用query annotation。 **作用／依据：**可部署性。
- **P.4.3** · *Agreement measures consistency between the probe masks; its relationship…*
  agreement衡量probe consistency；与target accuracy的关系在Appendix形式化。 **作用／依据：**信号边界。

**正文块 5**

- **P.5.1** · *The second batch evaluates four connected parts.…*
  第二批包含四个相连部分。 **作用／依据：**evaluation roadmap。
- **P.5.2** · *A fixed-candidate-pool ablation compares ranking by $s_rs_q$, $s_r$, and…*
  fixed-pool ablation固定candidate evidence和visual prompts，比较s_rs_q、s_r、s_q。 **作用／依据：**isolating scoring。
- **P.5.3** · *A fresh-query comparison tests the original full system, the…*
  fresh-query比较full、single-concept、one-initial-candidate。 **作用／依据：**独立泛化复查。
- **P.5.4** · *Domain transfer uses SUIM underwater images and public ISIC…*
  SUIM和ISIC2018 Task1测试domain transfer。 **作用／依据：**新domain。
- **P.5.5** · *Finally, the agreement gate is compared with original full…*
  AGS对照original full、staged always-search、initial-score gate、random routing。 **作用／依据：**排除只要少search就会快的解释。

**正文块 6**

- **P.6.1** · *The core study requests 400 fresh COCO and 1,000…*
  core请求400fresh COCO、1,000fresh LVIS queries，每个最多3references。 **作用／依据：**预设样本量。
- **P.6.2** · *SUIM and ISIC each request 1,000 queries under the…*
  SUIM/ISIC各请求1,000queries，pool定义在Appendix。 **作用／依据：**预设transfer规模。
- **P.6.3** · *The initial-score threshold is 0.8; random policies search with…*
  initial-score threshold0.8，random search概率0.25/0.5/0.75。 **作用／依据：**controls设置。
- **P.6.4** · *These controls test whether agreement identifies useful opportunities to…*
  这些controls检验agreement是否识别适合停止的inputs，超越单纯减少search数量。 **作用／依据：**核心可证伪问题。

**正文块 7**

- **P.7.1** · *Evaluation will compare paired accuracy, latency distributions, reference sensitivity,…*
  评价paired accuracy、latency distribution、reference sensitivity、requests、startup、memory。 **作用／依据：**完整metric设计。
- **P.7.2** · *The second batch is underway; its performance results will…*
  第二批仍在运行；collect并verify后才能加入performance results。 **作用／依据：**明确结果状态。

### 5.2 内的 Cost–Accuracy Analysis

**正文块 1**

- **G.1.1** · *Let $A=1$ mean that the gate accepts the initial…*
  令A=1表示接受initial concept。 **作用／依据：**变量。
- **G.1.2** · *After a shared prefix of cost $C_p$, let $C_e$…*
  shared prefix cost为Cp；其后early/search branches余下cost分别Ce/Cs，均含final segmentation。 **作用／依据：**cost边界。
- **G.1.3** · *The gated and staged always-search costs satisfy…*
  定义gated与staged always-search的cost。 **作用／依据：**公式 P3。

**正文块 2**

- **G.2.1** · *Writing $D=C_s-C_e$ and denoting the corresponding final-mask IoUs by…*
  令D=Cs−Ce，Je/Js为相应final-mask IoUs，可得expected cost和accuracy差。 **作用／依据：**公式 P4。

**正文块 3**

- **G.3.1** · *The first identity measures computation avoided on accepted inputs;…*
  两个identity分别量化accepted inputs节省的compute，以及同一accepted subset的accuracy变化。 **作用／依据：**identity解释。
- **G.3.2** · *If $H=C_{\mathrm{st}}-C_{\mathrm{full}}$ is the staging overhead relative to original…*
  若H为staging相对original full的extra cost，则真正比original更快当且仅当E[AD]>E[H]。 **作用／依据：**速度成立条件。
- **G.3.3** · *Consequently, an acceptance rate alone does not establish a…*
  因此acceptance rate本身不能证明speedup，branch savings必须覆盖probe overhead。 **作用／依据：**排除常见误推。
- **G.3.4** · *The staged always-search and random-routing controls separate this overhead…*
  staged always-search与random routing分别识别overhead和agreement signal的价值。 **作用／依据：**由分析导出control必要性。

### 6. Discussion 与 7. Conclusion

**正文块 1**

- **D.1.1** · *The two systems offer different operating points.…*
  两系统提供不同operating points。 **作用／依据：**整体收束。
- **D.1.2** · *CG-ICS provides higher average accuracy and stronger small-target performance…*
  CG平均更准且small-target更强，INS更便宜。 **作用／依据：**主结果综合。
- **D.1.3** · *Reference sensitivity adds another dimension: CG-ICS improves worst-reference accuracy…*
  reference维度补充：CG两dataset worst-reference更高，但LVIS variation更大。 **作用／依据：**robustness综合。
- **D.1.4** · *Choosing a system therefore depends on the importance of…*
  系统选择需同时权衡accuracy、latency、resources和reference consistency。 **作用／依据：**实际含义。

**正文块 2**

- **D.2.1** · *The encoder matrix and within-system ablations explain why the…*
  encoder matrix与within-system ablation说明full-system ranking不能简化为backbone ranking。 **作用／依据：**归因结论。
- **D.2.2** · *DINOv3 leads the tested feature-processing matrix, yet CG-ICS achieves…*
  DINOv3在feature matrix领先，但CG通过language/prompting/decoding组合取得更高system accuracy。 **作用／依据：**解释表面矛盾。
- **D.2.3** · *INSID3 processing also improves every tested trunk.…*
  INS processing对每个trunk都有帮助。 **作用／依据：**补充证据。
- **D.2.4** · *The representation and the procedure that converts it into…*
  所以representation与mask construction procedure应共同评价。 **作用／依据：**原则。

**正文块 3**

- **D.3.1** · *The strongest motivation for selective search is the difference…*
  selective search动机来自average gain与input-specific gain的区别。 **作用／依据：**新方法逻辑核心。
- **D.3.2** · *With visual boxes, full search improves reference-averaged IoU for…*
  full search在194/400COCO与535/1,000LVIS queries改善，在196/403个下降。 **作用／依据：**query-level异质性。
- **D.3.3** · *A positive dataset mean therefore does not imply that…*
  positive dataset mean不意味着每query受益。 **作用／依据：**由计数得出的结论。
- **D.3.4** · *The observed search--prompt interaction further shows that the value…*
  search–prompt interaction进一步表明component价值取决于其他component是否已存在。 **作用／依据：**连接factorial证据。

**正文块 4**

- **D.4.1** · *These interactions describe the executed paired-input cohort.…*
  前述interactions描述执行过的full paired cohort。 **作用／依据：**适用样本。
- **D.4.2** · *Requiring all four configurations to share a recorded GPU…*
  要求四config同GPU后，仅剩每dataset100H200queries。 **作用／依据：**稳健性subset。
- **D.4.3** · *The interaction is $+0.66$ points on COCO (95% CI…*
  该subset interaction为COCO+0.66CI跨0，LVIS−3.47CI负。 **作用／依据：**分层结果。
- **D.4.4** · *The LVIS direction persists, while the COCO estimate is…*
  LVIS方向保留，COCO证据不确定。 **作用／依据：**谨慎但必要的结论。
- **D.4.5** · *These subsets also change the fold and input composition,…*
  subset同时改变fold/input composition，不能据此归因hardware effect。 **作用／依据：**避免伪因果。

**正文块 5**

- **D.5.1** · *The appropriate next test is whether an observable signal…*
  下一步检验observable signal能否在fresh inputs有效分配search。 **作用／依据：**从描述到预测。
- **D.5.2** · *The proposed gate makes this question explicit: saved branch…*
  gate将要求写明：saved branch compute覆盖probe cost，accepted inputs决定accuracy变化。 **作用／依据：**数学分析回扣。

**正文块 6**

- **D.6.1** · *The study combines system comparison, paired component analysis, and…*
  本研究结合system、paired component与pixel-level failure分析。 **作用／依据：**scope概括。
- **D.6.2** · *Reference dependence is handled through query-level aggregation and image-cluster…*
  query-level aggregation与image-cluster uncertainty处理reference dependence。 **作用／依据：**统计原则。
- **D.6.3** · *Broader generalization will be assessed through the separate transfer…*
  transfer泛化与foreground-preserving preprocessing是两条具体扩展。 **作用／依据：**未来工作。
- **D.6.4** · *The main remaining question is whether agreement can predict…*
  当前关键未决问题是agreement能否跨domain可靠预测low-value search，第二批专门检验。 **作用／依据：**明确knowledge gap。

**正文块 7**

- **D.7.1** · *Visual correspondence and concept guidance yield distinct accuracy--cost trade-offs…*
  两条路线形成不同accuracy–cost trade-offs。 **作用／依据：**Conclusion总句。
- **D.7.2** · *CG-ICS is more accurate on the two initial benchmarks,…*
  CG在初始benchmark更准，INS更快。 **作用／依据：**结论1。
- **D.7.3** · *Component analysis attributes useful gains to feature processing, semantic…*
  components显示feature processing、concept、visual prompts贡献，并揭示dataset-dependent search收益。 **作用／依据：**结论2。
- **D.7.4** · *Reference sensitivity and small-target failures provide further distinctions beyond…*
  reference sensitivity与small-target failures补充mean accuracy之外的区别。 **作用／依据：**结论3。
- **D.7.5** · *These findings motivate AGS, whose formulation makes the required…*
  这些证据motivate AGS；formulation写清必要compute savings与accuracy trade-off。 **作用／依据：**proposal贡献。
- **D.7.6** · *Its ongoing controlled evaluation will determine whether the observed…*
  ongoing controlled evaluation将决定它是否成为实际有用的adaptive system。 **作用／依据：**不提前宣称成功。

### Appendix A. Protocol Details / B. Additional Results

**正文块 1**

- **A.1.1** · *Table records the first-batch scope.…*
  Table6记录第一批研究范围。 **作用／依据：**matrix导航。
- **A.1.2** · *Task counts resolve the shared controls used by seed…*
  task count解释seed/sweep复用controls，避免重复计数。 **作用／依据：**覆盖口径。

**正文块 2**

- **A.2.1** · *The first batch has 1,480 successful formal tasks and…*
  第一批1,480successful formal tasks＋4failed reference tasks。 **作用／依据：**执行覆盖。
- **A.2.2** · *All 615,976 expected records, including four errors, are accounted…*
  615,976expected records含4errors均已匹配，并核对532outputbindings。 **作用／依据：**审计完整性。
- **A.2.3** · *The standard cohorts contain 3,975 COCO and 13,750 LVIS…*
  standard cohorts含3,975/13,750 distinct targets，来自3,820/9,051 distinct image hashes。 **作用／依据：**区分episode/query/image。
- **A.2.4** · *COCO and LVIS share 462 standard-cohort image hashes, so…*
  COCO/LVIS共享462standard image hashes，所以dataset分别报告。 **作用／依据：**避免误当独立复现。

**正文块 3**

- **A.3.1** · *Common manifests and original-resolution ground truth preserve source class…*
  common manifests＋original GT保留class memberships，但sampling/aggregation可能不同于author protocol。 **作用／依据：**复现范围。
- **A.3.2** · *Empty LVIS source masks are excluded before sampling; all…*
  sampling前排除empty LVIS masks，仍保留all classes/folds。 **作用／依据：**数据有效性。
- **A.3.3** · *Pretraining-data overlap is not controlled, and intervals are not…*
  未控制pretraining overlap，也未做multiple-comparison adjustment。 **作用／依据：**剩余局限。

**正文块 4**

- **A.4.1** · *The executed INSID3 implementation estimates its positional basis from…*
  执行版INS positional basis用ImageNet-normalized all-zero image；论文描述Gaussian noise。 **作用／依据：**实际实现差异。
- **A.4.2** · *It retains original features for clustering and self-similarity and…*
  original features用于clustering/self-similarity，debiased用于cross-image matching。 **作用／依据：**分支实现。
- **A.4.3** · *The main route uses bilinear restoration without CRF refinement.…*
  mainroute为bilinear restoration，无CRF refinement。 **作用／依据：**后处理。
- **A.4.4** · *In CG-ICS, working images are size 512 before SAM…*
  CG working size512，之后SAM internal preprocessing，nearest restoration回原尺寸。 **作用／依据：**预处理/恢复。
- **A.4.5** · *The direct-concatenation control uses the implemented query-side instance union.…*
  direct concatenation control采用实现中的query-side instance union。 **作用／依据：**control定义。

**正文块 5**

- **A.5.1** · *Fresh core queries exclude previous formal-query image hashes and…*
  fresh core queries排除以往formal query image hashes以及跨初始dataset的canonical COCO IDs。 **作用／依据：**freshness规则。
- **A.5.2** · *The transfer scopes use the author-prepared merged SUIM pool…*
  transfer使用author-prepared merged SUIM pool和reorganized ISIC2018 Task1 training images。 **作用／依据：**数据来源。
- **A.5.3** · *SUIM's full source pool is checked before sampling, and…*
  SUIM sampling前检查full source pool，剔除image/mask尺寸不一致pairs而不改原raster。 **作用／依据：**已修复data validation。
- **A.5.4** · *All seven foreground classes and 1,000 unique queries are…*
  要求7foreground classes和1,000unique queries；eligible counts随结果报告。 **作用／依据：**计划覆盖。
- **A.5.5** · *ISIC reference selection uses the supplied diagnostic strata; it…*
  ISIC references用supplied diagnostic strata；不是patient-disjoint clinical test或official challenge-test。 **作用／依据：**transfer解释边界。

**正文块 6**

- **A.6.1** · *The fixed-pool scoring study generates candidates once per input,…*
  fixed-pool每input只生成一次candidates，各rankingrule用同一evidence。 **作用／依据：**公平scoring比较。
- **A.6.2** · *If the maximum combined score in the pool is…*
  pool最大combined score<1e−6时，各rule共享visual fallback。 **作用／依据：**特殊情况一致性。
- **A.6.3** · *Random search policies use fixed probabilities rather than probabilities…*
  random routing使用预设probabilities，不依据gate已观测search rate选概率。 **作用／依据：**避免事后选control。
- **A.6.4** · *Staged always-search measures the extra initial probing cost.…*
  staged always-search量化initial probing的extra cost。 **作用／依据：**必要control。

**正文块 7**

- **A.7.1** · *Owned memory sums simultaneous samples from the runner and…*
  owned memory在同一采样时刻汇总runner与其MLLM processes。 **作用／依据：**memory定义。
- **A.7.2** · *The language server's 0.85 utilization setting includes reserved/cache memory;…*
  MLLM0.85utilization含reservation/cache，故数值不是minimum model memory，也不是runner-only CUDA peak。 **作用／依据：**防止误读memory。

### Appendix C. Mathematical Analysis

**正文块 1**

- **C.1.1** · *For nonempty binary masks viewed as finite pixel sets,…*
  把非空binary masks看作pixel sets，定义J为IoU，d=1−J为Jaccard distance。 **作用／依据：**数学对象。
- **C.1.2** · *The reverse triangle inequality gives…*
  用reverse triangle inequality约束两probe对GT的IoU差。 **作用／依据：**公式 A1。

**正文块 2**

- **C.2.1** · *Indeed, applying the triangle inequality in both directions bounds…*
  证明思路：双向用triangle inequality得到distance difference上界。 **作用／依据：**推导。
- **C.2.2** · *High agreement therefore bounds the difference between the two…*
  high agreement只限制两probe accuracies之间的差，不保证它们的absolute accuracy。 **作用／依据：**准确解释。
- **C.2.3** · *For example, if both probes predict the same region…*
  反例：两probe完全一样却都选在GT外，则mutual IoU=1、target IoUs=0。 **作用／依据：**反驳错误充分条件。
- **C.2.4** · *The bound concerns the two probe masks, not the…*
  该bound仅约束probe masks，不直接约束后续joint-prompt output。 **作用／依据：**定理适用范围。

**正文块 3**

- **C.3.1** · *Within one class, let $U_e=\mathrm{TP}_e+\mathrm{FP}_e+\mathrm{FN}_e$ be episode union size…*
  在每个class内定义union size Ue及episode IoU Je。 **作用／依据：**加权变量。
- **C.3.2** · *With population covariance over episodes,…*
  使用episode population covariance展开pooled IoU与episode mean的差。 **作用／依据：**公式 A2。

**正文块 4**

- **C.4.1** · *where $G_e=\mathrm{TP}_e+\mathrm{FN}_e$ is target area.…*
  G表示GT target area，因此U=G+FP。 **作用／依据：**第二层分解变量。
- **C.4.2** · *The identity follows from $\mathrm{TP}_e=U_eJ_e$ and $\overline{UJ}=\overline{U}\,\overline{J}+\operatorname{Cov}(U,J)$.…*
  由TP=UJ及mean product=covariance+product of means推出identity。 **作用／依据：**推导。
- **C.4.3** · *It explains why a large erroneous foreground region can…*
  large false-positive region既降低IoU又增大union weight，所以pooled score可能受更大影响。 **作用／依据：**数学解释与经验现象连接。

**正文块 5**

- **C.5.1** · *For CG-ICS on LVIS, the mean episode IoU is…*
  LVIS CG episode mean55.19%，class-pooled47.44%。 **作用／依据：**待解释观测。
- **C.5.2** · *Averaging Eq. using the same class/fold weights yields a…*
  按相同class/fold weights平均identity得到−7.75pp，分为target-area+3.82与FP−11.57pp。 **作用／依据：**经验数值代入identity；已核对920类各25episodes，因此这里global episode mean与class/fold-weighted mean一致。
- **C.5.3** · *These are exact weighting components, not estimates of causal…*
  这两个是exact weighting components，不是causal effect estimates。 **作用／依据：**数学结论边界。

## 4. 公式、图和表分别承担什么论证任务

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
