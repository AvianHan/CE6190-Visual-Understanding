# CE6190 ICS — 持续研究与实验记录

最后更新：**2026-11-01（Asia/Singapore；UTC 2026-10-31）**。

这是当前项目的维护文档。每次修改记录：研究问题、输入输出、变化原因、运行证据、结论边界和下一步。正式实验还未开始，本文没有自有模型准确率。

## 1. 已确认的方向与工作方式

- 共同任务：reference image + reference mask + query image → query mask。先做 semantic；part／personalized 独立设计协议。
- 核心：[INSID3](https://arxiv.org/abs/2603.28480) 与 [CG-ICS](https://arxiv.org/abs/2606.28149)。代码分别为 [visinf/INSID3](https://github.com/visinf/INSID3)、[Kakarot1103/CG-ICS](https://github.com/Kakarot1103/CG-ICS)。
- 项目：[AvianHan/CE6190-Visual-Understanding](https://github.com/AvianHan/CE6190-Visual-Understanding)。不重选题；不预设最终方法、数据集或项目规模上限。
- 用户已同意先跑两种 anchor、组件／backbone／参数与 reference 分析，再根据结果扩展 baselines／datasets。
- 助手完成代码；用户下载更新已有本地 clone，自行在 VS Code Source Control commit／push，再 SSH pull 运行。本轮没有 commit／push。
- 讲解从用户真实理解出发：先任务／整体结构，再原图中的模块；新概念先用狗的例子。代码推进与学习相连，不默认已掌握之前出现的术语。

## 2. 当前状态及证据

| 项目 | 状态 | 依据／限制 | 下一步 |
|---|---|---|---|
| 2026-10-22 starter、研究 HTML、原始 evaluator | 已找回并继承 | 旧包和 `research/`；保留旧 README | 使用新记录区分历史状态 |
| 用户本地 VS Code clone 的后续修改 | 尚待核实 | 当前不能读取该工作树；远端无凭据读取未成功 | 本地 updater 备份后查看 diff |
| 固定 manifest、query GT 隔离、三种 IoU | 已实现并 CPU 验证 | `ics_eval.py`、contract tests；原 API 保留 | 实际数据 smoke |
| COCO／LVIS 和第二批数据 loaders | 已实现并合成验证 | `ics/data.py`、13 项数据测试 | 实际下载、split／图像抽查 |
| INSID3 adapter、组件／encoder 接口 | 已实现并合成验证 | 11 项测试，含真实 upstream 处理代码＋合成 encoder 等价检查 | 预训练权重 GPU 推理 |
| CG-ICS adapter、概念／visual 组件 | 已实现并合成验证 | 12 项 tests；无 SAM3／Qwen 权重 | GPU import、模型 smoke |
| runner、resume、失败重试审计、分析 | 已实现并合成验证 | runner／analysis tests；失败 attempt 不覆盖丢失 | 真实结果检查 |
| 批量调度、资源监测、诊断 | 已实现并合成验证 | 调度 mock tests；真实 Slurm 未执行 | 实际 account／QOS／配额与 GPU 探测 |
| 三个 GPU 环境、模型和数据下载 | 提供代码，实际资源尚未就绪 | bootstrap dry plan；本地 CPU 环境不能代替服务器 | 按已有资源复用和安装 |
| 作者原始随机 episode／resize-GT 协议的完整复现 | 尚未完成 | 本包默认统一 manifest＋原图 GT 协议 | 另建明确的 author-protocol run |
| 正式模型成绩、跨域结论、最佳组合 | 尚无结果 | 没有 GPU 模型运行日志 | 不能提前下结论 |

最终验证：72 项 CPU／合成测试全部通过、零跳过，36 个配置校验及三套 suite dry plan 通过，更新器备份／保留 Git／幂等行为通过。详细数量、命令和状态见 `research/validation_2026-11-01.json`。合成测试证明接口和受测逻辑可用，不证明真实权重、CUDA 或所有官方依赖可运行。

## 3. 怎样拆组件，才能回答问题

“寻找最有效组合”可以做，但先定义**有效**：准确率、reference 稳定性、跨域表现和实际成本的共同表现。报告 Pareto trade-off，避免只凭某个小 subset 的最高 IoU 宣称普遍最优。

| 层级 | 固定什么 | 改变什么 | 能支持的结论 |
|---|---|---|---|
| 完整系统比较 | 同一 reference/query、同一 GT 与指标 | INSID3 vs CG-ICS 整套系统 | 在指定资源／数据条件下哪个系统表现如何；不能单独归因 backbone |
| INSID3 组件组合 | DINOv3、输入、其余阈值 | debias on/off × self+cross／cross／seed aggregation | debias 和区域聚合的贡献／交互 |
| CG-ICS 组件组合 | SAM3、MLLM checkpoint、输入、采样规则 | single concept／candidate scoring／search × visual prompt on/off | 概念选择和视觉提示的贡献／交互 |
| Encoder × processing | 相同 episode、公共 1008 输入尺寸、记录特征层／维度 | DINOv3／DINOv2／SAM3 trunk × simple similarity／INSID3 processing | 不同 encoder 下同一处理增益；仍须披露上游训练差异 |
| SAM3 共同来源的系统对照 | SAM3 checkpoint 和 episode | SAM3 features+INSID3 vs SAM3 decoder+CG-ICS 等 | 共用来源的视觉模型时的系统差异；不是“只换一个组件” |

DINOv3 的 dense features 不能直接替代 SAM3 的 text／visual prompt 接口和 decoder；CG-ICS 的概念词也不能直接作为 INSID3 dense features 的输入。暂未实现无明确接口定义的跨方法任意拼接 hybrid。

具体例子：同一张 reference dog 和 query dog，用完整 INSID3 得到 mask A，关闭 debias 得到 mask B，其他条件保持一致。这回答 debias 在该流程中的作用。换成完整 CG-ICS 得到 mask C，同时改变了视觉模型、概念生成和 decoder，A/C 只能先解释为系统比较。

文献边界：INSID3 已有固定 encoder 的处理／debias 消融和 encoder 对照；CG-ICS 已有部分 text-only 及概念组件分析。因此不声称原文完全没研究组件。INSID3 supplementary C.3/Table 9 已比较 SAM3 拼接 baseline；复跑它不叫新的未比较方法。我们的新增价值需由完整组合、统一配对、跨域、稳定性及成本分析的实际证据支持；“某篇未做”不等于“领域首次”。

## 4. 已同意的第一批实验 E0–E7

| 编号 | 研究问题／输入输出 | 当前配置／实现 | 运行后能说明什么，不能说明什么 |
|---|---|---|---|
| E0 接口与协议验证 | 少量固定 episodes → mask、尺寸、日志、资源 | 每配置 smoke；pilot suite；源代码核对 | 检查真实依赖和可运行性；不能用少量 smoke 报主结果。作者原始协议完整复现另列。 |
| E1 完整方法比较 | COCO-20i／LVIS-92i 固定配对 → 三种明确 IoU | INSID3 full、CG-ICS full | 完整系统差异；不能单独归因 DINOv3／SAM3。 |
| E2 reference／运行稳定性 | 同一 query 换不同 references；同一输入换随机 seed | reference cohort；CG component-full seeds 0/1/2 | 分开 reference 选择波动与 MLLM 随机波动；重复 query 不算独立样本。 |
| E3 INSID3 组件 | 固定 DINOv3 → 各组件组合 mask | debias 2 × aggregation 3；另 simple similarity | 组件增益与交互；只对当前阈值和数据成立。 |
| E4 CG-ICS 组件 | 固定 SAM3／MLLM → 各提示／概念组合 mask | concept 3 × visual 2；另 visual-only、concat | 搜索、概念评分、视觉提示的价值与成本；不是所有组合均为作者原始配置。 |
| E5 表示与处理 | 三种冻结 encoder → simple similarity 或 INSID3 mask | DINOv3／DINOv2／SAM3 trunk × 2 | 各表示下处理增益；不等于控制了 encoder 的预训练数据。 |
| E6 参数／成本 | 同一 component cohort → accuracy／调用／耗时 | INSID3 tau、SVD rank、输入尺寸；CG BON、search loops | 敏感性与成本；一次单因素 sweep 不保证全局最优。 |
| E7 成功失败与跨域分析 | 已保存 prediction／GT／telemetry → CSV 和案例 HTML | 案例板、目标面积分组、reference spread、失败记录 | 描述小目标等现象；遮挡／边界归因需可靠标注和进一步检验。 |

现有分析尚未实现 bootstrap CI、boundary metric、自动跨 folds 总分或严格遮挡分层。所有结果按 dataset／fold／track／manifest／配置保留；后续根据研究问题扩充，不混成一个总排行榜。

首批配置：**36 个版本**，不是 36 个独立方法。包括 42 个 manifests、36 个 smoke tasks、1,484 个 production tasks；含 preflight 和 collector 为 1,522 个 Slurm task elements。请求 episode 上限估计 896,800；references 不足时实际数量下降。这是初始可调整执行计划，**不是已经完成的实验数，也不是未来上限**。如此规模下必须根据真实吞吐／时限／配额调整分片与并行；可用 pilot 得到依据。

- standard：COCO 每 fold 请求 1000 queries；LVIS 每 fold 2300；1-shot。
- reference：每 fold 请求 500 queries × 最多 50 distinct references。
- component：每 fold 请求 100 queries × 最多 3 references；完整控制版本和对应消融共用此 cohort。
- 同类参数／组件对照使用相同 manifest。修改 recipe 必须生成新 manifest 和 run directory，不能复用旧结果。
- 组件 CG 配置的 MLLM seed 跟随 episode seed；三个重复版本除 experiment ID／seed 外元数据一致。主比较固定 MLLM seed 也是相对 upstream 未指定 seed 的项目改动；seed 不保证服务端 bitwise deterministic。

## 5. 第二批 datasets：已有独立 suite，数据待准备

| Dataset | 选择理由 | 明确协议与待核实事项 |
|---|---|---|
| FSS-1000 | 类别广，测试 reference 概念是否可泛化 | 官方 test 类别列表；每类通常仅 10 图，固定 query 最多 9 个不同 references。下载／布局待核实。 |
| SUIM | 水下域、颜色和形态变化 | 当前兼容作者 merged 数据与 7 foreground classes；不能称官方 110-image test 协议。 |
| ISIC2018 | 医学域、低对比病灶 | 作者重组的训练图像；目录 1/2/3 在实际 loader 中是类别，不能直接称 held-out folds。需核对重组映射。 |
| iSAID-5i | 航拍、小目标、密集目标 | 3 folds；reference/query 必须来自不同原始航空图，避免不同 tiles 的同图泄漏。 |
| PASCAL-5i，可选 | 标准自然图像 few-shot 补充 | 独立 split；不是 PASCAL-Part；suite 中默认关闭。 |

`configs/suite.phase2.json` 与首批独立，第二批未准备好不会阻止 COCO／LVIS。当前为 8 个方法配置、12 个启用 manifests、84 个 production tasks，请求 episode 上限 25,600；同样只是可调整的计划。

之后继续根据结果扩展领域与方法。PASCAL-Part、PACO-Part、PerMIS 等需各自明确 part／personalized 语义，不能直接并入 semantic 表。旧 baseline 大表保留为研究线索，其每篇全部 datasets 的最终逐项核对不因本轮代码完成而自动视为完成。

## 6. 公平评测和实际成本

- manifest 锁定 image／mask 路径、SHA256、fold、concept、reference/query 配对与 mask 语义。concept 名和 query GT 仅用于评估，不传给 adapter。
- COCO 的作者 prepared 255 按两 anchor loader 映射背景；PASCAL／iSAID query 的 255 忽略；reference 的非目标像素映射背景。LVIS union 同一类别的所有实例。
- 三种总体指标：mean episode IoU、class mean episode IoU、macro class pooled IoU；同时保存原始 TP/FP/FN/TN 和 ignored。空 union 未定义并显式计数。
- 固定配对＋原图 GT 是统一协议；作者 resize-GT／随机采样协议另列。不能因调用原模型源码就称完全复现论文数值。
- repeated reference 以 query image hash＋concept 分组。抽样中重复 query／相同原图 tile 不增加独立样本量。
- 每 episode 保存 prediction、模型和源码版本、配置和 manifest 哈希、实际调用／耗时／失败。resume 校验这些身份，不用变化后的配置解释旧结果。
- 最新 attempt 用于 accuracy；旧失败 attempt 保留 traceback 和实际成本，累计成本不因重试成功而消失。进程突然中断、未提交的 episode 成本无法完全归属，job wall-time 另保留。
- SAM 顶层调用、image encoder、prompt/head 调用分别计数；MLLM requests、tokens、搜索评分次数来自实际执行点，未返回 usage 则为 null。
- PyTorch peak 仅覆盖模型进程；vLLM 独立进程显存按约 1 秒采样，报告作用域、每 GPU 和同一采样时刻总量。不同时间峰值不能相加叫瞬时峰值。

## 7. 论文／源码／本项目适配差异

| 位置 | 已核对事实与影响 |
|---|---|
| INSID3 debias | pinned source 实际用 normalized black image＋centered SVD；论文描述 Gaussian noise。adapter 保留 source 行为，未静默“修正”。 |
| INSID3 区域 | original features 做 target clustering／self similarity；debiased features 做跨图 matching。candidate gate 有 forward AND backward；聚合包含 area weighting，seed 不一定强制保留最终结果。 |
| INSID3 encoder matrix | SAM3 仅提取 trunk features，未使用 text／decoder；其输入尺寸、层和训练来源必须披露。公共尺寸 1008 为兼容 14／16 patch strides 的选择。 |
| CG-ICS 输入 | source 使用 512² working image，再进入 SAM3；MLLM 第二视图为 cropped highlight。输出恢复原图尺寸。 |
| CG-ICS 搜索 | 保留 pinned TreeSearcher 的实际停止行为、候选评分与最终 fallback；不能只看论文示意图推断调用次数。 |
| CG-ICS namespace | 隔离加载 vendored SAM3／源码 imports，避免覆盖两个方法环境；源码 pin／hash 校验，不在线修补原算法。 |
| 统一评估 | adapter 输出原始 query 分辨率；不沿用作者 CLI 缩放 GT 后的评分。 |
| 安装规格 | Python3.12＋独立 torch／vLLM 环境；numpy1.26.4 等显式兼容设置。CPU 环境完整 CG import 停于 triton 缺失；GPU 依赖仍需真实安装验证。 |

详细解释见 `docs/INSID3_ADAPTER.md`、`docs/CGICS_ADAPTER.md`、`docs/DATA_PROTOCOLS.md`。

## 8. 来源锁定和已有代码的改动理由

| Source | Commit |
|---|---|
| INSID3 | `0c165a10cf52ab91f335883d06260de86854adbe` |
| CG-ICS | `8da845743f15c18ec490567fbd86dba58c8a388f` |
| DINOv3 | `6876159a11b4df116f30f667f8c9888617df0751` |
| DINOv2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |

模型源 revision 在 `configs/source_pins.json`；运行时另记录实际文件 SHA256。请求的 revision 不代替实际文件身份。

旧 `ics_eval.py` 的 schema／指标 API 保留，仅增加单次验证调用内的文件哈希复用，避免大 reference cohort 反复读取相同文件；缓存键含路径／大小／mtime／ctime，跨运行不复用。每个预期 hash 仍需逐条匹配。

数据生成缓存现在重新解码并与源标注映射逐像素核对；损坏缓存会拒绝继续，不会被新 hash “认可”。HTTP 下载在完成前核对 Content-Length；截断文件保留为 partial。

所有新代码增量接到旧 evaluator。更新包校验 SHA256、备份将替换的文件、保留其他文件和 Git 状态，不自动 commit／push。

## 9. 执行顺序与当前阻塞

1. 下载更新包，用 updater 应用到本地 clone；检查 diff 后由用户 commit／push。
2. SSH pull 后运行 `scripts/probe_cluster.py`；确认真实 GPU／显存、Python3.12、驱动、联网、存储、Slurm account／partition／QOS／最大 walltime／array 和 pending-task 配额。
3. 复用已有资源；通过 bootstrap 固定来源／环境，获得官方授权 DINOv3 `.pth`、SAM3 与 Qwen 权重。准备 COCO／LVIS 数据。
4. 填本地 cluster 配置；dry plan 审核工作量；pilot 测吞吐／显存。也可提交带自动 smoke gate 的完整 suite；只有通过对应 smoke 的配置才执行正式部分。
5. 查看 failure／coverage、mask 和资源；处理实际依赖／协议问题后续跑。保存诊断包及结果概要。
6. 根据自己的结果决定第二批扩展、额外方法和更深入组件实验。论证应明确区分事实、假设、计划与结果。

当前阻塞不是用户未批准实验：实验已获同意。缺的是实际服务器信息、权重授权／文件、真实数据与 GPU 运行证据。不能用另一个课程项目的集群设置替代。

## 10. 课程要求与更新记录

已确认课程要求：Visual Understanding Individual Project 占 50%；截止 2026-11-17；至少 1 个方法、2 个 datasets、自己的实验，含参数分析／关键消融／优缺点／成功失败案例；report 至少 3 页，introduction／method／experiments and analysis／conclusion；提交 report PDF＋source-code ZIP，不含 datasets／models，云链接不代替提交；与 Group Visual Generation 避免方法重叠。

本次包未重新获取《0. coursework 2026.pdf》，以上继承此前已核准内容；最终提交前仍以课程原件为准，不把最低要求当项目上限。

| 日期 | 更新 | 证据与边界 |
|---|---|---|
| 2026-10-22 | evaluator／manifest／环境探测及文献源码审计 | 历史 HTML／starter；无模型成绩。 |
| 2026-11-01 | 继承旧代码，补齐 adapters、数据层、runner、分析、组件配置、并行调度、bootstrap、更新器、维护文档 | CPU／合成测试与 dry plan；未下载实际 datasets／权重，未执行 GPU 推理。 |

后续每轮追加：实际命令／run directory、配置与 manifest hash、成功／失败数量、实验观察、仍不能确定的结论、下一轮修改。不要将“会做”或“已写配置”改写为“已完成实验”。
