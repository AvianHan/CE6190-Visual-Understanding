# CE6190 ICS — 持续研究与实验记录

最后更新：**2026-11-01**。服务器状态以用户最近回传的终端输出为准，并非实时监控。

这是当前项目的维护文档。每次修改记录：研究问题、输入输出、变化原因、运行证据、结论边界和下一步。**四份模型权重已下载，三个环境的基础检查已有成功证据；真实数据、GPU 推理和正式 ICS 实验仍未验证，尚无新的 ICS Slurm 作业 ID 或自有模型准确率。**

维护记录纠正：仓库原本已有本文，包含研究设计与历史验证；最近的服务器配置／下载步骤没有同步写回。此前说“没有维护文档”不准确，应为“没有及时更新已有文档”。本次将终端证据补回本文，保留历史研究范围和结论边界，后续继续维护同一份记录。

## 1. 已确认的方向与工作方式

- 共同任务：reference image + reference mask + query image → query mask。先做 semantic；part／personalized 独立设计协议。
- 核心：[INSID3](https://arxiv.org/abs/2603.28480) 与 [CG-ICS](https://arxiv.org/abs/2606.28149)。代码分别为 [visinf/INSID3](https://github.com/visinf/INSID3)、[Kakarot1103/CG-ICS](https://github.com/Kakarot1103/CG-ICS)。
- 项目：[AvianHan/CE6190-Visual-Understanding](https://github.com/AvianHan/CE6190-Visual-Understanding)。不重选题；不预设最终方法、数据集或项目规模上限。
- 用户已同意先跑两种 anchor、组件／backbone／参数与 reference 分析，再根据结果扩展 baselines／datasets。
- 助手完成代码；用户下载更新已有本地 clone，自行在 VS Code Source Control commit／push，再 SSH pull 运行。本轮没有 commit／push。
- 用户最新要求：把必要任务全部纳入持久化队列，一次启动后按数据准备 → preflight → GPU smoke → 正式分片 → 汇总自动推进。已获授权，不再逐阶段要求确认；共享 Slurm 提交上限通过滚动提交处理。
- 不取消或修改 Visual Generation 作业；ICS 与 Generation 的状态、输出和研究结论分别记录。当前会话没有 Orion SSH 通道，实际服务器启动由用户执行交付命令。
- 公司服务器凭据使用临时方式；不把 HF token、带签名下载链接或授权邮件全文写入文档、Git、脚本或日志。不重新安装／下载已经成功的资源。
- 讲解从用户真实理解出发：先任务／整体结构，再原图中的模块；新概念先用狗的例子。代码推进与学习相连，不默认已掌握之前出现的术语。

## 2. 当前状态及证据

| 项目 | 状态 | 依据／限制 | 下一步 |
|---|---|---|---|
| 2026-10-22 starter、研究 HTML、原始 evaluator | 已找回并继承 | 旧包和 `research/`；保留旧 README | 使用新记录区分历史状态 |
| 远端仓库与用户本地修改 | 远端完整源码已取得；用户本地工作树仍未检查 | Git clone 成功，基线 HEAD `aabdbb07a25cf11cf47b95797f9d6e621cb649f6`；不代表用户本地或 Orion HEAD 已核实 | 用户应用更新前备份并检查 diff |
| 固定 manifest、query GT 隔离、三种 IoU | 已实现并 CPU 验证 | `ics_eval.py`、contract tests；原 API 保留 | 实际数据 smoke |
| COCO／LVIS 和第二批数据 loaders | 已实现并合成验证 | `ics/data.py`、13 项数据测试 | 实际下载、split／图像抽查 |
| INSID3 adapter、组件／encoder 接口 | 已实现并合成验证 | 11 项测试，含真实 upstream 处理代码＋合成 encoder 等价检查 | 预训练权重 GPU 推理 |
| CG-ICS adapter、概念／visual 组件 | 已实现并合成验证；服务器完整导入通过 | 历史 12 项 tests；SAM3／Qwen 权重已下载，`CG-ICS imports OK; model not loaded` | GPU 模型 smoke |
| runner、resume、失败重试审计、分析 | 已实现并合成验证 | runner／analysis tests；失败 attempt 不覆盖丢失 | 真实结果检查 |
| 批量调度、资源监测、诊断 | 滚动入口、冻结源码和后台恢复已完成本地合成验证 | 静态首批 1,522 元素，滚动含数据任务共 1,523；共享 `MaxSubmitJobsPU=60`；未提交 ICS 作业 | 在 Orion 部署启动、核对实际 job IDs |
| 三个环境与四份模型权重 | 基础检查／下载已通过 | 见下面实际版本、哈希和修复记录；尚无 GPU 模型加载证据 | 复用现有资源，在 Slurm 计算节点 smoke |
| 实际数据与 manifests | 尚待确认 | 未取得服务器数据目录、split／图像检查或 manifest 生成结果 | 纳入自动 CPU 准备及 preflight |
| 作者原始随机 episode／resize-GT 协议的完整复现 | 尚未完成 | 本包默认统一 manifest＋原图 GT 协议 | 另建明确的 author-protocol run |
| 正式模型成绩、跨域结论、最佳组合 | 尚无结果 | 没有 GPU 模型运行日志 | 不能提前下结论 |

历史代码包验证：72 项 CPU／合成测试全部通过、零跳过，36 个配置校验及三套 suite dry plan 通过，更新器备份／保留 Git／幂等行为通过。完整报告现已通过仓库取得，见 `research/validation_2026-11-01.json`（UTC `2026-10-31T18:43:26.264912+00:00`）。这是此前版本的报告，不是本轮新增改动已通过测试的声明。合成测试证明接口和受测逻辑可用，不证明真实权重、CUDA 或所有官方依赖可运行。

本轮验证（UTC `2026-11-01T05:13:41.794368+00:00`）：**140 项运行，132 项通过，8 项因本地缺少 torch／torchvision 的可选张量测试跳过，0 failures／errors**；报告见 [research/validation_2026-11-01_rolling.json](research/validation_2026-11-01_rolling.json)。其中 39 项 rolling、14 项 CLI 测试包含真实后台子进程／继承 flock／重复 start，在模拟 Slurm 共享 60 个名额已满时登记 1,523 个待办，以及冻结恢复、提交回执不明的查证、失败分支隔离和数据下载完整性。这里的调度器是模拟程序，没有真实 Orion 提交、实际数据下载或 GPU 推理；不能写成“140 项全部通过”。

### 2.1 服务器路径与环境

| 项目 | 已确认值 |
|---|---|
| 登录节点提示 | `hansy@orion-login01` |
| ICS 仓库 | `/scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding` |
| `ICS_RUNTIME` | `/scratch/prj0000000262_1/hansy/ce6190-ics-runtime` |
| 环境／源码 | `$ICS_RUNTIME/envs/{insid3,cgics,mllm}`；`$ICS_RUNTIME/sources/{INSID3,CG-ICS}` |
| 权重／日志／缓存 | `$ICS_RUNTIME/weights`、`logs`、`cache/huggingface`、`cache/pip` |
| Slurm account／QoS | `prj0000000262`／`highpriority`；account 没有路径中的 `_1` 后缀 |

共同 Python：`3.12.15`，pip 路径均属于对应环境。

| 环境 | 已有服务器证据 | 尚未验证 |
|---|---|---|
| insid3 | `pip check` 通过；`torch 2.7.1+cu126`、`torchvision 0.22.1+cu126` 导入成功 | 本次终端记录未直接确认完整上游 INSID3 导入；GPU 模型加载／推理 |
| cgics | `torch 2.9.0+cu128`、`torchvision 0.24.0+cu128` 导入；decord 元数据修复后 `pip check` 通过；`CG-ICS imports OK; model not loaded`，bootstrap `errors: []` | 真实图像／视频解码、GPU 模型推理 |
| mllm | 最初缺少 vLLM；限定 mllm 重跑原 bootstrap 后 `vllm 0.11.2`、`transformers 4.57.3` 导入及 `pip check` 通过；`qwen-vl-utils 0.0.14`、`huggingface_hub 0.36.0` | vLLM 服务启动、Qwen GPU 推理 |

环境问题与修复边界：

- 初始 `bootstrap_status.json` 缺失，安装日志停于 `Installing collected packages`，当时不能宣告安装完成。`pip check` 通过也不证明预期软件齐全，mllm 缺少 vLLM 就是实际例子。
- decord 0.6.0 可导入，但内部 WHEEL 标签为 `cp36-cp36m-manylinux2010_x86_64`。已按核对的发行包元数据问题改为 `py3-none-manylinux2010_x86_64`，同步更新 RECORD 中 WHEEL 的 SHA-256／大小，未改二进制；备份在 `$ICS_RUNTIME/repairs/decord-0.6.0-5iw2of1l`。第二次显示 already corrected 后中断重复检查，不撤销首次成功。
- 登录节点 CUDA 不可用不等于 GPU 环境坏了。模型必须在实际 Slurm GPU allocation 内验证。保留成功环境，不无故 `--refresh-envs` 或升级 torch／vLLM。

### 2.2 四份模型权重

下表路径相对 `$ICS_RUNTIME/weights`。`available` 与文件哈希不等于 GPU 推理成功。

| 模型 | 文件／目录 | 固定来源 | 实际下载证据 |
|---|---|---|---|
| DINOv3 ViT-L/16 | `dinov3_L.pth` | 授权邮件的 `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`，LVD-1689M | 1,213,050,671 bytes；`available` |
| DINOv2 ViT-L/14 | `dinov2_L.pth` | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth` | 1,217,586,395 bytes；`available` |
| Qwen3-VL-4B-Instruct | `Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Instruct`；revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 12 files，100%，`available`；目录大小未单独测量 |
| SAM3 | `sam3.pt` | `facebook/sam3`；revision `3c879f39826c281e95690f02c7821c4de09afae7` | 授权 `ACCEPTED`；3,450,062,241 bytes；`available` |

用户回传的 SHA-256：

```text
dinov3_L.pth  8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035
dinov2_L.pth  d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
sam3.pt      9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

DINOv3 前八位与官方文件名一致，尚未取得官方完整校验值逐位独立核对；Qwen 未生成单一整目录哈希。DINOv3 首次链接前缀校验失败在下载前发生，第二次成功；SAM3 曾 pending，后来获准并下载，旧 pending 不是当前阻塞。

### 2.3 授权和 bootstrap 状态

- DINOv2／Qwen 匿名下载时禁用隐式 token。SAM3 使用获准账号的专用 token，`read -r -s` 从 `/dev/tty` 隐藏输入，仅在括号子进程导出 `HF_TOKEN`，并设置 `HF_HUB_DISABLE_IMPLICIT_TOKEN=0`；没有调用 `hf auth login`。
- 子进程退出结束本次变量使用，不会自动撤销 token。已建议下载后撤销专用 `sam3-download`，尚未获得已撤销确认。这避免新建持久 HF 登录与把 token 写入命令历史，不证明没有服务器既有凭据或管理员审计记录。
- 日志位于 `logs/download-public-weights-*.log`、`logs/download-sam3-*.log`；不保存 token、签名 URL 或邮件全文。
- bootstrap 可限定 `--stages`、`--envs`、`--models`，默认 dry run；环境 marker 匹配时只检查。权重阶段切入 insid3 Python 子进程，父／子各输出一次状态是正常行为。
- `PREPARED_NOT_GPU_VALIDATED` 和 `errors: []` 只涵盖本次选定阶段。`bootstrap_status.json` 被当前调用覆盖；`weight_inventory.json` 可能只含最近一次选中的 SAM3，不代表其他文件丢失。已有文件被跳过后的 `available` 也不等于重新传输／完整内容验证。
- URL `.part` 重启会重新传输，HF 复用缓存；不要并发多个 bootstrap 写相同 inventory／status。

### 2.4 共享队列和存储证据

此前 `cluster_probe.json`：`MaxSubmitJobsPU=60`，计入同一用户／QoS 的 pending 与 running task elements，包括 Generation、ICS 及控制／汇总作业。数组 `%N` 限制同时运行数，不能减少提交名额。`cpup` 为 CPU；GPU 有 `h200p`、`l40sp`（此前 probe 还列 `h200p-long`）。此前时限为 h200p／l40sp 3 天、h200p-long 7 天；`MaxArraySize=1001`，模板用 1000。不同分区可能共享节点，不能将硬件或任务数相加视为个人 GPU 配额。

最近用户回传的完整展开队列：

| 分区 | RUNNING | PENDING | 合计 |
|---|---:|---:|---:|
| cpup | 2 | 2 | 4 |
| h200p | 3 | 2 | 5 |
| l40sp | 8 | 43 | 51 |
| 总计 | 13 | 47 | 60 |

该快照均为现有 Generation queue／recovery／baseline／caching 作业，没有新 ICS 作业 ID；不能把它们记作本项目已提交。Priority／Resources／Dependency 或节点不可用不构成 ICS 软件错误证据。每次滚动提交都需重新计数，不能从旧快照假设还有空位。

最后 `df` 是 DINOv2／Qwen 下载后、SAM3 下载前：`/scratch` 1.9T、剩余 359G、使用率 81%；SAM3 下载后尚未复查。这是文件系统空间，不是已确认的个人 quota。数据下载需再核对实际空间，不能直接按 359G 不变估算。

### 2.5 源码与证据完整性

此前附件 `20261101-045039` 只有 978 行测试尾段，缺核心实现、suite／实验配置及 COMMIT／WORKTREE；它不是完整快照。**该源码阻塞现已解除：已成功 clone 完整远端仓库，基线 HEAD 为 `aabdbb07a25cf11cf47b95797f9d6e621cb649f6`，可读取 `ics/orchestration.py`、runner、数据脚本、配置与历史验证报告。** GitHub 网页／raw 访问失败不再作为“拿不到仓库”的理由；用户本地未提交改动和 Orion HEAD 仍不可由这次 clone 推断。

原有 `launch_suite.py` 提供 plan／submit／worker；`probe_environment.py` 只做资源清单，不做 CUDA 运算测试。完整静态提交超过 `max_submit_tasks` 会在首个 sbatch 前拒绝，并不具备滚动补位能力。本轮新增 `scripts/queue_ics.py` 和 `ics/rolling_queue.py` 实现持久化滚动控制；本轮验证见第 2 节及新报告，未部署到 Orion。

服务器证据索引：`20261101-031955` 集群／早期 bootstrap；`033425` decord／bootstrap 源码；`034558` mllm 修复；`035227` CG-ICS 成功及重复检查中断；后续用户粘贴的下载哈希／版本、SAM3 ACCEPTED 截图与队列输出。授权邮件含敏感链接，本文仅保留模型身份和成功结果。

### 2.6 本轮交付的自动队列入口

在 Orion pull 更新后、仓库目录内一次启动：

```bash
export ICS_RUNTIME="/scratch/prj0000000262_1/hansy/ce6190-ics-runtime"
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/queue_ics.py start \
  --runtime "$ICS_RUNTIME" \
  --suite configs/suite.initial.json \
  --cluster configs/cluster.orion.json \
  --run-dir "$ICS_RUNTIME/runs/ics-initial-20261101"
```

这是待执行命令，不是已提交凭证。入口默认后台管理，每 60 秒检查一次；共享上限 60、本批最多 8 个在途任务、额外预留 0、数据 workers 2、数据准备最低空闲阈值 32 GiB。32 GiB 不代表完整数据容量需求。Orion 配置为 CPU `cpup`、GPU `h200p`、account `prj0000000262`、QoS `highpriority`；保持 Generation 原作业。

依赖链为 data → preflight → 36 smoke → 对应通过配置的 1,484 个正式分片 → collect。完整 1,523 个任务先登记到 `rolling_state.json`，仅在前置条件和共享空位满足时提交。data／preflight 失败阻塞下游，smoke 失败只阻塞对应配置，collector 仍报告失败／未完成；模型失败不自动重试。第二批数据准备仍有独立阻塞，不纳入 initial。

代码／配置冻结在 `<run-dir>/source/`，同一 start 恢复原冻结版；新代码或配置使用新 run directory。后台管理器通常可跨 SSH 断开，但依赖登录节点进程不被清理；停止后重复相同 start 恢复，不能保证它是永久服务。只读检查：

```bash
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/queue_ics.py status \
  --run-dir "$ICS_RUNTIME/runs/ics-initial-20261101"
```

`status` 读取最近持久化状态，另以 flock 判断 `manager` 为 `ACTIVE`／`STOPPED`／`NOT_STARTED`；未结束且 `STOPPED` 时重跑相同 start。`watch --run-dir SAME --once` 会进行单次协调和可能的提交，仅在管理器未运行时使用。已有管理器会持锁拒绝重复 watch。CPU 8 小时／GPU 12 小时为未实测的初始 walltime，超时失败不自动重试。输出中的登记数、取得 job ID 数与完成数分别解释，完整用法见 `docs/PARALLEL_RUNS.md`；原静态／local 入口保留。

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

本轮滚动队列另增加 1 个 CPU data 任务，完整待办为 **1,523**。原科学配置、manifests 和正式分片数量没有因此增加；持久化待办数不等于同时提交数。

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
| 安装规格 | Python3.12＋独立 torch／vLLM 环境；numpy1.26.4 等显式兼容设置。历史开发 CPU 环境完整 CG import 曾停于 triton 缺失；后续 Orion CG-ICS 完整导入已通过，实际版本和修复见 2.1，GPU 推理仍待验证。 |

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

1. 本轮已新增 Orion 配置、持久化待办、冻结源码与滚动入口，并完成第 2 节所述本地验证。交付时复用已成功的源码、环境和四份权重。
2. 用户将交付更新应用到本地 clone，检查 diff 后自行 commit／push，再在 Orion pull。核对服务器 HEAD、实际数据目录和当时空间；不得用远端基线替代服务器版本证据。
3. 一次启动自动流程：登记完整计划，等待共享名额，CPU 准备缺失数据与固定 manifests，完成 preflight。已有文件先检查，不重复下载；第二批独立 suite 不阻塞首批 COCO／LVIS。
4. 在 Slurm GPU 节点逐配置 smoke；成功分支自动放行正式实验分片，失败分支保留明确原因。用 pilot／真实吞吐与显存调整资源／分片时另建有身份的计划，不把 smoke 成绩当主结果。
5. 持续滚动补位、重启恢复、防重复提交，自动收集 failure／coverage、指标、mask 案例和资源报告。没有实际作业 ID／完成输出前，只能称登记／待提交，不能称已运行。
6. 根据自己的结果决定第二批扩展、额外方法和更深入组件实验。论证应明确区分事实、假设、计划与结果；没有普遍最优或跨域结论的预设。

自动调度必须统计共享用户／QoS 的展开数组元素，有空位才提交，给管理作业留余量；其他进程抢先提交导致 QoS 拒绝时退避重试。未满足依赖的任务优先保留在持久化待办中，不塞满 Slurm pending。提交成功但写回中断时需查证恢复，不因再次启动重复提交。CPU 数据与 GPU 推理申请各自资源，不在登录节点跑模型；控制程序部署与续跑应符合实际站点规则。

当前不再缺实验授权、SAM3／DINOv3 下载授权、四份权重或核心仓库源码。**待完成的是交付后的 Orion 部署启动、实际数据核验、GPU smoke 和真实结果。** 最新队列满额是当时的共享资源限制，可通过滚动等待处理；无需取消 Generation 作业，也无需再次要求用户逐阶段批准。当前会话没有 Orion SSH，不能声称已代为提交。

## 10. 课程要求与更新记录

已确认课程要求：Visual Understanding Individual Project 占 50%；截止 2026-11-17；至少 1 个方法、2 个 datasets、自己的实验，含参数分析／关键消融／优缺点／成功失败案例；report 至少 3 页，introduction／method／experiments and analysis／conclusion；提交 report PDF＋source-code ZIP，不含 datasets／models，云链接不代替提交；与 Group Visual Generation 避免方法重叠。

本次包未重新获取《0. coursework 2026.pdf》，以上继承此前已核准内容；最终提交前仍以课程原件为准，不把最低要求当项目上限。

| 日期 | 更新 | 证据与边界 |
|---|---|---|
| 2026-10-22 | evaluator／manifest／环境探测及文献源码审计 | 历史 HTML／starter；无模型成绩。 |
| 2026-11-01 | 继承旧代码，补齐 adapters、数据层、runner、分析、组件配置、并行调度、bootstrap、更新器、维护文档 | CPU／合成测试与 dry plan；未下载实际 datasets／权重，未执行 GPU 推理。 |
| 2026-11-01，服务器准备后补录 | 三个环境基础检查、decord 元数据修复、mllm 补全、四份权重、临时凭据方式、共享 60-task 限制 | 用户终端与截图证据；仍无真实数据／GPU／ICS 新作业结果。此前服务器步骤未及时写回本文，现已补齐。 |
| 2026-11-01，取得仓库源码 | Git clone 成功，基线 `aabdbb07a25cf11cf47b95797f9d6e621cb649f6`；确认原维护文档与历史 72 项验证报告存在 | 核心源码缺失阻塞解除；不代表用户本地工作树或 Orion 版本已检查，也不代表本轮新改动通过测试。 |
| 2026-11-01，滚动队列交付 | 新增 `queue_ics.py`、持久化共享限额调度、冻结源码、Orion 配置与完整用法；data 至 collect 共 1,523 个待办 | 本轮 140 项运行、132 通过、8 可选张量测试跳过、0 failures／errors；原研究范围保留，尚未在 Orion 启动或取得新 ICS job ID。 |

后续每轮追加：实际命令／run directory、配置与 manifest hash、成功／失败数量、实验观察、仍不能确定的结论、下一轮修改。不要将“会做”或“已写配置”改写为“已完成实验”。
