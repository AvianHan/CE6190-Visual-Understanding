# CE6190 ICS — 持续研究与实验记录

最后更新：**2026-11-04（新加坡时间）**。服务器状态以用户最近回传的终端输出为准，并非实时监控。

这是当前项目的维护文档。每次修改记录：研究问题、输入输出、变化原因、运行证据、结论边界和下一步。**11 月 4 日回传的完整诊断已定位 `ics-initial-20261102-pythonfix` 的根失败：data PASS，preflight 作业 187040 在生成 LVIS fold4 reference manifest 时遇到解码后无前景的源 mask；36 个 smoke 和 1,484 个正式分片均未运行。管理器已 STOPPED，终态 COMPLETE_WITH_FAILURES，不是继续等待容量。故障发生于 11 月 2 日，见 2.13 的时间戳。本次准备显式、可审计的 LVIS 空 mask 筛选，全部新 LVIS manifests 使用独立目录；保留旧 run／旧 manifests，复用 COCO、图像、环境、权重和缓存，恢复目标为 `ics-initial-20261104-lvisfix`。本次修复尚未取得 Orion 部署或新作业证据，交付流程见 2.14。**

维护记录纠正：仓库原本已有本文，包含研究设计与历史验证；最近的服务器配置／下载步骤没有同步写回。此前说“没有维护文档”不准确，应为“没有及时更新已有文档”。本次将终端证据补回本文，保留历史研究范围和结论边界，后续继续维护同一份记录。

## 1. 已确认的方向与工作方式

- 共同任务：reference image + reference mask + query image → query mask。先做 semantic；part／personalized 独立设计协议。
- 核心：[INSID3](https://arxiv.org/abs/2603.28480) 与 [CG-ICS](https://arxiv.org/abs/2606.28149)。代码分别为 [visinf/INSID3](https://github.com/visinf/INSID3)、[Kakarot1103/CG-ICS](https://github.com/Kakarot1103/CG-ICS)。
- 项目：[AvianHan/CE6190-Visual-Understanding](https://github.com/AvianHan/CE6190-Visual-Understanding)。不重选题；不预设最终方法、数据集或项目规模上限。
- 用户已同意先跑两种 anchor、组件／backbone／参数与 reference 分析，再根据结果扩展 baselines／datasets。
- 固定交付流程（用户已多次确认）：助手完成代码并提供更新 ZIP；用户下载到本地 Mac 的 `~/Downloads` 后，助手直接在聊天给一条可粘贴的 terminal 命令，自动解压、校验、备份并应用到当前本地仓库。随后由用户在 VS Code Source Control 手动检查 diff、commit／push，再 SSH pull 运行。不要让用户另开包内 README、手工解压或定位文件，也不要再次确认这套流程。之后每次代码更新必须同时给可直接复制的 commit message；用户只免除了 2026-11-01 那一次补发，本轮及后续仍须提供。此前用户已自行完成滚动队列版本的 commit／push，助手没有代为提交 Git。
- 用户最新要求：把必要任务全部纳入持久化队列，一次启动后按数据准备 → preflight → GPU smoke → 正式分片 → 汇总自动推进。已获授权，不再逐阶段要求确认；共享 Slurm 提交上限通过滚动提交处理。
- 不取消或修改 Visual Generation 作业；ICS 与 Generation 的状态、输出和研究结论分别记录。当前会话没有 Orion SSH 通道，实际服务器启动由用户执行交付命令。
- 公司服务器凭据使用临时方式；不把 HF token、带签名下载链接或授权邮件全文写入文档、Git、脚本或日志。不重新安装／下载已经成功的资源。
- 讲解从用户真实理解出发：先任务／整体结构，再原图中的模块；新概念先用狗的例子。代码推进与学习相连，不默认已掌握之前出现的术语。

## 2. 当前状态及证据

| 项目 | 状态 | 依据／限制 | 下一步 |
|---|---|---|---|
| 2026-10-22 starter、研究 HTML、原始 evaluator | 已找回并继承 | 旧包和 `research/`；保留旧 README | 使用新记录区分历史状态 |
| 远端仓库与用户本地修改 | 解释器修复已 push；最后确认 Orion 为 `20d6432` | `git pull --ff-only`：`3478ea0..20d6432`，8 个文件更新；本次 LVIS 修复尚未部署，本地未提交工作树未直接检查 | 本次保持一条应用命令＋commit message，用户手动 commit／push |
| 固定 manifest、query GT 隔离、三种 IoU | 已实现并 CPU 验证 | `ics_eval.py`、contract tests；原 API 保留 | 实际数据 smoke |
| COCO／LVIS 和第二批数据 loaders | 已实现；真实 LVIS preflight 暴露空源 mask | 历史 13 项数据测试不足以覆盖本次故障；修复协议与审计见 2.14 | 用新 LVIS 有效池生成独立 manifests，再检查真实 smoke |
| INSID3 adapter、组件／encoder 接口 | 已实现并合成验证 | 11 项测试，含真实 upstream 处理代码＋合成 encoder 等价检查 | 预训练权重 GPU 推理 |
| CG-ICS adapter、概念／visual 组件 | 已实现并合成验证；服务器完整导入通过 | 历史 12 项 tests；SAM3／Qwen 权重已下载，`CG-ICS imports OK; model not loaded` | GPU 模型 smoke |
| runner、resume、失败重试审计、分析 | 已实现并合成验证 | runner／analysis tests；失败 attempt 不覆盖丢失 | 真实结果检查 |
| 批量调度、资源监测、诊断 | 旧 run 已终止：PASS 2、FAILED 1、BLOCKED 1,520 | data／collect PASS；preflight FAILED；本次补阶段计数及优先失败的限长诊断 | 应用更新后启动独立 lvisfix run；不重复启动终态旧 run |
| 三个环境与四份模型权重 | 基础检查／下载已通过 | 见下面实际版本、哈希和修复记录；尚无 GPU 模型加载证据 | 复用现有资源，在 Slurm 计算节点 smoke |
| 实际数据与 manifests | data 阶段已 PASS；LVIS manifests 部分生成后失败 | fold4 standard 的 2,300 episodes 验证通过，fold4 reference 遇到空 mask；不代表所有 manifests 已就绪 | 复用 12 个 COCO manifest 路径；30 个 LVIS manifests 在 `manifests/lvis-nonempty-v2/` 重新按同一有效池生成 |
| 作者原始随机 episode／resize-GT 协议的完整复现 | 尚未完成 | 本包默认统一 manifest＋原图 GT 协议 | 另建明确的 author-protocol run |
| 正式模型成绩、跨域结论、最佳组合 | 尚无结果 | 没有 GPU 模型运行日志 | 不能提前下结论 |

历史代码包验证：72 项 CPU／合成测试全部通过、零跳过，36 个配置校验及三套 suite dry plan 通过，更新器备份／保留 Git／幂等行为通过。完整报告现已通过仓库取得，见 `research/validation_2026-11-01.json`（UTC `2026-10-31T18:43:26.264912+00:00`）。这是此前版本的报告，不是本轮新增改动已通过测试的声明。合成测试证明接口和受测逻辑可用，不证明真实权重、CUDA 或所有官方依赖可运行。

前一轮滚动队列版本验证（UTC `2026-11-01T05:13:41.794368+00:00`）：**140 项运行，132 项通过，8 项因本地缺少 torch／torchvision 的可选张量测试跳过，0 failures／errors**；报告见 [research/validation_2026-11-01_rolling.json](research/validation_2026-11-01_rolling.json)。其中 39 项 rolling、14 项 CLI 测试包含真实后台子进程／继承 flock／重复 start，在模拟 Slurm 共享 60 个名额已满时登记 1,523 个待办，以及冻结恢复、提交回执不明的查证、失败分支隔离和数据下载完整性。这里的调度器是模拟程序，没有真实 Orion 提交、实际数据下载或 GPU 推理；不能写成“140 项全部通过”。

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

ICS 启动前用户回传的完整展开队列（历史快照）：

| 分区 | RUNNING | PENDING | 合计 |
|---|---:|---:|---:|
| cpup | 2 | 2 | 4 |
| h200p | 3 | 2 | 5 |
| l40sp | 8 | 43 | 51 |
| 总计 | 13 | 47 | 60 |

该快照均为现有 Generation queue／recovery／baseline／caching 作业，没有新 ICS 作业 ID；不能把它们记作本项目已提交。Priority／Resources／Dependency 或节点不可用不构成 ICS 软件错误证据。每次滚动提交都需重新计数，不能从旧快照假设还有空位。

最后 `df` 是 DINOv2／Qwen 下载后、SAM3 下载前：`/scratch` 1.9T、剩余 359G、使用率 81%；SAM3 下载后尚未复查。这是文件系统空间，不是已确认的个人 quota。数据下载需再核对实际空间，不能直接按 359G 不变估算。

### 2.5 源码与证据完整性

此前附件 `20261101-045039` 只有 978 行测试尾段，缺核心实现、suite／实验配置及 COMMIT／WORKTREE；它不是完整快照。**2026-11-01 当时已成功 clone 完整远端仓库，基线 HEAD 为 `aabdbb07a25cf11cf47b95797f9d6e621cb649f6`，可读取 `ics/orchestration.py`、runner、数据脚本、配置与历史验证报告。** 这是历史取得源码的事实，用户本地未提交改动和 Orion HEAD 仍不可由那次 clone 推断。2026-11-02 本轮工作区已不含该完整 checkout；重新 clone 未成功，现已恢复上一轮滚动更新 ZIP 和本文最新第 4 版。此次针对已知源码锚点制作增量修复器，不以摘录冒充完整仓库，也不将摘录上的回归检查表述为完整仓库测试。

原有 `launch_suite.py` 提供 plan／submit／worker；`probe_environment.py` 只做资源清单，不做 CUDA 运算测试。完整静态提交超过 `max_submit_tasks` 会在首个 sbatch 前拒绝，并不具备滚动补位能力。此前新增 `scripts/queue_ics.py` 和 `ics/rolling_queue.py` 实现持久化滚动控制；当轮验证见第 2 节及历史报告。用户已部署并启动，首次 RUNNING 见 2.8；最新终态及此次修复见 2.9–2.10。

服务器证据索引：`20261101-031955` 集群／早期 bootstrap；`033425` decord／bootstrap 源码；`034558` mllm 修复；`035227` CG-ICS 成功及重复检查中断；后续用户粘贴的下载哈希／版本、SAM3 ACCEPTED 截图与队列输出。授权邮件含敏感链接，本文仅保留模型身份和成功结果。

### 2.6 2026-11-01 自动队列入口（历史用法）

此前在 Orion pull 滚动队列更新后、仓库目录内一次启动如下。该旧 run 已失败结束；本次修复使用 2.10 的新 run，不再以此命令恢复旧冻结代码。

```bash
export ICS_RUNTIME="/scratch/prj0000000262_1/hansy/ce6190-ics-runtime"
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/queue_ics.py start \
  --runtime "$ICS_RUNTIME" \
  --suite configs/suite.initial.json \
  --cluster configs/cluster.orion.json \
  --run-dir "$ICS_RUNTIME/runs/ics-initial-20261101"
```

用户已执行上述 start（实际加了 `-I -u`），返回 STARTED；这证明管理器启动，不证明 1,523 个任务已经进入 Slurm。入口默认后台管理，每 60 秒检查一次；共享上限 60、本批最多 8 个在途任务、额外预留 0、数据 workers 2、数据准备最低空闲阈值 32 GiB。32 GiB 不代表完整数据容量需求。Orion 配置为 CPU `cpup`、GPU `h200p`、account `prj0000000262`、QoS `highpriority`；保持 Generation 原作业。

依赖链为 data → preflight → 36 smoke → 对应通过配置的 1,484 个正式分片 → collect。完整 1,523 个任务先登记到 `rolling_state.json`，仅在前置条件和共享空位满足时提交。data／preflight 失败阻塞下游，smoke 失败只阻塞对应配置，collector 仍报告失败／未完成；模型失败不自动重试。第二批数据准备仍有独立阻塞，不纳入 initial。

代码／配置冻结在 `<run-dir>/source/`，同一 start 恢复原冻结版；新代码或配置使用新 run directory。后台管理器通常可跨 SSH 断开，但依赖登录节点进程不被清理；停止后重复相同 start 恢复，不能保证它是永久服务。只读检查：

```bash
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/queue_ics.py status \
  --run-dir "$ICS_RUNTIME/runs/ics-initial-20261101"
```

`status` 读取最近持久化状态，另以 flock 判断 `manager` 为 `ACTIVE`／`STOPPED`／`NOT_STARTED`；仅未结束且代码／配置不变、`STOPPED` 时可重跑相同 start。此次旧 run 已终止且需修代码，不适用相同 start 恢复。`watch --run-dir SAME --once` 会进行单次协调和可能的提交，仅在管理器未运行时使用。已有管理器会持锁拒绝重复 watch。CPU 8 小时／GPU 12 小时为未实测的初始 walltime，超时失败不自动重试。输出中的登记数、取得 job ID 数与完成数分别解释，完整用法见 `docs/PARALLEL_RUNS.md`；原静态／local 入口保留。

### 2.7 Orion 首次启动证据（保留初始快照）

- 服务器：`orion-login01`；`git pull --ff-only` 将 `aabdbb0` fast-forward 到 `3478ea0`，更新 11 个文件。
- run directory：`/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-initial-20261101`。
- 管理器：`STARTED`，`manager_pid=3608718`；这是登录节点进程 PID，不是 Slurm job ID。日志：该 run 下 `manager.log`。
- plan fingerprint：`64807c3901662032b9105cbbb57024a20dd024a536ff4c7803ab56f47f874ec4`。
- 启动快照：`status=READY`、`terminal=false`、`total=1523`、`PENDING=1523`、`active=0`、`submitted_ever=0`、`capacity=null`、`last_error=null`、`attention=[]`，`updated_at=1793510519.0452886`。
- 策略：同用户 `hansy`／QoS `highpriority` 共享上限 60，ICS 最多 8 个在途，额外预留 0；下载 COCO/LVIS，2 workers，最低余量 32 GiB。
- 代码顺序：start 在创建后台 watcher 之前读取 summary，因此打印的 READY／零提交／capacity=null 是初始快照；不能据此断言当前无名额、调度查询失败或管理器现在仍未提交。下一步直接运行只读 status 并查看日志，不重复 start 或创建新 run。

### 2.8 首次实际运行状态（保留中间快照）

同一 run、同一 plan fingerprint。用户执行只读 `queue_ics.py status` 与 `tail -n 40 manager.log`，两处状态一致：

| 字段 | 当时值／解释 |
|---|---|
| `status`／`manager` | `RUNNING`／`ACTIVE` |
| 总任务数 | 1,523 |
| 已取得提交记录 | `submitted_ever=1` |
| 当前状态 | 1 RUNNING、1,522 PENDING、无 PASS／FAILED／BLOCKED |
| 共享容量快照 | `shared_observed=60`，包含本批 `own_active=1`；`available_before_submission=0`；`unseen_own_reservations=0` |
| 错误 | `last_error=null`、`attention=[]` |
| 更新时间 | `observed_at=1793510766.6762266`；`updated_at=1793510767.6062646` |

已确认真实提交与运行，不再只是本地登记。按当前依赖图且尚无任何前置 PASS，首个运行任务对应 COCO/LVIS CPU 数据准备；摘要未显示具体 Slurm job ID 或下载日志，所以未确认下载进度、数据完整性或空间检查结果。余下 PENDING 是内部待办，并非已向 Slurm 提交的 1,522 个 pending 作业；它们也需要依赖通过，不能把等待都归因于共享名额满额。

当时的处理是继续等待已有管理器，不重复 start。`ACTIVE` 与无错误只描述该中间快照；后续实际发生数据失败并结束管理器，详见 2.9。现在不能继续沿用“仍在运行／只需等待”的结论。

### 2.9 首轮 Orion 失败证据与根因（2026-11-02 补录）

用户回传旧 run 的状态、数据日志和汇总：

| 项目 | 实际证据与解释 |
|---|---|
| 终态／管理器 | `COMPLETE_WITH_FAILURES`／`STOPPED`；不是仍在等待共享空位 |
| 任务总数／提交数 | 1,523 个内部待办；`submitted_ever=2`，对应 data 与 collector，不能理解为已跑两次模型 |
| 总状态计数 | `FAILED=1`、`BLOCKED=1521`、`PASS=1`；当前 active 0 |
| 各阶段 | data FAILED 1；preflight BLOCKED 1；smoke BLOCKED 36；experiment BLOCKED 1,484；collect PASS 1 |
| 失败作业 | data，Slurm job **185464**，`Slurm FAILED exit 1:0` |
| 汇总结果 | `NOT_RUN_OR_DEPENDENCY_BLOCKED=1520`；分析返回码 `None`；尚无汇总 CSV；collector PASS 只表示完成收集失败信息 |
| 最后状态时间 | `updated_at=1793514163.395816`；`last_error=null` 不否定具体任务失败 |
| 最后容量观察 | `shared_observed=60`、`own_active=0`、`available_before_submission=0`、`unseen_own_reservations=0`，`observed_at=1793514163.1585953`；是历史快照，不是实时容量 |

实际 **没有 preflight、GPU smoke 或正式实验执行**，所以目前没有模型指标、mask 案例或吞吐结论。1,520 对应 36 smoke＋1,484 正式分片；总 BLOCKED 1,521 另含 preflight，不是计数矛盾。

185464 已成功下载并解压的资源如下；路径均在原 `$ICS_RUNTIME/data` 下，必须复用：

| asset ID | 压缩包 bytes | verified_files | 解压目标（相对 data） | observed SHA-256 |
|---|---:|---:|---|---|
| `coco2014_val_images` | 6,645,013,297 | 40,504 | `COCO2014` | `fe9be816052049c34717e077d9e34aa60814a55679f804cd043e3cbee3b9fde0` |
| `coco2014_train_images` | 13,510,573,713 | 82,783 | `COCO2014` | `ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775` |
| `coco2017_val_images` | 815,585,330 | 5,000 | `LVIS/coco` | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `coco2017_train_images` | 19,336,861,798 | 118,287 | `LVIS/coco` | `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929` |

这些日志的 `expected_sha256=null`、`upstream_checksum_verified=false`；表中哈希是本地观察值，不能称为官方 checksum 核验。成功图像解压也不表示 COCO／LVIS 标注、split 或 manifests 已完整。失败项是 `coco2014_train_semantic`、`coco2014_val_semantic`、`coco2014_splits`、`lvis_annotations`，均在调用 `-m gdown` 时退出。

直接失败信息为 base Python 的 `No module named gdown`，命令实际使用：

```text
$ICS_RUNTIME/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
```

代码根因为 `ics.orchestration.resource_profile()` 对解释器使用通用 `resolve_path()`；其中 `Path.resolve()` 把 `$ICS_RUNTIME/envs/insid3/bin/python` 的虚拟环境符号链接解析成上述 base Python。这样虽然二进制文件可执行，Python 的环境身份已改变。该问题同时影响资源 `python` 与 owned MLLM 的 `mllm_python`，所以只给 base Python 安装 `gdown` 不能修好后续 GPU worker／vLLM 环境。**现有失败日志没有证明正确的 insid3 虚拟环境也缺少 `gdown`。**

### 2.10 此轮修复、检查与新 run 恢复方案

以下保留修复交付时的方案与验证边界；随后用户已完成部署和启动，见 2.11。

- 解释器路径独立使用 `resolve_executable()`：严格展开环境变量、转为绝对路径，同时保留虚拟环境符号链接；数据、配置等普通路径继续原有规范化行为。覆盖 resource `python` 和 `mllm_python`，不改模型算法、研究配置、原权重与已成功环境。
- `scripts/apply_python_fix.py` 在用户真实仓库中核对已知源码锚点后定点修补 `ics/orchestration.py`，保留其余未知源码与用户工作；不能安全匹配时拒绝应用，不以摘录替换整文件。更新流程沿用校验、备份与幂等保护，不自动 commit／push。
- 数据下载器在启动传输前检查本次缺失资源需要的依赖，并提供只读 `--check-dependencies-only`；报告所用 interpreter、prefix、base_prefix，便于核验实际环境。已经完整的 archive／有效 receipts 复用，不因不需下载的 Drive 项强制安装依赖。
- Orion 恢复先检查 **`$ICS_RUNTIME/envs/insid3/bin/python`** 下是否可用 `gdown`；只有缺失时才向该 insid3 环境补装 `gdown`。不修改 base Python，不重建三个模型环境，不升级 torch／vLLM，不重下四份权重。
- 旧 `$ICS_RUNTIME/runs/ics-initial-20261101` 已终止且保存旧冻结源码；保留其 `source/`、`rolling_state.json`、日志和失败记录，不手改状态、不直接改冻结源码，也不靠重复旧 `start` 获取修复。
- 用户应用更新、VS Code 手动 commit／push、Orion `git pull --ff-only` 后，启动新 **`$ICS_RUNTIME/runs/ics-initial-20261102-pythonfix`**。继续使用 `configs/suite.initial.json`、`configs/cluster.orion.json` 和原 `$ICS_RUNTIME/data`，复用上述成功图像资源，只补缺失项；新的冻结计划与旧失败计划分别保留。依赖仍是 data → preflight → smoke → experiment → collect。

本轮本地验证：**37 项全部通过，0 跳过、0 failures／errors**，包括 23 项数据下载器测试和 14 项解释器／更新器测试；报告见 [research/validation_2026-11-02_python_fix.json](research/validation_2026-11-02_python_fix.json)。其中真实本地 venv 复现了符号链接保留时进入 venv、解引用后进入 base Python 的差异，并验证 worker／MLLM 路径修补。此前 140 项报告仅属于上一轮版本。恢复的更新包不含完整当前 Git checkout，解释器回归基于已知原代码锚点／摘录和真实本地 venv；不能声称已重跑完整仓库测试、真实 Orion 作业或 GPU 推理。此时只完成本地修复准备，尚无用户应用本次更新、推送新 commit 或新 run 提交的证据。

本次交付包 `CE6190-ICS-python-fix-20261102.zip` 下载至 Mac `~/Downloads` 后，在本地项目仓库目录粘贴一条命令应用；不要求用户先解压或打开包内说明：

```bash
(ICS_FIX_DIR="$(mktemp -d)" && unzip -q "$HOME/Downloads/CE6190-ICS-python-fix-20261102.zip" -d "$ICS_FIX_DIR" && python3 "$ICS_FIX_DIR/CE6190-Visual-Understanding/scripts/apply_python_fix.py" --target "$PWD" --apply)
```

随后仍由用户在 VS Code 手动检查 diff、commit／push。本次 commit message：

```text
fix: preserve virtualenv Python paths and precheck data dependencies
```

完整 Orion 恢复命令见 [docs/PYTHON_ENV_RECOVERY.md](docs/PYTHON_ENV_RECOVERY.md)，实际聊天交付同时直接给可复制命令，不把阅读该文件作为前置要求。恢复包含 pull、检查 `resource_profile` 中普通 worker 与 MLLM 的环境路径、仅在正确 insid3 缺少时安装 `gdown>=5,<6`、只读依赖检查，以及启动上述新 run；这些操作尚未取得用户的实际执行输出。

### 2.11 修复已部署，新 run 已启动（2026-11-02）

用户回传实际执行结果，已完成上一节的本地手动 commit／push 与 Orion 恢复命令：

| 项目 | 当时服务器证据 |
|---|---|
| 代码 | `git pull --ff-only`：`3478ea0..20d6432`，8 个文件更新，包含 `ics/orchestration.py` 的解释器修复 |
| 路径检查 | preflight、insid3、cgics_visual、cgics_mllm 全部通过；也检查 owned MLLM 的 `mllm_python` |
| 下载依赖 | `dependencies_ready`；Python 为 `$ICS_RUNTIME/envs/insid3/bin/python`，prefix 为 `envs/insid3`，与 base_prefix 不同 |
| gdown | 正确 insid3 环境可以启动；此次输出未出现 pip 安装步骤，没有证据需要再次安装 |
| 仍需传输的 Drive 项 | `coco2014_train_semantic`、`coco2014_val_semantic`、`coco2014_splits`、`lvis_annotations` |
| 新 run | `$ICS_RUNTIME/runs/ics-initial-20261102-pythonfix` |
| 管理器 | `STARTED`，PID `3314891`，日志为新 run 的 `manager.log`；PID 不是 Slurm job ID |
| 初始状态 | `READY`，`terminal=false`，total 1,523，PENDING 1,523，active 0，submitted_ever 0 |
| 初始诊断 | `capacity=null`、`last_error=null`、`attention=[]`；`updated_at=1793599239.6199355` |
| 新 plan fingerprint | `b792c006fb247d9aed670405bf47d2612bfbfe2d58bac4dd6951ff8cd8e2bc56` |
| 策略 | ICS 最多 8 在途；同用户／QoS 共享上限 60；预留 0；coco／lvis、2 data workers、最低余量 32 GiB；仍用原 `$ICS_RUNTIME/data` |

这是启动时打印的初始快照，尚不证明 manager 当前仍 ACTIVE，也不证明已提交 Slurm 作业、补完标注或开始 GPU 推理。`capacity=null` 不能解释为名额已满。用户粘贴的命令末尾显示有错位，但实际输出已经 STARTED，不应仅凭回显排版再次 start 或新建第三个 run。

下一步只读查询新 run 的冻结入口；使用完整路径，不依赖上次括号子 shell 内的变量：

```bash
/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/envs/insid3/bin/python -I /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-initial-20261102-pythonfix/source/scripts/queue_ics.py status --run-dir /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-initial-20261102-pythonfix
```

先确认管理器与阶段进展，完成后再查看／导出结果；若出现 FAILED／BLOCKED／STOPPED，再查看对应任务日志。此次只补记实际部署证据，没有新的代码补丁或再次安装／提交动作。

后续只读 status 已回传（同一 plan fingerprint）：

| 字段 | 最新观察 |
|---|---|
| manager／status | `ACTIVE`／`WAITING_CAPACITY`，`terminal=false` |
| 待办／提交 | PENDING 1,523；active 0；submitted_ever 0 |
| 共享容量 | `shared_observed=60`；`available_before_submission=0`；`own_active=0`；`unseen_own_reservations=0` |
| 错误／关注项 | `last_error=null`；`attention=[]` |
| 时间戳 | `observed_at=1793599666.2166855`；`updated_at=1793599666.6423755` |

该快照已确认管理器当时仍持锁活动，等待的是同用户／QoS 的共享提交名额。快照中的 60 个占用不来自新 ICS run；未回传具体作业列表，不能断言全部属于 Generation。1,523 PENDING 是内部待办，尚未排入 Slurm。管理器按既定周期继续观察，有空位时自动尝试提交；其他提交者也可能竞争空位，不保证确切启动时间。当时无须重新 start、安装或修改代码；后来实际进入 data／preflight 并失败，最终证据见 2.13。

### 2.12 11 月 4 日首次异常回传：当时待查首个错误

用户再次查询同一新 run，上传 `粘贴的文本 (1)(20261104-030007).txt`（36,625 bytes、1,043 行）。附件从一个任务对象中间开始，只是输出尾部：包含 208 条 BLOCKED 状态、207 个完整任务 key，依赖报错涉及 15 个不同 `smoke-...` 名称，末尾为 `manager: STOPPED` 及 `resume: Repeat the same start command`。

- 这些只是附件内计数，**不是整个 run 的总失败／阻塞数量**。未看到总体 status、terminal、counts、submitted_ever、首个 FAILED 任务、Slurm job ID 或 traceback。
- `Failed prerequisites: smoke-...` 表示下游没有获准运行；不能据此认定这些 smoke 全部实际执行并失败。更早的 data 或 preflight 失败也可能造成连锁阻塞。
- 已有 CLI 的 manager 检查在未持锁时统一输出上述 resume 提示，不看 terminal；终态 run 再 start 只会返回 NOT_NEEDED。必须先取得总体状态和根失败日志，再决定恢复步骤。
- 因此当前不能宣称仍正常排队，也不能沿用旧根因认定再次缺 gdown、GPU 依赖或空间不足。没有足够证据进行安装、改环境或重新排队。
- 本次改用精简只读诊断：总体状态排除庞大的 attention，按阶段计数，列出真正 FAILED／UNCERTAIN 任务，最多选三个不同阶段／资源的代表任务，读取 Slurm 日志、rolling marker、runner／vLLM 日志。若无直接失败或尚未终态，补管理器尾部。这样避免长 BLOCKED 列表淹没关键错误。
- 当前本地重新发现旧基线工作树（HEAD aabdbb0，含 rolling 工作树文件），可核对已交付状态／日志逻辑；它不等同于服务器 20d6432，也不提供服务器最新运行文件。

当时下一步为回传精简只读诊断，尚未改代码、安装依赖或重启作业。随后用户已提供完整输出，根因与修复交付记录见 2.13–2.14；本节保留为诊断过程，不再代表当前缺少根失败日志。

### 2.13 完整诊断：数据通过，LVIS reference 预检失败（11 月 4 日回传）

| 项目 | 实际证据与解释 |
|---|---|
| run／plan | `ics-initial-20261102-pythonfix`；fingerprint `b792c006fb247d9aed670405bf47d2612bfbfe2d58bac4dd6951ff8cd8e2bc56` |
| 总体终态 | `manager=STOPPED`、`status=COMPLETE_WITH_FAILURES`、`terminal=true`；不是管理器失联或仍在排队 |
| 任务计数 | PASS 2、FAILED 1、BLOCKED 1,520；`submitted_ever=3`，对应 data、preflight、collect |
| 分阶段 | data PASS 1；preflight FAILED 1；smoke BLOCKED 36；experiment BLOCKED 1,484；collect PASS 1 |
| 直接失败 | preflight，Slurm job **187040**，`Slurm FAILED exit 1:0`；marker `status=FAIL`、`error_type=CalledProcessError` |
| 失败输入 | `LVIS/coco/train2017/000000578055.jpg` 的某个 image／concept 源 mask 解码后为空；发生在 reference asset 构造时 |
| 失败 recipe | LVIS fold 4，queries 500、references-per-query 50、shots 1、seed 0、query-seed 0，`--trust-prepared-pickle` |
| preflight 开始／结束 | `1793622940.253583`／`1793624697.7603624`，即新加坡时间 **2026-11-02 20:35:40–21:04:57**；11 月 4 日是回传时间 |
| 终态更新时间 | `1793624763.4117153`，新加坡时间 2026-11-02 21:06:03；`last_error=null` 不否定 task FAILED |
| 最后容量快照 | shared 60、own active 0、available 0、unseen reservations 0；`observed_at=1793624762.9600296`；容量不是此次直接失败原因 |

报错原文：

```text
ValueError: Selected source mask is empty: /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/data/LVIS/coco/train2017/000000578055.jpg
```

此前同一日志中 **LVIS fold4 standard** 已生成并验证 `valid=true`／`status=ready`：2,300 episodes，manifest SHA-256 为 `83d18e785ad81a55494d4f9f77804f160bd5764e995281dd6c1f691f2603845c`。query draws 2,300、unique query/concept pairs 1,532、query repeats 768；references-per-query 1、shots 1、seed／query-seed 均为 0。standard 成功并不保证扩大 reference 采样后不会碰到另一条空 mask。

本轮 data 已 PASS，说明上一轮解释器／gdown 故障未再次阻断该数据任务；**当前没有 GPU smoke 或正式模型结果**。collector PASS 只说明失败汇总完成。缺少服务器实际 pickle 内容，尚不能确定该源标注为什么解码为空，不能宣称图像损坏、标注整体下载损坏或作者也使用同样过滤规则。

### 2.14 LVIS 空 mask 修复与独立恢复计划（2026-11-04）

修复采用显式的 `--empty-mask-policy exclude`，仅适用于 LVIS；默认仍为严格 `error`。先固定原类别集合与 fold 分配，再扫描当前 fold 的所有 image／concept 候选：只有**成功解码、尺寸正确、合并后前景像素数为 0** 的源 mask 才排除。编码错误、尺寸不符、缺失文件、损坏的生成 mask 缓存等继续报错，不能把任意异常静默跳过。筛选后任何原类别无足够样本均明确失败，不允许类别消失或重新编号 fold。

query 和 reference 使用同一有效池；采样保持确定性、分离 query／reference RNG，并保持不同 reference budget 的 query 对齐。protocol sidecar 记录 `sampling.empty_mask_policy`，以及 `sample_eligibility` 的策略／版本、输入／有效／排除数量、逐类别计数、完整 `excluded_samples` 与有效池 `pool_fingerprint`。实际排除记录包含 image、concept、原因及标注数量；服务器真实排除数量要等新 preflight 输出，不能预填为 1。此筛选改变采样总体，必须使用新的协议身份，不能称为作者原始随机协议的精确复现。

新 suite 为 `configs/suite.initial.lvisfix.json`：12 个 COCO manifest 的路径和 recipe 保持原样；30 个 LVIS manifests 全部使用 `${ICS_RUNTIME}/manifests/lvis-nonempty-v2/`，不覆盖或混用旧 LVIS standard／reference／component manifests。已成功的图像、下载 receipts、archives、模型环境、四份权重和有效 mask 缓存继续复用。原 36 配置、42 manifests、36 smoke、1,484 正式分片及完整 1,523 项队列范围保持不变。

preflight 传递并核对 manifest 的策略／协议身份；旧 sidecar 未声明 policy 时只按严格 `error` 处理，所以旧 COCO 可按原身份复用。状态输出新增 `stage_counts`；`attention` 最多 10 项，优先真正 FAILED／UNCERTAIN／WAITING_QUOTA，再列 BLOCKED，附 kind／resource／job_id，并给总数与截断标识；完整任务记录仍保留。终态 STOPPED 不再建议重复 start；非终态意外停机仍可显示恢复提示。未修改自动重试或旧状态重置行为。

本次本地完整测试：**189 项运行，181 项通过，8 项因缺少 torch／torchvision 跳过，0 failures／errors**，耗时 1.700 秒；报告见 [research/validation_2026-11-04_lvis_fix.json](research/validation_2026-11-04_lvis_fix.json)。其中 23 项 data、18 项 orchestration、5 项 prepare CLI、58 项 queue／CLI 测试全部通过；使用 `pycocotools 2.0.10` 实际解码合成 polygon、压缩与未压缩 RLE，验证空 union 筛选、异常不被吞掉、类别／fold 保留、采样对齐和排除审计。8 项跳过的是可选模型 tensor 测试，不能写成 189 项全部通过；这些 CPU／合成与模拟调度测试不代表服务器部署、真实 LVIS 全量筛选或 GPU 验证。

另以真实 venv 子进程在本地合成 LVIS fold4（空 mask 1、有效样本 3）连续两次执行 `run_preflight`：standard／reference／component 三个 cohort 的 query 前缀和有效池一致，第二次文件 hash／mtime 不变，解释器路径保留 venv 身份。更新器验证备份字节匹配，保留 Git、未列入更新的文件及前次 Python 修复，重复应用返回 `UP_TO_DATE`；增量包包含 16 个 payload 文件加 1 份清单。

交付包 `CE6190-ICS-lvis-fix-20261104.zip` 下载到 Mac `~/Downloads` 后，在已有本地仓库目录执行一条命令，沿用更新器校验、备份和幂等行为：

```bash
(ICS_FIX_DIR="$(mktemp -d)" && unzip -q "$HOME/Downloads/CE6190-ICS-lvis-fix-20261104.zip" -d "$ICS_FIX_DIR" && python3 "$ICS_FIX_DIR/CE6190-Visual-Understanding/scripts/apply_update.py" --target "$PWD" --apply)
```

用户在 VS Code 手动检查 diff、commit／push。本次 commit message：

```text
fix: audit empty LVIS masks and isolate recovery manifests
```

随后 Orion `git pull --ff-only`，用新 suite 启动 **`$ICS_RUNTIME/runs/ics-initial-20261104-lvisfix`**。完整可复制命令见 [docs/LVIS_EMPTY_MASK_RECOVERY.md](docs/LVIS_EMPTY_MASK_RECOVERY.md)，聊天交付同时直接提供命令，不要求先打开说明。旧两轮 run 的 frozen source、状态和日志保留；不手改 BLOCKED／FAIL，不在旧终态 run 上重复 start。当前尚未取得用户应用此次更新、push、Orion pull 或新 run 启动证据。

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

1. 已完成的历史准备保留：三个模型环境的基础检查、四份权重、滚动队列配置与首次 Orion 部署；四份图像资源已实际下载／解压。首轮数据任务 185464 失败，旧 run 终止，无 GPU 实验结果。
2. 已完成：交付解释器路径与下载依赖早检查修复、维护记录、单条应用命令和 commit message；用户完成 commit／push，Orion pull 至 20d6432。
3. 已完成：真正的 insid3 Python 通过依赖检查；`ics-initial-20261102-pythonfix` 的 data PASS，但 preflight 187040 因 LVIS 空源 mask 失败并终止，36 smoke／1,484 正式分片均 BLOCKED。旧 run 状态与冻结源码保留，没有 GPU 成绩。
4. 当前恢复方案：应用本次 LVIS 显式筛选与审计修复，用户手动 commit／push 后 Orion pull；以 `configs/suite.initial.lvisfix.json` 启动 `ics-initial-20261104-lvisfix`，保留旧 manifests、复用成功资源，并在新目录生成统一有效池的 LVIS manifests。先取得实际新 preflight／smoke 日志；本次尚未部署或提交。
5. 持续滚动补位、防重复提交并收集 failure／coverage、指标、mask 案例和资源。先取得真实 smoke 吞吐与显存，必要时另建有身份的 pilot／调整分片，不把少量 smoke 当主结果。严格区分内部登记、取得 job ID、RUNNING 与 PASS。
6. 根据自己的结果决定第二批扩展、额外方法和深入组件实验。论证区分事实、假设、计划与结果；没有普遍最优或跨域结论的预设。

自动调度必须统计共享用户／QoS 的展开数组元素，有空位才提交，给管理作业留余量；其他进程抢先提交导致 QoS 拒绝时退避重试。未满足依赖的任务优先保留在持久化待办中，不塞满 Slurm pending。提交成功但写回中断时需查证恢复，不因再次启动重复提交。CPU 数据与 GPU 推理申请各自资源，不在登录节点跑模型；控制程序部署与续跑应符合实际站点规则。

当前不缺实验授权、SAM3／DINOv3 下载授权或四份权重。**最新阻塞已定位为 LVIS preflight 空源 mask；data 已通过，GPU 阶段未开始。本次准备可审计的采样修复与新 run，不再仅等待共享名额。** 保留两轮旧 run 的全部证据；实际筛选数量、完整 manifests、GPU 可运行性和正式结果仍待服务器执行。无需取消 Generation 作业，也无需再次要求用户逐阶段批准。当前会话没有 Orion SSH，不能声称已代为安装、提交或恢复服务器运行。

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
| 2026-11-01，Orion 首次启动 | 用户 pull 至 `3478ea0`，管理器 STARTED／PID 3608718，登记完整 1,523 个待办；固定之后每次更新附 commit message | 初始 READY 快照未含首次轮询；尚无实际 ICS job ID／数据完成／GPU 成绩证据。 |
| 2026-11-01，首次实际作业 RUNNING | 同一 plan 下 submitted_ever=1、RUNNING=1、PENDING=1522，manager ACTIVE；共享 60/60 含本批 1 个 | 无 last_error／attention；按依赖图为首个数据任务，具体 job ID／下载日志未展示，数据完成与 GPU 仍未验证。 |
| 2026-11-02，首轮终态与解释器修复 | 记录 data job 185464 失败、1,521 BLOCKED、collector PASS；定位 `Path.resolve()` 解引用 venv Python，补解释器路径修复及下载依赖早检查；恢复方案改用 `ics-initial-20261102-pythonfix` 复用成功资源 | 原 run 共提交 2 个作业，无 GPU 运行；本轮 37 项回归全过；完整 checkout 未恢复，未声称部署或重跑历史 140 项。 |
| 2026-11-02，修复部署与新 run STARTED | Orion 更新到 20d6432；四类 Python 路径检查和 insid3 gdown 依赖检查通过；新 run ics-initial-20261102-pythonfix，PID 3314891 | 初始 READY、1,523 PENDING、submitted_ever=0、capacity=null；尚未确认实际提交、数据完成或 GPU 运行，下一步只读 status。 |
| 2026-11-02，新 run 容量等待 | manager ACTIVE、WAITING_CAPACITY；共享 60/60，本批 0 提交、1,523 内部 PENDING | last_error=null、attention=[]；继续自动轮询，尚无新数据作业或 GPU 结果，不重复 start。 |
| 2026-11-04，新 run 异常回传 | 附件尾部显示 STOPPED、208 条 BLOCKED 片段及 smoke 前置依赖错误 | 不是完整统计；缺直接 FAILED／总状态／traceback；已准备精简只读诊断，暂不按通用 resume 提示重启。 |
| 2026-11-04，完整诊断与 LVIS 修复准备 | 已确认 data PASS、preflight 187040 因 fold4 reference 的空源 mask 失败；总 PASS 2／FAILED 1／BLOCKED 1,520。显式 LVIS 非空池、排除审计、新 manifest 目录及状态摘要修复；恢复 run 为 ics-initial-20261104-lvisfix | 故障实际发生于 11 月 2 日，11 月 4 日回传。旧标准 manifest 与所有旧 run 保留；本地 189 项运行／181 通过／8 可选 tensor 跳过／0 failures／errors，尚未部署或提交新 run，无 GPU 验证。 |

后续每轮追加：实际命令／run directory、配置与 manifest hash、成功／失败数量、实验观察、仍不能确定的结论、下一轮修改。不要将“会做”或“已写配置”改写为“已完成实验”。
