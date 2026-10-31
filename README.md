# CE6190 Visual Understanding — In-context Segmentation

更新：2026-11-01（Singapore）。项目主线是 INSID3 与 CG-ICS 的复现、统一比较和组件分析。**已有可执行实现及 CPU／合成输入验证；尚未完成真实权重推理、公司 GPU 验证或正式实验，没有自有模型成绩。**

持续记录见 [CE6190_ICS_PROJECT_LOG.md](CE6190_ICS_PROJECT_LOG.md)。旧版说明保留于 `research/README_2026-10-22.md`；旧审计文件是历史记录，不能代替本次验证状态。

## 代码分别解决什么问题

| 文件／目录 | 输入 → 输出 | 要解决的问题 |
|---|---|---|
| `ics_eval.py` | 固定 episode、预测与 GT → 校验和三种 IoU 聚合 | 比较对象、GT 隔离和指标定义是否一致？继承旧实现。 |
| `ics/data.py`、`scripts/prepare_episodes.py` | 公开图像／标注／split → manifest、协议侧录、binary masks | 所有方法是否拿到同一只 reference dog 和同一张 query？ |
| `ics/adapters/insid3.py` | reference image/mask + query image → query mask、成本 | 冻结 features、matching、clustering、aggregation 的作用。 |
| `ics/adapters/cgics.py` | 同样三种输入 → query mask、概念／搜索记录、成本 | 概念生成、搜索和 SAM3 visual/text prompts 的作用。 |
| `ics/runner.py` | 配置 + manifest → 逐样本预测、统计、失败记录 | 可续跑、可追溯，并且不把 query GT 交给模型。 |
| `ics/analysis.py` | 已保存结果 → CSV、案例 HTML | accuracy、reference 敏感性、重复运行、成本和失败覆盖率。 |
| `ics/orchestration.py` | suite + 实际资源配置 → Slurm／local jobs | 先试跑，再让独立配置并行，避免重复成功任务。 |
| `scripts/bootstrap_runtime.py` | 固定版本配置 → 项目外环境／源码／权重清单 | 隔离 DINO、SAM3、vLLM 的依赖，复用已准备资源。 |

例如 reference 是草地上的狗，query 是沙发上的另一只狗：模型只得到 reference image、狗的 reference mask、query image。query 的狗 mask 直到模型返回预测后才由 evaluator 读取。这个例子属于 semantic 目标，不等于寻找同一只宠物；part／personalized 将使用独立协议。

## 下载更新到现有 VS Code clone

在本地终端进入已有的 `CE6190-Visual-Understanding` clone。将下载 ZIP 解压到独立目录，然后运行其中的 updater：

```bash
python3 /absolute/path/to/extracted/CE6190-Visual-Understanding/scripts/apply_update.py --target "$PWD"
python3 /absolute/path/to/extracted/CE6190-Visual-Understanding/scripts/apply_update.py --target "$PWD" --apply
```

第一条仅显示变更；第二条核对包内 SHA256，并在 clone 的同级 backup 目录备份所有将替换的旧文件。它保留 `.git`、未列入更新的文件，不做 commit／push。随后由你在 VS Code Source Control 检查 diff、手动 commit／push。旧环境探测结果不代表服务器配置。

本次无法读取你本地 VS Code 工作树，也未能通过无凭据访问读取项目远端；更新基于找回的 2026-10-22 starter 和已核对的官方源代码。因此若 clone 中有之后的独立修改，应从备份和 Source Control diff 核对，不能假定本包已经包含它们。

## 服务器准备

SSH 后进入已有 clone，先同步并只读探测：

```bash
git pull --ff-only
python3 scripts/probe_cluster.py --output /tmp/ce6190-ics-cluster.json
```

探测记录调度命令、partition／QOS／配额可见信息、Python／module 和 GPU 可见性。登录节点探测不等于已分配 GPU；在实际 allocation 内另运行 `scripts/probe_environment.py`。`disk free` 不等于你的项目 quota。

选择**项目目录以外**的持久运行目录，将占位路径改为真实路径：

```bash
export ICS_RUNTIME="/your/project-storage/ce6190-ics-runtime"
python3 scripts/bootstrap_runtime.py --runtime "$ICS_RUNTIME"
python3 scripts/bootstrap_runtime.py --runtime "$ICS_RUNTIME" --python /actual/path/to/python3.12 --execute
```

默认只显示准备计划；`--execute` 才执行。可用 `--stages sources envs weights`、`--envs insid3 cgics mllm` 和 `--models dinov3 dinov2 sam3 qwen` 选择阶段。已锁定源码和成功安装的环境会校验后复用，失败阶段有清单；不自动覆盖不同版本或本地修改。运行目录保存实际 `pip freeze`。

三个独立环境使用 Python 3.12：INSID3 的 torch 2.7.1/cu126、CG-ICS 的 torch 2.9.0/cu128，以及 vLLM 0.11.2 环境。它们是本项目安装规格，**尚未在你的驱动／GPU 上验证**，也不宣称等同作者的全部原始环境。CG-ICS 的 vendored SAM3 使用直接源码加载和显式依赖；其中 numpy 1.26.4 的选择及 import 路径适配均有记录。

SAM3 下载需要已有的官方授权；DINOv3 需要官方原始 `.pth`，放在 `$ICS_RUNTIME/weights/dinov3_L.pth`，或私下设置 `ICS_DINOV3_URL`。不能将 Transformers 格式 safetensors 改名代替。脚本不会代你接受模型条款。若已有数据／权重，先复用，不重复下载。

准备第一批 COCO／LVIS（先看 dry plan，再下载）：

```bash
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/download_data.py --datasets coco lvis --data-root "$ICS_RUNTIME/data"
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/download_data.py --datasets coco lvis --data-root "$ICS_RUNTIME/data" --workers 2 --execute
```

数据布局、split 和标注规则见 [docs/DATA_PROTOCOLS.md](docs/DATA_PROTOCOLS.md)。下载 URL 来自固定版本作者说明；真实文件下载与服务器网络仍待验证。自动准备 manifest 会明确使用该来源的 pickle，不能替换成未知来源 pickle。

## 一次提交可并行实验

将 `configs/cluster.example.json` 复制为 `configs/cluster.local.json`，依据探测填写真实的 account、partition、time、QOS、每类 max_parallel 和提交配额。**这些值没有预先假定。**环境 Python 路径默认从 `ICS_RUNTIME` 展开。

```bash
python3 scripts/launch_suite.py --suite configs/suite.initial.json --cluster configs/cluster.local.json --run-dir "$ICS_RUNTIME/runs/batch-001"
python3 scripts/launch_suite.py --suite configs/suite.initial.json --cluster configs/cluster.local.json --run-dir "$ICS_RUNTIME/runs/batch-001" --submit
```

第一条写冻结计划；第二条一次提交依赖链：data／manifest preflight → 各配置 smoke → 通过配置的并行正式任务 → analysis。每项正式任务检查其对应 smoke 的结果。`suite.pilot.json` 是同一配置集合的少量样本方案，可先用来测实际吞吐和资源；使用独立 run directory。

CG-ICS 概念配置默认每 job 请求两张 GPU，分别运行 SAM3 与自己的 vLLM；这是隔离测量方案，不是已验证的最低硬件要求。不同任务拥有独立端口／进程；结束时清理自己的进程组。支持按实际条件改资源规格。各类并行数不是全局 GPU 总数；应综合公司的真实配额填写。

长 cohort 已支持分片和 Slurm array 分块；成功 episodes 可续跑，不重算。首批工作量和分片数是可调整的执行配置，不是研究规模上限。详见 [docs/PARALLEL_RUNS.md](docs/PARALLEL_RUNS.md)。

第二批独立方案 `configs/suite.phase2.json` 预设 FSS-1000、SUIM、ISIC2018、iSAID；PASCAL-5i 为可选禁用条目。其数据准备未全部自动化，不会阻塞第一批。没有创建 part／personalized 混合排行榜。

## 分析、续跑与诊断

```bash
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/analyze_results.py --root "$ICS_RUNTIME/runs/batch-001/results" --output "$ICS_RUNTIME/runs/batch-001/analysis"
python3 scripts/collect_diagnostics.py --run-dir "$ICS_RUNTIME/runs/batch-001" --output /tmp/ce6190-ics-diagnostics.tar.gz
```

分析读已有预测，不占 GPU 重新推理。诊断包仅选取有限日志／概要，排除数据、模型和样本图像；分享前自行检查。失败不会自动反复消耗 GPU。手动重试用 runner 的 `--resume --retry-failed`，旧失败记录和成本会保留。

同时报告 mean episode IoU、class mean episode IoU、macro class pooled IoU，不能都简称 mIoU。原始计数、fold、manifest、模型与源码哈希、失败／缺失数量均保存。当前不自动混合 folds，也不提供 bootstrap confidence interval 或 boundary metric。原图 GT 评估、固定配对是**统一比较协议**，不同于作者随机采样／resize-GT 的原始复现；两者不得混称。

资源记录包括加载与推理 wall-time、PyTorch 本进程峰值、实际 SAM／MLLM 调用和 tokens。vLLM 另用进程归属 GPU 显存采样，采样峰值是下界；不同时间的设备峰值不相加冒充瞬时峰值。重试后的 accuracy 使用最新 attempt，成本同时报告保留下来的全部 attempts。

## 验证与提交

```bash
python3 -m unittest discover -s tests -v
ICS_TEST_INSID3_SOURCE="$ICS_RUNTIME/sources/INSID3" python3 -m unittest discover -s tests -v
```

需要相应 CPU torch、numpy、Pillow 等测试依赖。第二条包含真实 upstream 处理代码＋小型合成 encoder 的输出等价检查，仍不是预训练模型实验。最新验证详见 `research/validation_2026-11-01.json`。

课程提交仍需最终 report PDF 和 source-code ZIP，不能以仓库／云链接代替；代码包排除 datasets、weights、环境及 credentials。截止 2026-11-17。最低 1 方法／2 datasets、至少 3 页、自己的实验、参数分析／消融／优缺点／成功失败案例是最低要求，不是项目上限。最终提交前须核对仓库内课程原件和第三方源码／license 整理情况。
