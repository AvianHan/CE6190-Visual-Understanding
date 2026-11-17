# 管理器停止：本地审计与原 run 只读诊断

最新明确指向 `ics-initial-20261104-lvisfix` 的输出是 **manager STOPPED**，但磁盘仍保存 data PASS、preflight RUNNING。`updated_at=1793764423.4704173` 对应新加坡时间 **2026-11-04 11:53:43.470417**；这是状态保存时间，不是已经确认的退出时间。

当前缺少 manager.log 的退出证据，**实际停止原因未知**。已提交的 preflight 由 Slurm 管理，管理器停止不等于它已取消；它此刻仍运行、已完成或已失败，需要查询现有作业。此前 preflight 187040 的空 mask 错误属于旧 pythonfix run，不能作为这次 lvisfix 管理器退出的解释。

## 本轮审计与修复

本轮直接基于真实仓库 `9f01d28f324a57522b3b904c591b1fabb45479fd`，保留既有 Python 路径、LVIS 非空池和独立 manifest 目录修复。局部审计发现并处理以下边界；这些可复现缺陷**尚未被服务器日志确认为此次退出原因**。

| 边界 | 修复行为 |
|---|---|
| completion marker 是合法 JSON 但不是 object | 先核验结构／身份，避免 `.get()` 的 `AttributeError` 直接终止 watcher；异常标记不当作成功 |
| 停止信号抵达一轮提交途中 | 后续提交边界响应停止；已拿到的 job ID 仍保存，恢复不因丢失回执而重复提交 |
| 管理器退出缺少记录 | 新 watcher 写生命周期阶段、心跳／轮询时间及可捕获的退出原因；诊断写失败不覆盖原始错误 |
| 旧退出记录被误作当前原因 | 记录早于最新队列观察，或时间不能可靠比较时，只保留历史诊断，当前退出原因仍为 unknown |
| 启动就绪检查读取旧／损坏记录 | 只接受 object 且 PID 匹配新子进程的就绪证据；不把无效旧记录当作新子进程失败，实际提前退出仍报错 |
| manager STOPPED 但主状态仍显示旧 RUNNING | 改为 `MANAGER_STOPPED`，另保留 `saved_queue_status`、`state_age_seconds`，明确不代表实时 Slurm 状态 |
| 复用 manifest 只比较部分 recipe | 补 dataset、split、reference 不足处理等身份核验，拒绝同路径另一协议 |
| sidecar 已发布、manifest 尚未写入就中断 | 仅在重新计算出的完整 metadata／身份匹配时补齐缺失 manifest；不覆盖不匹配 sidecar |

旧 manager 缺少新生命周期字段时，退出原因仍为 unknown。不可捕获的中止也不能保证留下退出记录，不能把缺记录本身解释为某种确定原因。

## 验证顺序与范围

用户已明确要求以后所有代码更新遵循：**小范围逻辑测试 → 整体测试／真实子进程集成 → 确认后再交付服务器动作**。具体缺陷先证明修复前失败、修复后通过，覆盖正常、异常、停止与恢复路径，而不是只测正常分支。

首次 218 项整体检查通过后，独立复核又找到旧退出记录和启动就绪检查的边界；继续先复现失败、修复后通过，再完整复跑。**最终 227 项全部通过，0 failures、0 errors、0 skips，13.517 秒**；包含 2 项真实 manager／worker 子进程＋fake Slurm 端到端测试，不额外相加。报告见 [validation_2026-11-04_manager_audit.json](../research/validation_2026-11-04_manager_audit.json)，完整输出见 [validation_2026-11-04_manager_audit.tests.txt](../research/validation_2026-11-04_manager_audit.tests.txt)。

小范围检查包括生命周期 13、CLI 17、rolling 47、orchestration 23、prepare 10、adapter 23、diagnose 7 项。隔离测试环境为 `torch 2.7.1+cpu`、`torchvision 0.22.1+cpu`、`pycocotools 2.0.10`；此前跳过的 tensor 分支和固定 INSID3 上游处理源码＋合成 encoder 的输出等价性均已补测。dry plan 保留 42 manifests（12 COCO／30 LVIS）、36 smoke、1,484 正式分片，含 data／preflight／collect 共 1,523 项；venv 解释器路径保留，未提交真实作业。

fake Slurm 不是真实 Orion 调度；CPU 合成 encoder 不是真实权重 GPU 推理；本地小样本不代表真实全量 LVIS 已通过。当前 36 个模型 smoke 都取 COCO fold0 的前 2 个 episodes，没有真实 LVIS 模型 smoke；不能用其通过来代替真实 LVIS 全量数据或 GPU 验证。测试也不能替代本次服务器退出日志。

## 本地应用与手动 commit／push

把 `CE6190-ICS-manager-audit-20261104.zip` 下载到 Mac 的 `~/Downloads`，在已有本地仓库终端粘贴：

```bash
(ICS_FIX_DIR="$(mktemp -d)" && unzip -q "$HOME/Downloads/CE6190-ICS-manager-audit-20261104.zip" -d "$ICS_FIX_DIR" && python3 "$ICS_FIX_DIR/CE6190-Visual-Understanding/scripts/apply_update.py" --target "$PWD" --apply)
```

在 VS Code 检查 diff，手动 commit／push；commit message：

```text
fix: harden queue lifecycle and validate recovery paths
```

## Orion：同步后只读诊断现有 run

在 Orion 粘贴这一条命令。它从**更新后的仓库 clone**调用 `scripts/diagnose_queue.py`：运行新版只读 status，读取 manager.log、data／preflight 的保存状态、日志与 marker，再查询现有作业的 sacct；日志中的常见凭据形式会被遮蔽。不要换成旧 `run/source/` 中的脚本。命令不 start、不 reset、不创建新 run、不重新准备数据。

```bash
(cd /scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding && git pull --ff-only && /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/envs/insid3/bin/python -I scripts/diagnose_queue.py --run-dir /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-initial-20261104-lvisfix)
```

贴回这段完整输出后，结合实际退出信息与 preflight 状态决定恢复方式。pull 只更新仓库，不会改写现有 run 的 frozen source；相同 start 仍会复用旧冻结逻辑，不能当作已启用本次 manager 修复。当前保留原 source、状态、manifests、成功 data 和已提交作业；不立即 start、不覆盖 frozen source、不新建重复 run。
