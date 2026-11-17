# 第二批：SUIM 全池筛选恢复与 GPU 分区调度

这次更新解决两个独立问题：SUIM 数据准备被图像与标注尺寸不一致阻断，以及已有 GPU 作业只在原分区等待。它继续使用 `ics-batch2-20261116`，保留四部分研究计划、已有任务身份、原冻结模型源码和历史证据。

适用仓库基线为 `caf7be44042440a43d77876adf28a77158459f61`。2026 年 11 月 16 日 17:35 SGT 的直接证据是 SUIM data 作业 `206900` 因 `f_r_1289_.jpg` 的尺寸不一致失败，39 个 smoke 仍处于 Slurm `PENDING (Priority)`；这不能当作之后的实时状态，也不证明 GPU 已经运行。

## SUIM：先审计全池，再从有效 pair 取样

在取样前检查 SUIM 全池的原始图像与对应类别 mask。尺寸相同的有效 pair 才进入候选池；尺寸不一致的 pair 标为不合格，并保留其文件身份、原始尺寸及原因。策略身份为 `suim-strict-geometry-filter-v1`。**不 resize、不 crop，不修改原图或 GT。** 这不是恢复原标注几何关系，也不能写成完成了作者完整协议复现。

筛选不减少七个类别、1,000 个不同 query 的目标或 13 个实验配置；不能满足原计划的数量和 reference 约束时明确失败，不能靠重复 query 或静默降低目标继续。输出审计与 manifest 记录实际候选池及排除范围。后续 SUIM 结论适用于这个明确记录的有效尺寸 cohort，应连同覆盖率报告，不能隐藏被排除的原始 pair。

本次新增的排除原因仅为已解码确认的尺寸不一致；原空 mask 仍按作者的非空候选规则处理并单独计数。解码、缺失文件、非正常 EXIF 或源码身份问题仍明确失败。审计文件为 scope 下 `data/suim_geometry_receipt.json` 与 `data/suim_geometry_inputs-<auditsha>.json`，由 `data_receipt.json` 绑定。13 个配置共享同一份新 SUIM manifest。

恢复控制器核对预期 SUIM plan、原冻结源码、失败作业和失败标记。它为修复建立独立冻结源码 `suim-repair-20261116/source`，只让 SUIM 的 data/preflight 使用修复路径；模型与 collector 继续使用原冻结实验源码。原失败 data 和空 collector 的状态、标记及输出先归档；仅恢复这两项并解除其后 170 项从未执行任务的阻断，不重跑成功模型。第一批及 core/ISIC 的科学结果不被清空或重建。

## 调度：保持资源配置，使用已核实可访问的分区

调度策略独立于不可变实验 plan，绑定每个 scope 的原计划。先读取当前分区状态、账户/QoS 访问规则和额度，再准备策略，之后才暂停旧管理器。共享在途上限不高于继承的 60，也不高于实时核实的限制；这是提交窗口，不等于 60 个作业都已获得 GPU。

| 任务 | 调度行为 | 保留的资源边界 |
|---|---|---|
| 新提交 smoke | 使用通过访问检查的 `l40sp`，时限 1 小时 | CG-ICS 仍为 2 GPU；INSID3 仍为 1 GPU |
| 正式任务 | 对应配置、对应数据集的 L40S smoke 经 Slurm `COMPLETED/0:0` 核实后，可使用 `h200p,l40sp`；此前保留原 H200 路径 | CG-ICS 的独立语言模型 GPU、CPU/内存配置和 `highpriority` 保持 |
| 已提交且仍 PENDING 的任务 | 严格核对身份后，仅通过 `scontrol` 更新 `Partition=l40sp` | 原 job ID、时限和 worker 命令保持；不取消后重提 |
| RUNNING、已完成或身份不确定任务 | 不迁移、不猜测成功、不重复提交 | 保留原作业及证据 |

原 GPU 节点排除保留：所有 GPU profile 排除 `A484652X3B10601`，CG-ICS 另排除 `2FS9B74`。这是继承的故障规避记录，不是对节点当前硬件状态的新判断。

迁移 PENDING 作业要求保存状态仍为 `SUBMITTED`、唯一精确 job name、最新 `squeue` 与即时 `scontrol` 均确认 PENDING、用户/账户/QoS 匹配、原 intent/plan/source/task/attempt 身份一致、worker 命令不可变且尚无执行 claim。检查失败或排队状态发生竞争时记录为跳过/延后；不会修改 RUNNING 作业，也不会取消、重提或生成新的 job ID。迁移前后的调度证据保存在各 scope 的 `scheduling/updates` 下。

## Mac 应用

使用交付 ZIP 的 SHA256 校验、解压及 `apply_suim_repair_update.py --target "$PWD" --apply` 命令。安装器逐文件验证基线与新文件哈希，先备份再应用；重复执行返回 `UP_TO_DATE`，保留未列出的用户文件，不运行 Git 操作。用户检查 diff 后手动 commit/push。

建议 commit message：

```text
Fix SUIM data validation and enable batch-2 H200/L40S scheduling
```

## Orion：一次激活并继续后台运行

用户完成 Mac commit/push 后，在 **Orion 登录终端** 执行：

```bash
(
    set -e
    cd /scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding
    git pull --ff-only
    export ICS_RUNTIME=/scratch/prj0000000262_1/hansy/ce6190-ics-runtime
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
    export PATH=/cm/shared/apps/slurm/current/bin:$PATH
    ICS_RUN="$ICS_RUNTIME/runs/ics-batch2-20261116"
    "$ICS_RUNTIME/envs/insid3/bin/python" -I -B -u \
        scripts/recover_batch2_suim.py activate --run-dir "$ICS_RUN"
)
```

`activate` 自带失败证据和调度规则检查，不需要用户手改状态或先执行 `scancel`。只有完成这些检查后，才用已验证的 PID/pidfd 向当前管理器请求停止继续提交；现有 Slurm 作业保留。随后在锁保护下完成 SUIM 恢复事务，启动脱离 SSH 的新管理器，继续依照 data/preflight/逐配置逐数据集 smoke 的依赖自动推进正式实验。

看到 `STARTED_RECOVERY`/`STARTING_RECOVERY` 表示后台管理器已启动/正在启动，不表示数据或 GPU smoke 已通过。重复同一命令会识别已经运行的修复管理器，或继续尚未完成的恢复事务；不要改 plan、删除标记或新建另一份 run。

如果队列已被核实为终态，重复激活返回 `manager_start: NOT_NEEDED`，不会重启管理器或重跑任务：`COMPLETE` 退出码为 0，`COMPLETE_WITH_FAILURES` 退出码为 2。后者保留失败供审计，不会把失败清成成功或自动重试。

如果旧管理器已经停止、但恢复事务随后报错，现有 Slurm 作业仍继续，新提交会保持暂停。完整输出会明确说明这一状态；执行同一条 `activate` 命令继续事务，直到看到修复管理器的启动/已运行回执，或经核实的终态 `NOT_NEEDED`。不要把一次中途失败当成后台恢复已经完成。

只读核对可使用：

```bash
(
    ICS_RUNTIME=/scratch/prj0000000262_1/hansy/ce6190-ics-runtime
    ICS_RUN="$ICS_RUNTIME/runs/ics-batch2-20261116"
    ICS_REPO=/scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
    export PATH=/cm/shared/apps/slurm/current/bin:$PATH
    TZ=Asia/Singapore date
    timeout -k 2s 45s "$ICS_RUNTIME/envs/insid3/bin/python" -I -B -u \
        "$ICS_REPO/scripts/recover_batch2_suim.py" status --run-dir "$ICS_RUN"
    printf 'status_exit=%s\n' "$?"
    timeout -k 2s 20s squeue -r -u "$(id -un)" \
        -o '%.18i %.16P %.12T %.12M %.64j %R'
    printf 'squeue_exit=%s\n' "$?"
    tail -n 40 "$ICS_RUN/manager.log"
)
```

`status` 展示保存的最近队列观察与管理器状态；实时 Slurm 队列单独解释。`audit --run-dir "$ICS_RUN"` 可在激活前只读检查预期失败证据。`collect PASS` 不等于全部模型成功，PENDING 也不是实际 GPU 运行。分区放宽能增加可调度机会，不能保证集群马上分配 GPU 或保证具体完成时间。

## 验证边界

交付的验证记录分别覆盖 SUIM 全池筛选、幂等恢复与历史保留、真实本地子进程/模拟 Slurm 的继承调度，以及 Mac 更新包的基线、备份、回滚和实际 ZIP 应用。它们不是 Orion GPU 实测结果。真实 SUIM 有效池覆盖率、data/preflight 和各数据集 smoke 仍由激活后的服务器证据确认；失败原因与已成功实验必须保留。
