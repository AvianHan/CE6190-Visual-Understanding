# LVIS 空源 mask：诊断与恢复（2026-11-04）

`ics-initial-20261102-pythonfix` 已终止，不能靠继续等待容量或重复旧 start 恢复。11 月 4 日回传的完整输出显示：

| 阶段 | 结果 |
|---|---|
| data | PASS 1 |
| preflight | FAILED 1；Slurm job 187040，exit 1:0 |
| smoke | BLOCKED 36 |
| experiment | BLOCKED 1,484 |
| collect | PASS 1，只表示完成失败汇总 |

总体为 `STOPPED`／`COMPLETE_WITH_FAILURES`／`terminal=true`，PASS 2、FAILED 1、BLOCKED 1,520，共提交 3 个作业。没有 GPU smoke 或正式实验结果。

preflight 已成功生成并验证 LVIS fold4 standard 的 2,300 episodes，SHA-256 为 `83d18e785ad81a55494d4f9f77804f160bd5764e995281dd6c1f691f2603845c`。随后生成 fold4 reference（500 queries × 每 query 50 个 reference sets，1-shot，seed/query-seed 0）时失败：

```text
ValueError: Selected source mask is empty: /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/data/LVIS/coco/train2017/000000578055.jpg
```

失败发生在 reference mask 构造；日志没有提供该 image／concept 的实际 pickle 内容，所以不能断言空标注产生的原因。preflight 时间戳为 `1793622940.253583`–`1793624697.7603624`，对应新加坡时间 **11 月 2 日 20:35:40–21:04:57**；11 月 4 日是证据回传日期。全部历史证据保存在 [项目维护记录](../PROJECT_LOG_before_cleanup.md) 的 2.13。

## 修复与实验协议

新增显式 `--empty-mask-policy exclude`，仅用于 LVIS；默认 `error` 保留原严格行为。固定原类别集合和 fold 划分后，扫描所有 image／concept 候选。只有正确解码、尺寸正确但合并后前景为零的源 mask 可以被排除；其他解码错误、尺寸不符、文件缺失或损坏缓存仍会失败。筛选不能让原类别消失，也不能悄悄重划 folds；样本不足时明确报错。

query 和 reference 共享同一有效池，保持确定性及不同 reference budget 的 query 对齐。每个 protocol sidecar 记录 `sampling.empty_mask_policy` 和 `sample_eligibility`，后者包含策略／版本、输入／有效／排除数量、逐类别计数、完整 `excluded_samples` 与 `pool_fingerprint`。排除数量要由真实服务器扫描确定，不能凭本次 traceback 认定只有一个。新筛选改变采样总体，应作为显式实验协议披露，不能宣称精确复现作者原始随机采样。

新 suite `configs/suite.initial.lvisfix.json` 保留原实验范围：36 配置、42 manifests、36 smoke、1,484 正式分片，加 data／preflight／collect 共 1,523 个任务。

| 资源 | 恢复方式 |
|---|---|
| 12 个 COCO manifests | 原路径和 recipe 不变，通过身份检查后复用 |
| 30 个 LVIS manifests | 全部使用 `${ICS_RUNTIME}/manifests/lvis-nonempty-v2/`，按统一有效池生成 |
| 旧 LVIS manifests、旧 run 状态／日志／冻结源码 | 保留，不覆盖、不手改 FAIL／BLOCKED，不混入新协议结果 |
| 图像、有效 archives／receipts、生成 mask 缓存、三个模型环境、四份权重 | 继续复用；data 阶段仍验证可复用状态，缺失项才需要传输 |

preflight 检查已有 manifest 的 policy／协议身份；旧 sidecar 未声明 policy 只视为严格 `error`。因此不把旧 LVIS standard 的成功文件拼接进新 reference 有效池。

## 本地应用与手动 commit／push

将 `CE6190-ICS-lvis-fix-20261104.zip` 下载到 Mac 的 `~/Downloads`，在已有本地项目仓库目录粘贴这一条命令：

```bash
(ICS_FIX_DIR="$(mktemp -d)" && unzip -q "$HOME/Downloads/CE6190-ICS-lvis-fix-20261104.zip" -d "$ICS_FIX_DIR" && python3 "$ICS_FIX_DIR/CE6190-Visual-Understanding/scripts/apply_update.py" --target "$PWD" --apply)
```

更新器校验包、备份被替换的文件并保留 Git 与其他文件；不会自动 commit／push。随后在 VS Code Source Control 检查 diff，使用以下 commit message 手动 commit／push：

```text
fix: audit empty LVIS masks and isolate recovery manifests
```

## Orion 启动新 run

本地 push 完成后，在 Orion 粘贴以下完整命令。它从仓库冻结更新后的源码，建立独立恢复 run；依赖关系仍是 data → preflight → smoke → experiment → collect。

```bash
(
    set -euo pipefail
    cd /scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding
    git pull --ff-only
    export ICS_RUNTIME="/scratch/prj0000000262_1/hansy/ce6190-ics-runtime"
    unset PYTHONPATH PYTHONHOME
    export PYTHONNOUSERSITE=1

    "$ICS_RUNTIME/envs/insid3/bin/python" -I -u scripts/queue_ics.py start \
        --runtime "$ICS_RUNTIME" \
        --suite configs/suite.initial.lvisfix.json \
        --cluster configs/cluster.orion.json \
        --run-dir "$ICS_RUNTIME/runs/ics-initial-20261104-lvisfix"
)
```

不要用旧终态 run 名称启动修复。新 run 继续使用既有共享 60、ICS 最多 8 在途的策略，不取消其他项目作业。

## 查看进度并回传

直接运行以下只读命令，贴回完整 JSON：

```bash
(
    ICS_RUN="/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-initial-20261104-lvisfix"
    /scratch/prj0000000262_1/hansy/ce6190-ics-runtime/envs/insid3/bin/python -I \
        "$ICS_RUN/source/scripts/queue_ics.py" status --run-dir "$ICS_RUN"
)
```

新版输出包含 `stage_counts`；`attention` 最多显示 10 项，真正 FAILED／UNCERTAIN／WAITING_QUOTA 优先，包含 kind、resource、job_id，另给 `attention_total` 和 `attention_truncated`。全部任务仍保存在 `rolling_state.json`。终态 STOPPED 不再建议重复 start；非终态停机仍保留续跑提示。

`STARTED` 仅表示管理器启动；`WAITING_CAPACITY` 且 manager ACTIVE 表示继续等共享名额；必须看到 preflight PASS、smoke PASS 和正式分片结果，才能逐步确认真实运行成功。若 preflight 再失败，依据新的具体任务错误处理，不重置旧状态或静默丢弃异常。

## 验证边界

本次本地完整测试运行 189 项：**181 通过，8 项因缺少 torch／torchvision 跳过，0 failures／errors**，耗时 1.700 秒。23 项 data、18 项 orchestration、5 项 prepare CLI、58 项 queue／CLI 测试全部通过；`pycocotools 2.0.10` 实际解码合成 polygon、压缩及未压缩 RLE，覆盖非空池、排除审计、类别／fold 保留、采样对齐、解码与缓存错误继续失败。报告见 [validation_2026-11-04_lvis_fix.json](../research/validation_2026-11-04_lvis_fix.json)。

真实 venv 子进程还在本地合成 LVIS fold4（空 mask 1、有效样本 3）两次执行 `run_preflight`，standard／reference／component 的 query 前缀与有效池一致，第二次 hash／mtime 不变，venv 解释器路径保留。更新器已验证备份字节匹配，保留 Git、未列入更新的文件及前次 Python 修复，重复应用返回 `UP_TO_DATE`；增量包共 16 个 payload 文件加 1 份清单。

8 项跳过的是可选模型 tensor 测试。上述本地 CPU／合成和模拟调度结果不等于已在 Orion 部署、新 run 已提交、全部真实 LVIS 样本通过筛选或 GPU 推理通过；这些需后续实际日志证明。
