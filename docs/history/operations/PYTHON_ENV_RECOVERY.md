# Orion Python 环境路径修复：2026-11-02

数据任务 `185464` 使用了基础 Python，而不是配置中的 `envs/insid3/bin/python`，因此四份 Google Drive 标注／split 下载报 `No module named gdown`。原调度器对可执行文件调用 `Path.resolve()`，把虚拟环境的 Python 符号链接解析成了基础解释器。修复同时覆盖所有资源配置的 `python` 和自管 vLLM 的 `mllm_python`；模型／数据路径仍采用原来的规范化方式。

旧 run `ics-initial-20261101` 已终止：1 FAILED、1521 BLOCKED、1 PASS。PASS 属于失败汇总任务，不代表模型成功。GPU smoke／正式实验尚未运行。日志中的四组 COCO 图像已解压，恢复时通过同一 data 目录的已有 ZIP 和 extraction receipts 校验复用；不删除这些文件。

## 本地应用与手动提交

下载 `CE6190-ICS-python-fix-20261102.zip` 到 Mac 的 `~/Downloads`，在本地 **CE6190-Visual-Understanding 仓库**的 VS Code 终端执行：

```bash
(ICS_FIX_DIR="$(mktemp -d)" && unzip -q "$HOME/Downloads/CE6190-ICS-python-fix-20261102.zip" -d "$ICS_FIX_DIR" && python3 "$ICS_FIX_DIR/CE6190-Visual-Understanding/scripts/apply_python_fix.py" --target "$PWD" --apply)
```

更新器先核对包内文件哈希和实际 `ics/orchestration.py` 的源码结构，再只修改解释器路径表达式。遇到不认识或只修了一半的源码会在任何修改前停止；不会用缺少上下文的完整模块覆盖仓库。所有被替换文件先备份到仓库同级的 `CE6190-Visual-Understanding-update-backups`。重复应用可安全识别已有修复，其他文件与 Git 元数据保留。

随后由用户在 VS Code 检查 diff、手动 commit／push。Commit message：

```text
fix: preserve virtualenv Python paths and precheck data dependencies
```

## Orion 恢复

完成本地 push 后，在 Orion 执行下面整个代码块。它检查真实仓库生成的资源配置，只有 `insid3` 确实没有 `gdown` 时才补装该依赖，然后检查下载依赖并启动新 run。数据下载、preflight、smoke 和正式实验仍交给已有滚动队列在 Slurm 上执行。

```bash
(
    set -euo pipefail
    cd /scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding
    git pull --ff-only
    export ICS_RUNTIME="/scratch/prj0000000262_1/hansy/ce6190-ics-runtime"
    unset PYTHONPATH PYTHONHOME
    export PYTHONNOUSERSITE=1
    ICS_PY="$ICS_RUNTIME/envs/insid3/bin/python"

    "$ICS_PY" -I - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from ics.orchestration import resource_profile
runtime = Path(os.environ["ICS_RUNTIME"])
assert sys.prefix != sys.base_prefix, "当前 Python 未进入虚拟环境"
cluster = json.loads(Path("configs/cluster.orion.json").read_text())
for name, env in (("preflight", "insid3"), ("insid3", "insid3"),
                  ("cgics_visual", "cgics"), ("cgics_mllm", "cgics")):
    profile = resource_profile(cluster, name)
    expected = str(runtime / "envs" / env / "bin/python")
    assert profile["python"] == expected, (name, profile["python"])
    if name == "cgics_mllm":
        expected_mllm = str(runtime / "envs/mllm/bin/python")
        assert profile["mllm_python"] == expected_mllm, profile["mllm_python"]
    print(name, "Python 路径检查通过")
PY

    if ! "$ICS_PY" -I -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("gdown") is None)'; then
        "$ICS_PY" -I -m pip install 'gdown>=5,<6'
    fi
    "$ICS_PY" -I -u scripts/download_data.py \
        --datasets coco lvis --data-root "$ICS_RUNTIME/data" \
        --check-dependencies-only
    "$ICS_PY" -I -u scripts/queue_ics.py start \
        --runtime "$ICS_RUNTIME" \
        --suite configs/suite.initial.json \
        --cluster configs/cluster.orion.json \
        --run-dir "$ICS_RUNTIME/runs/ics-initial-20261102-pythonfix"
)
```

新目录很重要：旧 run 有冻结源码、旧计划和终态记录，单纯 pull 后重启旧 run 不能修复它。这里保留旧 run 作为失败证据，不改状态文件、不取消其他项目作业、不重新下载模型。继续使用原共享 60 名额、ICS 最多 8 在途的配置。启动成功不代表 GPU 验证通过；数据依赖通过后才会继续。

## 本次验证边界

本地针对性测试 **37 项通过，0 跳过、0 失败、0 错误**：23 项 downloader 测试、14 项解释器路径／更新器测试。包括真实符号链接虚拟环境中的 sentinel 模块导入、worker 与 vLLM 路径保留、资产路径规范化、缺依赖时提前停止、缓存复用、更新备份与重复应用。

本次可用源码来自此前已交付的 rolling 更新包和此前读取的原始函数片段；当前无法重新取得完整 Git 仓库。解释器测试明确使用原始片段 fixture，不冒充完整项目测试。此前的 140 项测试是历史结果，本次未重跑全套。尚未在 Orion 应用此修复、提交新作业或验证 GPU。报告见 `research/validation_2026-11-02_python_fix.json`，持续记录见 `CE6190_ICS_PROJECT_LOG.md`。

Python 官方说明：[Python 3.12 venv](https://docs.python.org/3.12/library/venv.html)。虚拟环境包含自己的 `pyvenv.cfg` 和解释器入口，`sys.prefix` 与 `sys.base_prefix` 可用于判断当前解释器是否处于虚拟环境。
