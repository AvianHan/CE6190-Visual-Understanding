#!/usr/bin/env python3
"""Read-only cluster inventory; never allocates GPUs or prints credential variables."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def bounded(command, timeout=15):
    if not shutil.which(command[0]):
        return {"available": False, "command": command}
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"available": True, "command": command, "returncode": p.returncode,
                "stdout": p.stdout[:30000], "stderr": p.stderr[:5000]}
    except subprocess.TimeoutExpired:
        return {"available": True, "command": command, "timeout_seconds": timeout}


def probe():
    user = os.environ.get("USER", "")
    commands = [
        ["sinfo", "-h", "-o", "%P|%a|%l|%D|%G|%m|%c"],
        ["scontrol", "-o", "show", "partition"],
        ["scontrol", "show", "config"],
        ["sacctmgr", "-n", "-P", "show", "assoc", "where", f"user={user}",
         "format=Account,User,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs,MaxWall"],
        ["sacctmgr", "-n", "-P", "show", "qos", "format=Name,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU"],
        ["squeue", "-u", user, "-o", "%.18i %.12P %.20j %.12T %.12M %.12l %R"],
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        ["df", "-h", str(Path.cwd())],
        ["quota", "-s"],
        ["python3", "--version"], ["python3.10", "--version"], ["python3.12", "--version"],
        ["uv", "--version"], ["conda", "--version"],
        ["bash", "-lc", "if type module >/dev/null 2>&1; then module -t avail 2>&1 | awk 'tolower($0) ~ /cuda|python|conda/'; else echo 'environment modules unavailable'; fi"],
    ]
    results = [bounded(c) for c in commands]
    for result in results:
        if result["command"] == ["scontrol", "show", "config"] and "stdout" in result:
            result["stdout"] = "\n".join(line for line in result["stdout"].splitlines()
                if re.match(r"\s*(MaxArraySize|MaxJobCount|ClusterName|SchedulerType|SelectType|AccountingStorageEnforce)\s*=", line))
    return {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": platform.node(), "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "allocated_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "executables": {name: shutil.which(name) for name in ("python3", "python3.10", "python3.12", "uv", "conda", "git", "curl", "wget", "rsync", "ssh", "sbatch", "srun")},
            "results": results,
            "interpretation": ["Login-node inventory does not establish allocated compute resources.",
                               "Visible GPUs are not automatically available allocations.",
                               "Fill account/partition/qos/time and concurrency from real association/QOS limits.",
                               "Disk free is not a project quota; kernel and model execution are checked by smoke jobs."]}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    value = json.dumps(probe(), indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(value + "\n")
    print(value)
