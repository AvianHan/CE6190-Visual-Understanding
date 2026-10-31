#!/usr/bin/env python3
"""Read-only resource inventory. Does not install packages or inspect secrets."""
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def probe():
    result = {"probe_version": 1, "python": sys.version.split()[0],
              "os": platform.system(), "architecture": platform.machine(),
              "cpu_count": os.cpu_count(), "packages": {},
              "commands": {name: shutil.which(name) is not None for name in
                           ("nvidia-smi", "nvcc", "conda", "uv", "git", "sbatch", "sinfo", "squeue")}}
    for package in ("torch", "torchvision", "transformers", "vllm", "numpy", "Pillow",
                    "opencv-python", "scipy", "timm", "xformers", "flash-attn"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        result["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        result["ram_bytes"] = None
    usage = shutil.disk_usage(Path.cwd())
    result["working_filesystem_bytes"] = {"total": usage.total, "free": usage.free}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    result["cuda_visible_devices_declared_count"] = (
        None if visible is None else 0 if visible.strip() in ("", "-1")
        else len([item for item in visible.split(",") if item.strip()]))
    result["gpu_inventory"] = []
    if result["commands"]["nvidia-smi"]:
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader,nounits"], text=True,
                                  capture_output=True, timeout=15)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    name, memory, driver = [part.strip() for part in line.rsplit(",", 2)]
                    result["gpu_inventory"].append({"name": name, "memory_mib": int(memory),
                                                    "driver_version": driver})
            else:
                result["gpu_probe_error"] = "nvidia-smi returned nonzero"
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            result["gpu_probe_error"] = type(exc).__name__
    result["notes"] = ["Run inside the intended compute allocation, not only on a login node.",
                       "GPU inventory alone does not establish allocated or usable GPUs.",
                       "No CUDA/PyTorch execution test was performed.",
                       "Disk free space is not a project storage quota."]
    return result


if __name__ == "__main__":
    print(json.dumps(probe(), ensure_ascii=False, indent=2))
