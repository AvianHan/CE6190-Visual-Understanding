#!/usr/bin/env python3
"""Plan (default), submit, or execute dependency-aware ICS experiment suites."""
import argparse
import json
import os
import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.orchestration import build_plan, write_plan, plan_summary, submit_slurm, run_local, run_preflight, run_task, collect, load_json


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}; closing owned processes")
    signal.signal(signal.SIGTERM, interrupted)
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        p = argparse.ArgumentParser()
        p.add_argument("worker")
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--phase", choices=["preflight", "smoke", "experiment", "collect"], required=True)
        p.add_argument("--resource")
        p.add_argument("--index", type=int)
        p.add_argument("--offset", type=int, default=0)
        a = p.parse_args()
        plan = load_json(a.plan)
        if a.phase == "preflight":
            run_preflight(plan)
        elif a.phase == "collect":
            print(json.dumps(collect(plan), indent=2))
        else:
            tasks = [t for t in plan["smoke" if a.phase == "smoke" else "tasks"] if t["resource"] == a.resource]
            index = a.index if a.index is not None else a.offset + int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
            if not 0 <= index < len(tasks):
                p.error("Task index out of range")
            run_task(plan, tasks[index])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=Path("configs/suite.initial.json"))
    p.add_argument("--cluster", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    p.add_argument("--gpus", help="Local execution only: explicit allocated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--submit", action="store_true", help="Actually submit/execute; otherwise only write and show plan")
    a = p.parse_args()
    plan = build_plan(a.suite, a.cluster, a.run_dir)
    path = write_plan(plan)
    print(json.dumps(plan_summary(plan), indent=2))
    if a.submit:
        if a.backend == "slurm":
            result = submit_slurm(path)
        else:
            if not a.gpus:
                p.error("Local execution requires --gpus explicitly")
            result = run_local(path, [x.strip() for x in a.gpus.split(",") if x.strip()])
        print(json.dumps(result, indent=2))
        if result.get("errors"):
            raise SystemExit(1)

if __name__ == "__main__":
    main()
