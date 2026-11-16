"""Independent CPU integration checks; fake Slurm and synthetic backend only.

These exercise the real adapter/runner/analyzer and actual detached manager
process boundaries. They do not validate model quality, GPU execution or Orion.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from ics.adapters.cgics import CGICSAdapter


ROOT = Path(__file__).resolve().parents[1]


def _fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_FIXTURE = _fixture_module("batch2_review_model_fixture", "test_cgics_batch2.py")
ANALYSIS_FIXTURE = _fixture_module("batch2_review_analysis_fixture", "test_batch2_runner_analysis.py")


class _RectangularBackend(MODEL_FIXTURE.FakeBackend):
    def final(self, *args):
        return np.pad(super().final(*args), ((0, 0), (0, 1)))


class Batch2ReviewIntegrationTests(unittest.TestCase):
    def _adapter_runner_analysis(self, variant):
        fixture = ANALYSIS_FIXTURE.Batch2RunnerAnalysisTests()
        fixture.setUp()
        try:
            fixture.config.update(experiment_id="review_" + variant,
                                  variant=variant, method="cgics",
                                  adapter={"variant": variant})
            result = fixture.run_fake(adapter_factory=lambda config:
                CGICSAdapter(config, _backend=_RectangularBackend()))
            self.assertEqual(result["new_ok"], 3)
            fixture.plan()
            report = fixture.analyze()
            self.assertEqual(report["formal_ok"], 3)
            self.assertEqual(report["status"], "COMPLETE")
            if variant in ("agreement_gate", "random_gate"):
                for row in fixture.rows():
                    self.assertEqual(set(row["alternative_predictions"]),
                                     {"gate_text", "gate_visual"})
                self.assertEqual(report["configuration_results"][0]["gate_observations"], 3)
            if variant == "fixed_pool_scores":
                self.assertEqual(len(report["configuration_results"]), 3)
                self.assertEqual(report["observed_cost_points"], [])
        finally:
            fixture.tearDown()

    def test_fixed_pool_real_adapter_runner_analysis(self):
        self._adapter_runner_analysis("fixed_pool_scores")

    def test_agreement_real_adapter_runner_analysis(self):
        self._adapter_runner_analysis("agreement_gate")

    def test_random_real_adapter_runner_analysis(self):
        self._adapter_runner_analysis("random_gate")

    def test_score_gate_real_adapter_runner_analysis(self):
        self._adapter_runner_analysis("initial_score_gate")

    def test_actual_detached_start_duplicate_stop_restart_retains_jobs(self):
        with tempfile.TemporaryDirectory(prefix="batch2-review-lifecycle-") as temp:
            top = Path(temp)
            binaries, runtime, run = top / "bin", top / "runtime", top / "run"
            binaries.mkdir()
            runtime.mkdir()
            database = top / "scheduler.json"
            fake_slurm = """#!/usr/bin/env python3
import fcntl,json,os,sys
from pathlib import Path
p=Path(os.environ["ICS_REVIEW_DB"])
with p.with_suffix(".lock").open("a+") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 rows=json.loads(p.read_text()) if p.exists() else []
 tool=Path(sys.argv[0]).name
 if tool=="sbatch":
  jid=str(500+len(rows)); name=sys.argv[sys.argv.index("--job-name")+1]
  rows.append({"id":jid,"name":name}); p.write_text(json.dumps(rows)); print(jid)
 elif tool=="squeue":
  for r in rows: print(r["id"]+"|"+r["name"]+"|PENDING|highpriority")
"""
            for name in ("squeue", "sacct", "sbatch"):
                path = binaries / name
                path.write_text(fake_slurm)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       ICS_REVIEW_DB=str(database))
            base = [sys.executable, "-I", "-B", str(ROOT / "scripts/batch2_ics.py")]
            start = base + ["start", "--runtime", str(runtime), "--run-dir", str(run),
                            "--interval", "10"]
            metadata = run / "manager.json"
            children = []

            def call(command):
                result = subprocess.run(command, env=env, text=True,
                                        capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr)
                return json.loads(result.stdout)

            def wait_for(condition):
                end = time.monotonic() + 15
                while time.monotonic() < end:
                    if condition():
                        return
                    time.sleep(.05)
                self.fail("Detached manager condition timed out")

            def stop(pid):
                self.assertEqual(json.loads(metadata.read_text())["pid"], pid)
                os.kill(pid, signal.SIGTERM)
                wait_for(lambda: json.loads(metadata.read_text()).get("exit") is not None)

            try:
                first = call(start)
                self.assertEqual(first["manager"], "STARTED")
                children.append(first["manager_pid"])
                wait_for(lambda: database.exists() and len(json.loads(database.read_text())) == 3)
                second = call(start)
                self.assertEqual(second["manager"], "ALREADY_RUNNING")
                self.assertEqual(len(json.loads(database.read_text())), 3)
                stop(first["manager_pid"])
                stopped = call(base + ["status", "--run-dir", str(run)])
                self.assertEqual(stopped["manager"], "STOPPED")
                self.assertEqual(stopped["batch2_status"], "MANAGER_STOPPED")
                restarted = call(start)
                self.assertEqual(restarted["manager"], "STARTED")
                children.append(restarted["manager_pid"])
                wait_for(lambda: json.loads(metadata.read_text()).get("phase") == "waiting")
                self.assertEqual(len(json.loads(database.read_text())), 3)
                state = call(base + ["status", "--run-dir", str(run)])
                self.assertEqual(state["manager"], "ACTIVE")
                self.assertEqual(state["counts"]["SUBMITTED"], 3)
                stop(restarted["manager_pid"])
            finally:
                # Only stop the exact test manager, never an unrelated recycled PID.
                for pid in children:
                    try:
                        command = Path(f"/proc/{pid}/cmdline").read_bytes()
                        if str(run / "source/scripts/batch2_ics.py").encode() in command:
                            os.kill(pid, signal.SIGTERM)
                    except (FileNotFoundError, ProcessLookupError):
                        pass


if __name__ == "__main__":
    unittest.main()
