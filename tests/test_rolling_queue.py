"""Synthetic Slurm failure/restart tests. No cluster or model is contacted."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ics.orchestration import build_plan, write_plan
from ics import rolling_queue as rq


class Scheduler:
    def __init__(self):
        self.queue = {}
        self.history = {}
        self.commands = []
        self.number = 200
        self.response = None

    def snapshot(self, state):
        return copy.deepcopy(self.queue), copy.deepcopy(self.history)

    def submit(self, command, **kwargs):
        assert command[0] == "sbatch", command
        self.commands.append(command)
        if self.response:
            if isinstance(self.response, BaseException):
                raise self.response
            return self.response
        self.number += 1
        jid = str(self.number)
        name = command[command.index("--job-name") + 1]
        self.queue[jid] = self.row(jid, name)
        return mock.Mock(returncode=0, stdout=jid + ";cluster\n", stderr="")

    @staticmethod
    def row(jid, name, status="PENDING", exit_code=None, qos="highpriority"):
        result = {"job_id": jid, "job_name": name, "state": status, "qos": qos}
        if exit_code is not None:
            result["exit_code"] = exit_code
        return result

    def occupied(self, number, qos="highpriority"):
        self.queue = {str(i): self.row(str(i), "generation-or-other-account", qos=qos)
                      for i in range(1, number + 1)}


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        experiments = []
        for name in ("a", "b"):
            cfg = self.root / f"{name}.json"
            cfg.write_text(json.dumps({"experiment_id": name, "method": "insid3",
                                       "variant": "full", "seed": 0, "adapter": {}}))
            experiments.append({"config": str(cfg), "resource": "insid3", "manifest_ids": ["coco"]})
        suite = self.root / "suite.json"
        suite.write_text(json.dumps({"schema_version": 1, "smoke_episodes": 1,
                                    "prepare_missing_manifests": True,
                                    "manifests": [{"id": "coco", "path": str(self.root / "manifest.json"),
                                                   "dataset_root": str(self.root / "data"),
                                                   "prepare": {"dataset": "coco", "fold": 0}}],
                                    "experiments": experiments}))
        cluster = self.root / "cluster.json"
        cluster.write_text(json.dumps({"schema_version": 1, "max_parallel": 2,
                                      "launcher_python": "/usr/bin/python3",
                                      "defaults": {"account": "account", "partition": "gpu", "qos": "highpriority",
                                                   "time": "00:30:00", "cpus": 2},
                                      "resources": {"preflight": {"python": "/usr/bin/python3", "gpus": 0},
                                                    "insid3": {"python": "/usr/bin/python3", "gpus": 1}}}))
        self.plan = build_plan(suite, cluster, self.root / "run")
        self.plan_path = write_plan(self.plan)
        self.state_path = self.root / "run" / rq.STATE_NAME
        self.state = rq.initialize_queue(self.plan_path)
        self.scheduler = Scheduler()
        p = mock.patch.object(rq, "_scheduler_snapshot", side_effect=self.scheduler.snapshot)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rq.subprocess, "run", side_effect=self.scheduler.submit)
        p.start(); self.addCleanup(p.stop)

    def read(self):
        return json.loads(self.state_path.read_text())

    def save(self, state):
        rq._atomic_json(self.state_path, state)

    def ready_smokes(self):
        state = self.read()
        for key in ("data", "preflight"):
            state["tasks"][key]["status"] = "PASS"
        self.save(state)

    def active(self, key="data", status="SUBMITTED", jid="100"):
        state = self.read()
        task = state["tasks"][key]
        task.update(status=status, job_id=jid, attempts=1)
        self.save(state)
        return task

    def marker(self, task, *, fingerprint=None, job_id=None, model=True):
        body = {"key": task["key"], "status": "PASS",
                "plan_fingerprint": fingerprint or self.plan["fingerprint"],
                "slurm_job_id": job_id or task["job_id"]}
        rq._atomic_json(rq._marker_path(self.plan, task["key"]), body)
        if task["kind"] in {"smoke", "experiment"} and model:
            body["task_id"] = task["key"]
            rq._atomic_json(self.root / "run/jobs" / task["key"] / "complete.json", body)

    def complete_record(self, task, status="COMPLETED", exit_code="0:0"):
        self.scheduler.history[task["job_id"]] = self.scheduler.row(
            task["job_id"], task["job_name"], status, exit_code)

    def test_initialize_is_idempotent_and_does_not_submit(self):
        again = rq.initialize_queue(self.plan_path)
        self.assertEqual(again, self.state)
        self.assertEqual(rq.queue_status(self.plan_path)["total"], 7)
        self.assertEqual(self.scheduler.commands, [])
        with self.assertRaisesRegex(ValueError, "different plan or policy"):
            rq.initialize_queue(self.plan_path, max_inflight=7)

    def test_static_submission_cannot_be_mixed(self):
        (self.root / "run/submission.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "static submission"):
            rq.initialize_queue(self.plan_path)

    def test_later_static_submission_pauses_the_existing_queue(self):
        (self.root / "run/submission.json").write_text("{}")
        self.assertEqual(rq.tick(self.plan_path)["status"], "PAUSED")
        self.assertEqual(self.scheduler.commands, [])

    def test_missing_state_with_previous_worker_artifacts_cannot_reinitialize(self):
        rq.tick(self.plan_path)
        self.state_path.unlink()
        with self.assertRaisesRegex(ValueError, "state is missing"):
            rq.initialize_queue(self.plan_path)

    def test_sixty_shared_jobs_prevents_submission(self):
        self.scheduler.occupied(60)
        result = rq.tick(self.plan_path)
        self.assertEqual(result["status"], "WAITING_CAPACITY")
        self.assertEqual(self.scheduler.commands, [])

    def test_one_free_shared_slot_submits_only_one_ready_smoke(self):
        self.ready_smokes()
        self.scheduler.occupied(59)
        result = rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        self.assertEqual(result["active"], 1)

    def test_other_qos_does_not_consume_same_qos_budget(self):
        self.scheduler.occupied(60, qos="normal")
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)

    def test_local_max_inflight_is_enforced(self):
        self.state_path.unlink()
        rq.initialize_queue(self.plan_path, max_inflight=1)
        self.ready_smokes()
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)

    def test_reserved_shared_slots_are_preserved(self):
        self.state_path.unlink()
        rq.initialize_queue(self.plan_path, reserved_slots=2)
        self.scheduler.occupied(58)
        rq.tick(self.plan_path)
        self.assertEqual(self.scheduler.commands, [])

    def test_resource_cap_counts_previously_submitted_work(self):
        self.ready_smokes()
        state = self.read()
        # One pending production job also consumes this family's allocation cap.
        state["tasks"]["a--coco--0"].update(status="SUBMITTED", job_id="101")
        state["tasks"]["smoke-a"].update(status="SUBMITTED", job_id="102")
        self.save(state)
        for key in ("a--coco--0", "smoke-a"):
            task = state["tasks"][key]
            self.scheduler.queue[task["job_id"]] = self.scheduler.row(task["job_id"], task["job_name"])
        rq.tick(self.plan_path)
        self.assertEqual(self.scheduler.commands, [])

    def test_intent_not_yet_visible_reserves_shared_slot(self):
        self.ready_smokes()
        state = self.read()
        state["tasks"]["smoke-a"]["status"] = "SUBMITTING"
        self.save(state)
        self.scheduler.occupied(59)
        rq.tick(self.plan_path)
        self.assertEqual(self.scheduler.commands, [])
        self.assertEqual(self.read()["tasks"]["smoke-a"]["status"], "UNCERTAIN")

    def test_running_success_marker_is_not_completion(self):
        task = self.active()
        self.marker(task)
        self.scheduler.queue["100"] = self.scheduler.row("100", task["job_name"], "RUNNING")
        self.complete_record(task)
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "RUNNING")
        self.assertEqual(self.scheduler.commands, [])

    def test_completion_requires_scheduler_zero_and_matching_marker(self):
        task = self.active()
        self.marker(task)
        self.complete_record(task)
        rq.tick(self.plan_path)
        state = self.read()
        self.assertEqual(state["tasks"]["data"]["status"], "PASS")
        self.assertEqual(state["tasks"]["preflight"]["status"], "SUBMITTED")

    def test_zero_exit_with_signal_is_failure(self):
        task = self.active()
        self.marker(task)
        self.complete_record(task, exit_code="0:15")
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "FAILED")

    def test_stale_success_marker_is_rejected(self):
        task = self.active()
        self.marker(task, job_id="99")
        self.complete_record(task)
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "FAILED")

    def test_gpu_requires_both_wrapper_and_original_worker_markers(self):
        self.ready_smokes()
        task = self.active("smoke-a")
        self.marker(task, model=False)
        self.complete_record(task)
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["smoke-a"]["status"], "FAILED")

    def test_missing_marker_has_bounded_visibility_grace(self):
        task = self.active()
        self.complete_record(task)
        with mock.patch.object(rq.time, "time", return_value=1000):
            rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "SUBMITTED")
        with mock.patch.object(rq.time, "time", return_value=1061):
            rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "FAILED")

    def test_accounting_lag_is_uncertain_not_failed_or_resubmitted(self):
        self.active()
        rq.tick(self.plan_path)
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "UNCERTAIN")
        self.assertEqual(self.scheduler.commands, [])

    def test_crash_after_durable_intent_does_not_resubmit(self):
        self.scheduler.response = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "SUBMITTING")
        self.scheduler.response = None
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "UNCERTAIN")

    def test_ambiguous_submission_recovers_exact_queue_name(self):
        self.scheduler.response = subprocess.TimeoutExpired("sbatch", 60)
        rq.tick(self.plan_path)
        task = self.read()["tasks"]["data"]
        self.scheduler.queue["987"] = self.scheduler.row("987", task["job_name"])
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["job_id"], "987")
        self.assertEqual(len(self.scheduler.commands), 1)

    def test_ambiguous_submission_recovers_accounting_and_marker(self):
        state = self.read()
        task = state["tasks"]["data"]
        task["status"] = "SUBMITTING"
        self.save(state)
        self.scheduler.history["987"] = self.scheduler.row("987", task["job_name"], "COMPLETED", "0:0")
        self.marker({**task, "job_id": "987"})
        rq.tick(self.plan_path)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "PASS")

    def test_duplicate_recovery_matches_do_not_choose_arbitrarily(self):
        state = self.read(); state["tasks"]["data"]["status"] = "SUBMITTING"; self.save(state)
        name = state["tasks"]["data"]["job_name"]
        self.scheduler.queue = {jid: self.scheduler.row(jid, name) for jid in ("123", "124")}
        result = rq.tick(self.plan_path)
        self.assertEqual(result["status"], "PAUSED")
        self.assertNotIn("job_id", self.read()["tasks"]["data"])
        self.assertEqual(self.scheduler.commands, [])

    def test_malformed_acknowledgement_is_not_retried(self):
        self.scheduler.response = mock.Mock(returncode=0, stdout="unparseable", stderr="")
        rq.tick(self.plan_path); rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "UNCERTAIN")

    def test_explicit_quota_rejection_backs_off_then_retries(self):
        self.scheduler.response = mock.Mock(returncode=1, stdout="", stderr="QOSMaxSubmitJobPerUserLimit")
        rq.tick(self.plan_path); rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        state = self.read()
        self.assertEqual(state["tasks"]["data"]["status"], "WAITING_QUOTA")
        state["tasks"]["data"]["retry_after"] = 0; self.save(state)
        self.scheduler.response = None
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 2)

    def test_standard_slurm_policy_rejection_backs_off_with_original_reason(self):
        reason = ("sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
                  "(job submit limit, user's size and/or time limits)")
        self.scheduler.response = mock.Mock(returncode=1, stdout="", stderr=reason)
        result = rq.tick(self.plan_path)
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        self.assertEqual(result["status"], "WAITING_CAPACITY")
        self.assertEqual(self.read()["tasks"]["data"]["error"], reason)

    def test_unclassified_submission_failure_remains_uncertain(self):
        self.scheduler.response = mock.Mock(returncode=1, stdout="", stderr="socket connection lost")
        rq.tick(self.plan_path); rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "UNCERTAIN")

    def test_definite_rejection_is_terminal_not_repeated(self):
        self.scheduler.response = mock.Mock(returncode=1, stdout="", stderr="Invalid partition name specified")
        rq.tick(self.plan_path)
        calls = len(self.scheduler.commands)
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), calls)
        self.assertEqual(self.read()["tasks"]["data"]["status"], "FAILED")

    def test_query_failure_is_closed_to_submissions(self):
        with mock.patch.object(rq, "_scheduler_snapshot", side_effect=RuntimeError("query failed")):
            result = rq.tick(self.plan_path)
        self.assertEqual(result["status"], "PAUSED")
        self.assertEqual(self.scheduler.commands, [])

    def test_failed_smoke_blocks_only_its_own_experiment(self):
        self.ready_smokes()
        state = self.read()
        state["tasks"]["smoke-a"]["status"] = "FAILED"
        state["tasks"]["smoke-b"]["status"] = "PASS"
        self.save(state)
        rq.tick(self.plan_path)
        state = self.read()
        self.assertEqual(state["tasks"]["a--coco--0"]["status"], "BLOCKED")
        self.assertEqual(state["tasks"]["b--coco--0"]["status"], "SUBMITTED")

    def test_failed_data_blocks_all_descendants_but_collection_runs(self):
        state = self.read(); state["tasks"]["data"]["status"] = "FAILED"; self.save(state)
        rq.tick(self.plan_path)
        state = self.read()
        self.assertEqual(state["tasks"]["preflight"]["status"], "BLOCKED")
        self.assertEqual(state["tasks"]["collect"]["status"], "SUBMITTED")
        self.assertEqual(sum(t["status"] == "BLOCKED" for t in state["tasks"].values()), 5)

    def test_lock_blocks_second_manager_but_persistent_lock_file_is_reusable(self):
        with rq._lock(self.root / "run"):
            with self.assertRaisesRegex(RuntimeError, "Another rolling"):
                rq.tick(self.plan_path)
        rq.tick(self.plan_path)
        self.assertEqual(len(self.scheduler.commands), 1)

    def test_worker_rejects_login_node_execution(self):
        self.active()
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
            rq.execute_queue_task(self.plan_path, "data")

    def test_collection_analysis_failure_writes_failure_marker(self):
        task = self.active("collect")
        environment = {"SLURM_JOB_ID": "100", "SLURM_JOB_NAME": task["job_name"]}
        with mock.patch.dict(os.environ, environment), mock.patch.object(rq, "collect", return_value={"analysis_returncode": 1}):
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                rq.execute_queue_task(self.plan_path, "collect")
        marker = json.loads(rq._marker_path(self.plan, "collect").read_text())
        self.assertEqual(marker["status"], "FAIL")

    def test_successful_collection_makes_queue_terminal(self):
        state = self.read()
        for key, task in state["tasks"].items():
            if key != "collect":
                task["status"] = "PASS"
        self.save(state)
        task = self.active("collect")
        self.marker(task)
        self.complete_record(task)
        result = rq.tick(self.plan_path)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(self.scheduler.commands, [])

    def test_worker_script_excludes_credentials_and_inference_uses_local_models(self):
        self.ready_smokes()
        rq.tick(self.plan_path)
        script = (self.root / "run/rolling_slurm/smoke-a.sh").read_text()
        self.assertIn("unset HF_TOKEN HUGGING_FACE_HUB_TOKEN ICS_DINOV3_URL", script)
        self.assertIn("HF_HUB_OFFLINE=1", script)


class SchedulerParsingTests(unittest.TestCase):
    def state(self):
        return {"policy": {"user": "test"}, "created_at": 1700000000}

    def test_expanded_arrays_and_step_records(self):
        outputs = ["12_0|other|PENDING|highpriority\n12_1|other|RUNNING|highpriority\n",
                   "20|owned|COMPLETED|0:0|highpriority\n20.batch|batch|COMPLETED|0:0|highpriority\n"]
        with mock.patch.object(rq, "_command", side_effect=outputs) as command:
            queue, history = rq._scheduler_snapshot(self.state())
        self.assertEqual(len(queue), 2)
        self.assertEqual(set(history), {"20"})
        self.assertIn("--array", command.call_args_list[0].args[0])
        self.assertIn("--starttime", command.call_args_list[1].args[0])
        self.assertNotIn("--account", command.call_args_list[0].args[0])

    def test_compressed_arrays_fail_closed(self):
        with mock.patch.object(rq, "_command", side_effect=["12_[0-9]|other|PENDING|highpriority\n", ""]):
            with self.assertRaisesRegex(ValueError, "unexpanded"):
                rq._scheduler_snapshot(self.state())

    def test_cancelled_by_user_is_terminal_state(self):
        with mock.patch.object(rq, "_command", side_effect=["", "20|owned|CANCELLED by 42|0:15|highpriority\n"]):
            _, history = rq._scheduler_snapshot(self.state())
        self.assertEqual(history["20"]["state"], "CANCELLED")

    def test_hidden_partitions_heterogeneous_jobs_and_empty_unrelated_qos(self):
        with mock.patch.object(rq, "_command", side_effect=[
            "20+0|other|RUNNING|highpriority\n20+1|other|PENDING|highpriority\n",
            "17|old-unrelated|COMPLETED|0:0|\n"]) as command:
            queue, history = rq._scheduler_snapshot(self.state())
        self.assertEqual(len(queue), 2)
        self.assertEqual(history["17"]["qos"], "")
        self.assertIn("--all", command.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
