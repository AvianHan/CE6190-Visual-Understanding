"""Real queue/worker/runner subprocess integration with a local fake scheduler.

Only the copied test repository replaces the network downloader and model
adapter. Preflight, LVIS sampling, generated worker shells, the inference runner,
completion markers, reconciliation, and result analysis are production code.
No real Slurm command, model, GPU computation, or network transfer is used.
The normal background CLI is exercised for start/kill/resume; subsequent polls
use the production ``watch --once`` CLI to avoid ten-second test sleeps.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This script is installed only in the temporary fake command directory. A
# separate session runs each accepted batch script, as a scheduler would, so
# killing the manager does not kill already accepted work.
FAKE_SLURM = r'''
import fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

root = Path(os.environ['ICS_TEST_SCHEDULER'])
jobs = root/'jobs'
jobs.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)

def records():
    return [read(p) for p in sorted(jobs.glob('*.json'))]

if len(sys.argv) > 1 and sys.argv[1] == '--execute':
    path = Path(sys.argv[2])
    job = read(path)
    job.update(state='RUNNING', pid=os.getpid(), started_at=time.time())
    write(path, job)
    gate = root/('hold-' + job['key'])
    deadline = time.monotonic() + 20
    while gate.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    if gate.exists():
        job.update(state='FAILED', exit_code='99:0', ended_at=time.time())
        write(path, job)
        sys.exit(99)
    env = dict(os.environ, SLURM_JOB_ID=job['job_id'],
               SLURM_JOB_NAME=job['job_name'], CUDA_VISIBLE_DEVICES='0')
    output = Path(job['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as stream:
        result = subprocess.run(['/bin/bash', job['script']], env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    job.update(state='COMPLETED' if result.returncode == 0 else 'FAILED',
               exit_code=str(max(0, result.returncode)) + ':' + str(-min(0, result.returncode)),
               ended_at=time.time())
    write(path, job)
    sys.exit(0)

name = Path(sys.argv[0]).name
if name == 'nvidia-smi':
    sys.exit(1)  # Never inspect or use host GPUs in a CPU integration test.
if name == 'squeue':
    for job in records():
        if job['state'] in ('PENDING', 'RUNNING'):
            print('|'.join(job[k] for k in ('job_id', 'job_name', 'state', 'qos')))
elif name == 'sacct':
    for job in records():
        print('|'.join(job.get(k, '') for k in ('job_id', 'job_name', 'state', 'exit_code', 'qos')))
elif name == 'sbatch':
    args = sys.argv[1:]
    def flag(key):
        return args[args.index(key) + 1]
    with (root/'submit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jid = str(1000 + len(records()) + 1)
        script = Path(args[-1])
        job = dict(job_id=jid, job_name=flag('--job-name'), qos=flag('--qos'),
                   state='PENDING', exit_code='', key=script.stem,
                   script=str(script), output=flag('--output').replace('%j', jid))
        path = jobs/(jid + '.json')
        write(path, job)
        with (root/'scheduler.log').open('a') as log:
            subprocess.Popen([sys.executable, __file__, '--execute', str(path)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
    lost = root/'lose-ack'
    print('accepted-with-lost-job-id' if lost.exists() and lost.read_text() == job['key'] else jid)
else:
    raise RuntimeError('Unexpected fake scheduler command: ' + name)
'''

FIXTURE_DOWNLOADER = r'''
"""Test-only local fixture check; intentionally contains no network code."""
import os
from pathlib import Path
root = Path(os.environ['ICS_RUNTIME'])
assert (root/'data/LVIS/lvis_val.pkl').is_file()
with (root/'fixture-data-invocations.txt').open('a') as stream:
    stream.write('validated local synthetic data\n')
'''

FIXTURE_ADAPTER = r'''
"""Test-only predictor injected before freezing the temporary source copy."""
import numpy as np
from PIL import Image
from .base import Prediction

class FixtureAdapter:
    def __init__(self, config):
        self.fail = config.get('deliberate_failure', False)
    def metadata(self):
        return {'kind': 'synthetic_cpu_queue_integration_only'}
    def close(self):
        pass
    def predict(self, inputs, *, seed=0):
        assert set(inputs) == {'query_image', 'references'}
        assert all(set(ref) == {'image', 'mask'} for ref in inputs['references'])
        if self.fail:
            raise RuntimeError('Deliberate test-only model failure')
        with Image.open(inputs['query_image']) as image:
            width, height = image.size
        return Prediction(np.ones((height, width), dtype=bool), {}, {'synthetic': True})

def create_adapter(config):
    assert config.get('name') == 'synthetic_queue_fixture'
    return FixtureAdapter(config)
'''


class QueueEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ics-queue-e2e-')
        self.base = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.root = self.base/'repo'
        shutil.copytree(PROJECT_ROOT, self.root, ignore=shutil.ignore_patterns(
            '.git', '__pycache__', '.pytest_cache', 'tests', '*update-backups'))
        self.runtime = self.base/'runtime'
        self.run = self.runtime/'runs/test'
        self.scheduler = self.base/'fake-scheduler'
        self.bin = self.base/'bin'
        self.bin.mkdir()
        self.scheduler.mkdir()
        for name in ('squeue', 'sacct', 'sbatch', 'nvidia-smi'):
            path = self.bin/name
            path.write_text('#!' + sys.executable + '\n' + FAKE_SLURM)
            path.chmod(0o755)
        self.env = dict(os.environ, ICS_RUNTIME=str(self.runtime),
                        ICS_TEST_SCHEDULER=str(self.scheduler),
                        PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        self.managers = set()
        self.addCleanup(self.cleanup_processes)
        (self.root/'scripts/download_data.py').write_text(FIXTURE_DOWNLOADER)
        (self.root/'ics/adapters/__init__.py').write_text(FIXTURE_ADAPTER)
        self._make_data()
        self._make_plan_inputs()

    def _make_data(self):
        folder = self.runtime/'data/LVIS/coco/train2017'
        folder.mkdir(parents=True)
        annotations = {}
        for number in range(4):
            relative = f'train2017/{number:012d}.jpg'
            Image.new('RGB', (3, 2), (number * 40, 20, 50)).save(folder/Path(relative).name)
            mask = np.zeros((2, 3), dtype=np.uint8)
            if number:
                mask[0, 1] = 1
            annotations[relative] = {'annotations': [{'segmentation': mask}]}
        metadata = {cid: copy.deepcopy(annotations) for cid in range(10)}
        self.pickle = self.runtime/'data/LVIS/lvis_val.pkl'
        with self.pickle.open('wb') as stream:
            pickle.dump(metadata, stream)
        self.pickle_digest = hashlib.sha256(self.pickle.read_bytes()).hexdigest()

    def _make_plan_inputs(self, fail_second=False):
        config_dir = self.root/'configs'
        experiments = []
        for name, fail in (('good', False), ('other', fail_second)):
            config = {'experiment_id': name, 'method': 'synthetic_queue_fixture',
                      'variant': 'test_only', 'seed': 0,
                      'adapter': {'name': 'synthetic_queue_fixture', 'deliberate_failure': fail}}
            (config_dir/(name + '.json')).write_text(json.dumps(config))
            # Paths must refer to the source copy after freezing, rather than
            # absolute mutable config files in the original test checkout.
            experiments.append({'config': 'configs/' + name + '.json',
                                'resource': 'insid3', 'manifest_ids': ['lvis']})
        suite = {'schema_version': 1, 'smoke_episodes': 1, 'prepare_missing_manifests': True,
                 'manifests': [{'id': 'lvis', 'path': '${ICS_RUNTIME}/manifests/lvis-nonempty-v2/test.json',
                                'dataset_root': '${ICS_RUNTIME}/data', 'prepare': {
                                    'dataset': 'lvis', 'fold': 4, 'queries': 3,
                                    'references_per_query': 1, 'shots': 1, 'seed': 0, 'query_seed': 0,
                                    'trust_prepared_pickle': True, 'empty_mask_policy': 'exclude',
                                    'protocol_id': 'test-only-lvis-queue-integration'}}],
                 'experiments': experiments}
        (config_dir/'suite.queue-test.json').write_text(json.dumps(suite))
        cluster = {'schema_version': 1, 'launcher_python': sys.executable, 'max_parallel': 2,
                   'defaults': {'account': 'test', 'partition': 'fake', 'qos': 'test',
                                'time': '00:10:00', 'cpus': 1},
                   'resources': {'preflight': {'gpus': 0, 'python': sys.executable},
                                 'insid3': {'gpus': 1, 'python': sys.executable, 'max_parallel': 2}}}
        (config_dir/'cluster.queue-test.json').write_text(json.dumps(cluster))

    def state(self):
        return json.loads((self.run/'rolling_state.json').read_text())

    def jobs(self):
        return [json.loads(p.read_text()) for p in sorted((self.scheduler/'jobs').glob('*.json'))]

    def diagnostics(self):
        paths = [self.run/'manager.log', self.scheduler/'scheduler.log']
        paths += list((self.run/'logs').glob('*.log'))
        return '\n'.join(str(p) + '\n' + '\n'.join(p.read_text().splitlines()[-12:])
                         for p in paths if p.is_file())

    def wait_for(self, predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail('Timed out waiting for test-owned process.\n' + self.diagnostics())

    def cli(self, *args, source=False, allowed=(0,)):
        root = self.run/'source' if source else self.root
        result = subprocess.run([sys.executable, '-I', str(root/'scripts/queue_ics.py'), *args],
                                env=self.env, capture_output=True, text=True, timeout=20)
        self.assertIn(result.returncode, allowed, result.stderr + '\n' + self.diagnostics())
        return json.loads(result.stdout)

    def start(self):
        result = self.cli('start', '--runtime', str(self.runtime), '--run-dir', str(self.run),
                          '--suite', 'configs/suite.queue-test.json',
                          '--cluster', 'configs/cluster.queue-test.json', '--interval', '10')
        if result.get('manager_pid'):
            self.managers.add(result['manager_pid'])
        self.assertIn(result['manager'], ('STARTED', 'STARTING', 'ALREADY_RUNNING'))
        return result

    def is_active(self):
        path = self.run/'manager.lock'
        if not path.exists():
            return False
        with path.open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False

    def stop(self, pid, sig=signal.SIGTERM):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        self.wait_for(lambda: not self.is_active())

    def poll_once(self):
        self.assertFalse(self.is_active(), 'Never race the background manager')
        return self.cli('watch', '--run-dir', str(self.run), '--once', source=True, allowed=(0, 2))

    def finish(self):
        seen_uncertain = False
        for _ in range(14):
            self.wait_for(lambda: all(j['state'] not in ('RUNNING', 'PENDING') for j in self.jobs()))
            summary = self.poll_once()
            seen_uncertain |= any(t['status'] == 'UNCERTAIN' for t in self.state()['tasks'].values())
            if summary['terminal']:
                return summary, seen_uncertain
        self.fail('Queue did not finish after 14 actual reconciliation rounds\n' + self.diagnostics())

    def cleanup_processes(self):
        # Every spawned executable/script/config is under this private path.
        # Scan it as well as recorded jobs: startup can fail before the child
        # persists its PID, and inference runners own independent sessions.
        # Recheck ownership before every signal to avoid recycled PIDs.
        prefix = (str(self.base) + os.sep).encode()

        def owned(pid):
            try:
                args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                return False
            return any(arg.startswith(prefix) for arg in args)

        def live():
            return [int(p.name) for p in Path('/proc').iterdir()
                    if p.name.isdigit() and owned(p.name)]

        # Bounded cleanup runs before TemporaryDirectory removes source/data.
        for sig, timeout in ((signal.SIGTERM, 2), (signal.SIGKILL, 1)):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                remaining = live()
                if not remaining:
                    return
                for pid in remaining:
                    if owned(pid):
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                time.sleep(.02)
        self.assertFalse(live(), 'Test-owned subprocess survived bounded cleanup')

    def test_killed_manager_resumes_accepted_preflight_without_duplicate_jobs(self):
        initial = self.start()
        self.wait_for(lambda: self.jobs() and self.jobs()[0]['state'] == 'COMPLETED')
        self.stop(initial['manager_pid'])
        (self.scheduler/'hold-preflight').write_text('test gate')
        self.poll_once()  # Reconcile data and accept the real preflight worker.
        self.wait_for(lambda: any(j['key'] == 'preflight' and j['state'] == 'RUNNING' for j in self.jobs()))
        first = self.state()
        data_id = first['tasks']['data']['job_id']
        preflight_id = first['tasks']['preflight']['job_id']
        running = self.start()
        self.wait_for(lambda: self.state()['tasks']['preflight']['status'] == 'RUNNING')
        self.stop(running['manager_pid'], signal.SIGKILL)
        self.assertEqual(self.state()['tasks']['preflight']['job_id'], preflight_id)
        stopped = self.cli('status', '--run-dir', str(self.run), source=True)
        self.assertEqual(stopped['manager'], 'STOPPED')
        self.assertEqual(stopped['status'], 'MANAGER_STOPPED')
        self.assertEqual(stopped['saved_queue_status'], 'RUNNING')
        self.assertEqual(stopped['manager_exit_reason'], 'unknown')
        self.assertFalse(stopped['terminal'])
        self.assertFalse(stopped['state_is_live'])
        self.assertFalse(stopped['scheduler_queried'])
        (self.scheduler/'hold-preflight').unlink()
        # Accepted work completes even though its submitting manager was killed.
        self.wait_for(lambda: next(j for j in self.jobs() if j['key'] == 'preflight')['state'] == 'COMPLETED')
        self.assertFalse(self.is_active())
        resumed = self.start()
        self.wait_for(lambda: self.state()['tasks']['preflight']['status'] == 'PASS')
        self.stop(resumed['manager_pid'])
        summary, _ = self.finish()
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertEqual(summary['counts'], {'PASS': 7})
        tasks = self.state()['tasks']
        self.assertEqual(tasks['data']['job_id'], data_id)
        self.assertEqual(tasks['preflight']['job_id'], preflight_id)
        self.assertTrue(all(t['attempts'] == 1 for t in tasks.values()))
        self.assertEqual(len({j['key'] for j in self.jobs()}), 7)
        self.assertEqual(len(self.jobs()), 7)
        self.assertEqual((self.runtime/'fixture-data-invocations.txt').read_text().splitlines(),
                         ['validated local synthetic data'])
        self.assertEqual(hashlib.sha256(self.pickle.read_bytes()).hexdigest(), self.pickle_digest)
        audit = json.loads((self.runtime/'manifests/lvis-nonempty-v2/test.protocol.json').read_text())
        self.assertEqual(audit['sample_eligibility']['excluded_count'], 1)
        self.assertEqual(audit['sample_eligibility']['eligible_count'], 3)
        self.assertEqual(json.loads((self.run/'execution_summary.json').read_text())['analysis_returncode'], 0)
        self.assertTrue(list((self.run/'analysis').glob('*.csv')))

    def test_failed_smoke_blocks_only_its_branch_and_lost_ack_is_not_resubmitted(self):
        self._make_plan_inputs(fail_second=True)
        (self.scheduler/'lose-ack').write_text('smoke-good')
        started = self.start()
        self.wait_for(lambda: bool(self.jobs()))
        self.stop(started['manager_pid'])
        summary, saw_uncertain = self.finish()
        self.assertEqual(summary['status'], 'COMPLETE_WITH_FAILURES')
        self.assertEqual(summary['counts'], {'PASS': 5, 'FAILED': 1, 'BLOCKED': 1})
        self.assertTrue(saw_uncertain)
        tasks = self.state()['tasks']
        self.assertEqual(tasks['smoke-good']['attempts'], 1)
        self.assertIn('recovered_at', tasks['smoke-good'])
        self.assertEqual(tasks['other--lvis--0']['status'], 'BLOCKED')
        self.assertEqual(tasks['other--lvis--0']['attempts'], 0)
        self.assertEqual(tasks['good--lvis--0']['status'], 'PASS')
        self.assertEqual(tasks['collect']['status'], 'PASS')
        self.assertEqual(len(self.jobs()), 6)
        self.assertEqual(len([j for j in self.jobs() if j['key'] == 'smoke-good']), 1)
        self.assertFalse(any(j['key'] == 'other--lvis--0' for j in self.jobs()))
        report = json.loads((self.run/'execution_summary.json').read_text())
        self.assertEqual(report['counts'], {'FAIL': 1, 'NOT_RUN_OR_DEPENDENCY_BLOCKED': 1, 'PASS': 2})
        self.assertEqual(report['analysis_returncode'], 0)


if __name__ == '__main__':
    unittest.main()
