"""Submission/circuit-breaker regressions with fake Slurm executables, never GPU work."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ics import rolling_queue as rq
from ics.orchestration import fingerprint, write_plan
from test_scaling_integration import create_fixture, atomic

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('repair_engine_queue_cli',PROJECT/'scripts/queue_ics.py')
CLI=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CLI)


class RepairEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        fixture=create_fixture(self.root,PROJECT,PROJECT)
        for key,value in fixture.items():setattr(self,key,value)
        # Many independent ready tasks exercise the existing full 60-job window.
        original=copy.deepcopy(self.plan['tasks'])
        self.plan['tasks']=[dict(original[i % len(original)],id='formal-'+str(i)) for i in range(80)]
        self.plan.pop('fingerprint');self.plan['fingerprint']=fingerprint(self.plan)
        atomic(self.run / "plan.json", self.plan)  # Build the enlarged synthetic fixture only.
        state=json.loads((self.run/'rolling_state.json').read_text())
        state.update(plan_fingerprint=self.plan['fingerprint'],tasks=rq._task_table(self.plan,state['queue_id']))
        for key in ('data','preflight'):state['tasks'][key].update(status='PASS',job_id='9'+str(len(key)))
        scheduler={'jobs':[],'submissions':[],'updates':[]}
        for number,task in enumerate(state['tasks'].values()):
            if task['kind']!='smoke':continue
            jid=str(900+number);task.update(status='PASS',job_id=jid)
            scheduler['jobs'].append(dict(job_id=jid,job_name=task['job_name'],state='COMPLETED',qos='highpriority',
                exit_code='0:0',partition='l40sp',node='GOOD',key=task['key']))
        state['tasks']['formal-79'].update(status='PASS',job_id='999')
        atomic(self.run/'rolling_state.json',state)
        atomic(self.scheduler/'scheduler.json',scheduler)
        policy={'plan_fingerprint':self.plan['fingerprint'],'worker_source':self.plan['project_root'],
            'shared_limit':60,'inherited_shared_limit':60,'verified_bound':60,
            'qos_evidence':{'status':'VERIFIED','actual_limit':60},
            'smoke_partition':'l40sp','formal_partitions':'h200p,l40sp'}
        policy['fingerprint']=fingerprint(policy);atomic(self.run/'scheduling/policy.json',policy)
        self.repair_policy={'fingerprint':'repair-test','exclude_nodes':['A484652X3B10601']}
        def exclusions(profile,policy):
            result=dict(profile)
            if result.get('gpus',0):
                result['exclude']=','.join(filter(None,[result.get('exclude',''),*policy['exclude_nodes']]))
            return result
        def launch(plan_path,plan,state,task):
            attempt=task['attempts']+1
            task['repair_attempt']={'policy_fingerprint':'repair-test','attempt_number':attempt}
            task['job_name']+='-repair'+str(attempt)
            return [sys.executable,str(self.run/'repair/source/scripts/repair_worker.py'),'--intent','sealed-'+task['key']]
        self.stub=types.SimpleNamespace(load_repair_policy=lambda p:self.repair_policy,
                                      make_worker_launch=launch,effective_exclusions=exclusions)
        p=patch.dict(sys.modules,{'ics.repair':self.stub});p.start();self.addCleanup(p.stop)
        p=patch.dict(os.environ,self.env);p.start();self.addCleanup(p.stop)
    def queue(self):return json.loads((self.run/'rolling_state.json').read_text())
    def scheduler_state(self):return json.loads((self.scheduler/'scheduler.json').read_text())
    def tick(self,**kwargs):return rq.tick(self.run/'plan.json',repair=True,**kwargs)
    def hashes(self):return {str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in self.source.rglob('*') if p.is_file()}
    def fail_repaired(self,content,filename='runner.log'):
        self.tick()
        state=self.scheduler_state()
        job=next(j for j in state['jobs'] if j['state']=='PENDING')
        job.update(state='FAILED',exit_code='1:0');atomic(self.scheduler/'scheduler.json',state)
        target=(self.run/'logs'/(job['key']+'-'+job['job_id']+'.log') if filename=='slurm.log'
                else self.run/'jobs'/job['key']/filename)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
        return job
    def test_real_fake_slurm_window_exclusions_and_preserved_source(self):
        before=self.hashes();plan=(self.run/'plan.json').read_bytes()
        result=self.tick();state=self.queue();scheduler=self.scheduler_state()
        self.assertEqual(len(scheduler['submissions']),60)
        self.assertEqual(result['effective_policy']['max_inflight'],60)
        self.assertEqual(state['tasks']['formal-79']['status'],'PASS')
        for job in scheduler['jobs']:
            if job['state']!='PENDING':continue
            args=job['argv'];excluded=args[args.index('--exclude')+1]
            self.assertIn('A484652X3B10601',excluded)
            if state['tasks'][job['key']]['resource']=='cgics_mllm':self.assertIn('2FS9B74',excluded)
            self.assertEqual(args[args.index('--partition')+1],'h200p,l40sp')
            command=shlex.split(Path(job['script']).read_text().splitlines()[-1])
            self.assertIn(str(self.run/'repair/source/scripts/repair_worker.py'),command)
            self.assertNotIn(str(self.source/'scripts/queue_ics.py'),command)
        self.tick()
        self.assertEqual(len(self.scheduler_state()['submissions']),60)
        self.assertEqual(self.scheduler_state()['updates'],[])
        self.assertEqual(self.hashes(),before);self.assertEqual((self.run/'plan.json').read_bytes(),plan)
    def test_new_cuda_failure_stops_before_replacement_submission_and_latches(self):
        job=self.fail_repaired('Traceback\nRuntimeError: CUDA is unavailable; run inside an allocated GPU job\n')
        result=self.tick();self.assertEqual(result['status'],'PAUSED')
        self.assertEqual(result['repair_circuit_breaker']['job_id'],job['job_id'])
        self.assertEqual(result['repair_circuit_breaker']['reason'],'cuda_unavailable')
        self.assertEqual(len(self.scheduler_state()['submissions']),60)
        # Re-running even the ordinary controller cannot silently clear this latch.
        rq.tick(self.run/'plan.json')
        self.assertEqual(len(self.scheduler_state()['submissions']),60)
        self.assertEqual(sum(j['state']=='PENDING' for j in self.scheduler_state()['jobs']),59)
    def test_vllm_cuda_failure_and_queue_state_error_classified(self):
        task={'key':'smoke-x','status':'FAILED','job_id':'123','repair_attempt':{'policy_fingerprint':'repair-test'}}
        for name,text,reason in [('vllm.log','RuntimeError: No CUDA GPUs are available','cuda_unavailable'),
                ('slurm.log',"Queue error: [Errno 2] No such file or directory: '/run/rolling_state.json'",'queue_state_unreadable')]:
            with self.subTest(name=name):
                path=(self.run/'logs/smoke-x-123.log' if name=='slurm.log' else self.run/'jobs/smoke-x'/name)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
                found=rq._repair_infrastructure_failure(self.plan,{'tasks':{'smoke-x':task}})
                self.assertEqual(found['reason'],reason)
                path.unlink()
    def test_historical_failure_and_model_error_do_not_trip_infrastructure_guard(self):
        task={'key':'old','status':'FAILED','job_id':'3'}
        path=self.run/'jobs/old/runner.log';path.parent.mkdir(parents=True);path.write_text('CUDA is unavailable')
        self.assertIsNone(rq._repair_infrastructure_failure(self.plan,{'tasks':{'old':task}}))
        task['repair_attempt']={'policy_fingerprint':'repair-test'}
        path.write_text('ValueError: invalid mask geometry')
        self.assertIsNone(rq._repair_infrastructure_failure(self.plan,{'tasks':{'old':task}}))
    def test_repair_worker_startup_missing_file_stops_before_submission(self):
        job=self.fail_repaired("Repair worker error: [Errno 2] No such file or directory: '/run/repair/intents/task/attempt-1.json'\n",
                               filename='slurm.log')
        result=self.tick()
        self.assertEqual(result['repair_circuit_breaker']['reason'],'worker_startup_unreadable')
        self.assertEqual(result['repair_circuit_breaker']['job_id'],job['job_id'])
        self.assertEqual(len(self.scheduler_state()['submissions']),60)
        # Missing model assets alone are not reclassified as wrapper startup failures.
        state={'tasks':{'x':{'key':'x','status':'FAILED','job_id':'2',
                            'repair_attempt':{'policy_fingerprint':'repair-test'}}}}
        path=self.run/'jobs/x/runner.log';path.parent.mkdir(parents=True)
        path.write_text("FileNotFoundError: [Errno 2] No such file or directory: '/data/image.jpg'\n")
        self.assertIsNone(rq._repair_infrastructure_failure(self.plan,state))
    def test_stop_request_has_no_submission_side_effect(self):
        self.tick(stop_requested=lambda:True)
        self.assertEqual(self.scheduler_state()['submissions'],[])
    def test_watcher_exits_and_records_infrastructure_reason(self):
        summary={'terminal':False,'status':'PAUSED','repair_circuit_breaker':{'reason':'cuda_unavailable'}}
        with patch.object(CLI,'ROOT',self.source),patch('ics.rolling_queue.tick',return_value=summary) as tick,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.watcher(self.run,repair=True),3)
        self.assertTrue(tick.call_args.kwargs['repair'])
        self.assertTrue(tick.call_args.kwargs['scheduling'])
        record=json.loads((self.run/'manager.json').read_text())
        self.assertEqual(record['exit']['reason'],'infrastructure_failure')
        self.assertEqual(CLI.manager_status(self.run)['manager'],'STOPPED')

if __name__=='__main__':unittest.main(verbosity=2)
