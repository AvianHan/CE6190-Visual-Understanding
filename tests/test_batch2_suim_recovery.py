"""Selective SUIM recovery contracts with fake Slurm; no GPU/network use."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ics import batch2, batch2_data, batch2_recovery as recovery, orchestration, rolling_queue as rq
from scripts import queue_ics

ROOT=Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.top=Path(self.temp.name);self.runtime=self.top/'runtime';self.runtime.mkdir()
        python=self.runtime/'envs/insid3/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
        self.run=self.top/'run';self.run.mkdir()
        self.env=mock.patch.dict(os.environ,ICS_RUNTIME=str(self.runtime),ICS_BATCH2_ROOT=str(self.run))
        self.env.start();self.addCleanup(self.env.stop)
        self.source=queue_ics.freeze_source(self.run,ROOT/'configs/suite.batch2.core.json',ROOT/'configs/cluster.orion.batch2.json',self.runtime)
        with mock.patch.object(orchestration,'PROJECT_ROOT',self.source):
            batch2.prepare(self.source,self.run,batch2.selection(self.runtime,self.run))
        self.scope=self.run/'scopes/suim';self.plan=rq._plan(self.scope/'plan.json')
        self.state=rq._read_state(self.plan)
        for item in self.state['tasks'].values():item.update(status='BLOCKED',error='Failed prerequisite')
        for key,jid,status in [('data','206900','FAILED'),('collect','206902','PASS')]:
            task=self.state['tasks'][key]
            command=batch2.make_worker_launch(self.scope/'plan.json',self.plan,task)
            intent=Path(task['batch2_intent'])
            intent.with_suffix('.claim.json').write_text(json.dumps({'slurm_job_id':jid,'job_name':task['job_name'],'intent_sha256':task['batch2_intent_sha256']}))
            task.update(status=status,job_id=jid,attempts=1)
            marker={'schema_version':1,'key':key,'kind':key,'plan_fingerprint':self.plan['fingerprint'],
                    'slurm_job_id':jid,'status':'FAIL' if key=='data' else 'PASS'}
            if key=='data':marker.update(error_type='ValueError',error='Image/mask dimensions differ: f_r_1289_.jpg')
            rq._atomic_json(rq._marker_path(self.plan,key),marker)
        rq._atomic_json(self.scope/rq.STATE_NAME,self.state)
        (self.scope/'batch2_analysis').mkdir();(self.scope/'batch2_analysis/summary.json').write_text('{"formal_ok":0}')
        self.other={s:(self.run/'scopes'/s/rq.STATE_NAME).read_bytes() for s in ('core','isic')}
        self.original_snapshot=(self.source/'snapshot.json').read_bytes()
        self.claim_bytes=(self.scope/'batch2_intents/data/attempt-00001.claim.json').read_bytes()
        self.folder=self.run/recovery.TAG;self.folder.mkdir()
        self.overlay=queue_ics.freeze_source(self.folder,ROOT/'configs/suite.batch2.core.json',ROOT/'configs/cluster.orion.batch2.json',self.runtime)
        self.history={str(t['job_id']):{'job_id':str(t['job_id']),'job_name':t['job_name'],
            'qos':self.state['policy']['qos'],'state':'FAILED' if k=='data' else 'COMPLETED',
            'exit_code':'1:0' if k=='data' else '0:0'} for k,t in self.state['tasks'].items() if t.get('job_id')}
        self.queue={'99999':{'job_id':'99999','job_name':'other-scope-running','state':'RUNNING','qos':self.state['policy']['qos']}}
        self.scheduler=mock.patch.object(rq,'_scheduler_snapshot',return_value=(self.queue,self.history))
        self.scheduler.start();self.addCleanup(self.scheduler.stop)
        self.implementation=mock.patch.object(batch2_data,'SUIM_REPAIR_IMPLEMENTATION_ID','suim-strict-geometry-filter-v1',create=True)
        self.implementation.start();self.addCleanup(self.implementation.stop)

    def policy(self):
        return recovery.prepare_policy(self.run,self.overlay,self.plan['fingerprint'],'206900')

    def assert_preserved(self):
        for s,wanted in self.other.items():self.assertEqual((self.run/'scopes'/s/rq.STATE_NAME).read_bytes(),wanted)
        self.assertEqual((self.source/'snapshot.json').read_bytes(),self.original_snapshot)
        self.assertEqual((self.scope/'batch2_intents/data/attempt-00001.claim.json').read_bytes(),self.claim_bytes)

    def test_audit_accepts_other_live_work_but_rejects_extra_suim_attempt(self):
        state,proof=recovery.audit_failure(self.run,self.plan,'206900')
        self.assertEqual(len(proof['unblocked']),170)
        row=copy.deepcopy(self.history['206900']);row['job_name']=self.state['tasks']['preflight']['job_name']
        self.history['206901']=row
        with self.assertRaisesRegex(ValueError,'Unrecorded SUIM'):
            recovery.audit_failure(self.run,self.plan,'206900')

    def test_recovery_archives_failed_data_and_empty_collect_only(self):
        policy=self.policy();receipt=recovery.apply_transaction(self.run,policy)
        after=rq._read_state(self.plan)
        self.assertEqual({t['status'] for t in after['tasks'].values()},{'PENDING'})
        self.assertEqual(after['tasks']['data']['attempts'],1)
        self.assertNotEqual(after['tasks']['data']['job_name'],self.state['tasks']['data']['job_name'])
        self.assertNotEqual(after['tasks']['collect']['job_name'],self.state['tasks']['collect']['job_name'])
        self.assertEqual(after['tasks']['collect']['attempt_history'][-1]['status'],'PASS')
        self.assertTrue((self.folder/'archive/rolling_markers/data.json').is_file())
        self.assertTrue((self.folder/'archive/rolling_markers/collect.json').is_file())
        self.assertTrue((self.folder/'archive/batch2_analysis/summary.json').is_file())
        self.assertFalse((self.scope/'batch2_analysis').exists())
        self.assertEqual(receipt['successful_models_reset'],0)
        self.assert_preserved()
        first=(self.scope/rq.STATE_NAME).read_bytes()
        recovery.apply_transaction(self.run,policy)
        self.assertEqual((self.scope/rq.STATE_NAME).read_bytes(),first)

    def test_interrupted_archive_and_state_swap_roll_forward(self):
        policy=self.policy()
        def fault(stage,index):
            if stage=='archive' and index==1:raise RuntimeError('synthetic interruption')
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            recovery.apply_transaction(self.run,policy,fault=fault)
        self.assertEqual(rq._read_state(self.plan),self.state)
        def state_fault(stage,index):
            if stage=='state':raise RuntimeError('state swapped before applied receipt')
        with self.assertRaisesRegex(RuntimeError,'state swapped'):
            recovery.apply_transaction(self.run,policy,fault=state_fault)
        self.assertFalse((self.folder/'applied.json').exists())
        recovery.apply_transaction(self.run,policy)
        self.assertTrue((self.folder/'applied.json').exists());self.assert_preserved()

    def test_external_state_change_blocks_interrupted_transaction(self):
        policy=self.policy()
        def fault(stage,index):
            if stage=='prepared':raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):recovery.apply_transaction(self.run,policy,fault=fault)
        p=self.run/'scopes/core'/rq.STATE_NAME;p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'Other scope queue changed'):
            recovery.apply_transaction(self.run,policy)

    def test_successful_or_partial_suim_model_never_reset(self):
        key=next(k for k,t in self.state['tasks'].items() if t['kind']=='experiment')
        self.state['tasks'][key].update(status='PASS',job_id='206903',attempts=1)
        rq._atomic_json(self.scope/rq.STATE_NAME,self.state)
        with self.assertRaisesRegex(ValueError,'already attempted'):
            self.policy()

    def test_marker_wrong_failure_and_live_suim_refused(self):
        row=dict(self.history['206900'],state='PENDING')
        self.queue['206900']=row
        with self.assertRaisesRegex(ValueError,'queued/running'):
            self.policy()
        self.queue.pop('206900')
        marker=rq._marker_path(self.plan,'data');data=json.loads(marker.read_text());data['error']='out of memory'
        marker.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'not the audited'):
            self.policy()

    def test_data_dispatch_only_overlay_models_and_collector_original(self):
        policy=self.policy();recovery.apply_transaction(self.run,policy)
        state=rq._read_state(self.plan)
        for key in ('data','preflight'):
            command=batch2.make_worker_launch(self.scope/'plan.json',self.plan,state['tasks'][key])
            self.assertIn(str(self.overlay/'scripts/recover_batch2_suim.py'),command)
            intent=json.loads(Path(command[command.index('--intent')+1]).read_text())
            self.assertEqual(intent['worker_source'],str(self.source))
        for key in ('collect',next(k for k,t in state['tasks'].items() if t['kind']=='experiment')):
            command=batch2.make_worker_launch(self.scope/'plan.json',self.plan,state['tasks'][key])
            self.assertIn(str(self.source/'scripts/batch2_ics.py'),command)
        self.assert_preserved()

    def test_real_corrected_data_worker_new_intent_claim_before_ack(self):
        # Inject a synthetic CPU-only data implementation into the isolated test
        # overlay before sealing its policy. Original model source remains intact.
        fixture=self.overlay/'ics/batch2_data.py'
        fixture.write_text("from pathlib import Path\nSUIM_REPAIR_IMPLEMENTATION_ID='suim-strict-geometry-filter-v1'\ndef prepare_scope(runtime, output_dir, **kwargs):\n Path(runtime,'repaired-data-worker-observed.txt').write_text(kwargs['scope'])\ndef verify_scope(*a,**k): pass\n")
        snap=load_json(self.overlay/'snapshot.json');snap['files']['ics/batch2_data.py']=batch2.digest(fixture)
        rq._atomic_json(self.overlay/'snapshot.json',snap)
        policy=self.policy();recovery.apply_transaction(self.run,policy)
        state=rq._read_state(self.plan);task=state['tasks']['data']
        command=batch2.make_worker_launch(self.scope/'plan.json',self.plan,task)
        self.assertEqual(state['tasks']['data']['status'],'PENDING');self.assertNotIn('job_id',task)
        proc=subprocess.run(command,env=dict(os.environ,SLURM_JOB_ID='300001',SLURM_JOB_NAME=task['job_name']),
                            text=True,capture_output=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual((self.runtime/'repaired-data-worker-observed.txt').read_text(),'suim')
        marker=load_json(rq._marker_path(self.plan,'data'))
        self.assertEqual(marker['status'],'PASS');self.assertEqual(marker['slurm_job_id'],'300001')
        self.assertTrue((self.scope/'batch2_intents/data/attempt-00002.claim.json').is_file())
        self.assert_preserved()
        task.update(status='SUBMITTED',attempts=2,job_id='300001')
        state['tasks']['data']=task
        history=dict(self.history,**{'300001':dict(self.history['206900'],job_id='300001',job_name=task['job_name'],state='COMPLETED',exit_code='0:0')})
        rq._reconcile(self.plan,state,{},history,1)
        self.assertEqual(state['tasks']['data']['status'],'PASS')
        duplicate=subprocess.run(command,env=dict(os.environ,SLURM_JOB_ID='300001',SLURM_JOB_NAME=task['job_name']),text=True,capture_output=True,timeout=30)
        self.assertNotEqual(duplicate.returncode,0)
        self.assertEqual(load_json(rq._marker_path(self.plan,'data')),marker)

    def test_changed_old_claim_and_duplicate_accounting_identity_refused(self):
        claim=self.scope/'batch2_intents/data/attempt-00001.claim.json'
        original=claim.read_bytes();value=json.loads(original);value['slurm_job_id']='BAD'
        claim.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'claim identity'):
            self.policy()
        claim.write_bytes(original)
        self.history['206999']=dict(self.history['206900'],job_id='206999')
        with self.assertRaisesRegex(ValueError,'Duplicate historical'):
            self.policy()

    def test_applied_receipt_checked_without_reverting_progress(self):
        policy=self.policy();recovery.apply_transaction(self.run,policy)
        current=rq._read_state(self.plan);current['tasks']['data'].update(status='SUBMITTED',job_id='300001')
        rq._atomic_json(self.scope/rq.STATE_NAME,current)
        recovery.apply_transaction(self.run,policy)
        self.assertEqual(rq._read_state(self.plan),current)
        path=self.folder/'applied.json';record=load_json(path);record['after_sha256']='tampered'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'receipt identity'):
            recovery.apply_transaction(self.run,policy)
        self.assertEqual(rq._read_state(self.plan),current)

    def test_foreign_manager_refuses_signal(self):
        with mock.patch.object(recovery,'acquire_lock',return_value=None), \
             mock.patch.object(recovery,'load_json',return_value={}), \
             mock.patch.object(recovery,'verified_manager_pid',side_effect=ValueError('foreign PID')), \
             mock.patch.object(recovery.signal,'pidfd_send_signal') as send:
            with self.assertRaisesRegex(ValueError,'foreign PID'):
                recovery.manager_handover_lock(self.run)
        send.assert_not_called()


def load_json(path):return json.loads(Path(path).read_text())
if __name__=='__main__':unittest.main()
