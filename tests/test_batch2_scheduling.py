"""Exact accepted batch2 job migration, no real scheduler or GPU work."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ics import batch2, batch2_scheduling as bs, orchestration, rolling_queue as rq
from ics.scheduling import effective_profile, load_policy
from scripts.queue_ics import freeze_source

ROOT = Path(__file__).resolve().parents[1]


class Batch2SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.top = Path(self.tmp.name); self.runtime = self.top / 'runtime'; self.runtime.mkdir()
        self.parent = self.top / 'run'; self.parent.mkdir()
        self.env = patch.dict(os.environ, ICS_RUNTIME=str(self.runtime), ICS_BATCH2_ROOT=str(self.parent))
        self.env.start(); self.addCleanup(self.env.stop)
        self.source = freeze_source(self.parent, ROOT/'configs/suite.batch2.core.json', ROOT/'configs/cluster.orion.batch2.json', self.runtime)
        with patch.object(orchestration, 'PROJECT_ROOT', self.source):
            batch2.prepare(self.source, self.parent, batch2.selection(self.runtime, self.parent))
        self.controller = self.parent / 'controller/source'; self.controller.mkdir(parents=True)
        (self.controller/'snapshot.json').write_text('{}')
        self.plan = rq._plan(self.parent/'scopes/core/plan.json'); self.path = Path(self.plan['run_dir'])/'plan.json'
        self.state = rq._read_state(self.plan)
        self.qos = {'status':'VERIFIED','actual_limit':60,'effective_limit':60}
        self.parts = {p:{'PartitionName':p,'State':'UP','AllowQos':'ALL','AllowAccounts':'ALL'} for p in ('h200p','l40sp')}
    def prepare(self, **kw):
        with patch.object(bs,'qos_limit',return_value=kw.get('qos',self.qos)), patch.object(bs,'partition_evidence',return_value=self.parts):
            return bs.prepare_policies(self.parent,self.controller)
    def accepted(self):
        key=next(k for k,t in self.state['tasks'].items() if t['kind']=='smoke' and t['resource']=='cgics_mllm')
        task=self.state['tasks'][key]
        with patch.object(rq.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'123\n','')):
            rq._submit(self.path,self.plan,self.state,task,10)
        self.task=task
        self.queue={'123':{'state':'PENDING','job_name':task['job_name'],'qos':self.state['policy']['qos']}}
        self.policy=load_policy(self.plan)
        return task
    def record(self, state='PENDING',partition='h200p',**updates):
        task=self.task;profile=rq._profile(self.plan,task)
        values={'JobId':'123','JobName':task['job_name'],'UserId':self.state['policy']['user']+'(1234)',
          'QOS':self.state['policy']['qos'],'Account':profile['account'],'JobState':state,'Partition':partition,
          'Command':str(Path(self.plan['run_dir'])/'rolling_slurm'/(task['key']+'.sh')),
          'NumCPUs':'8','NumNodes':'1','MinMemoryNode':'64G','TresPerNode':'gres/gpu:2',
          'ReqTRES':'cpu=8,mem=64G,node=1,gres/gpu=2','TimeLimit':'12:00:00',
          'ExcNodeList':profile['exclude']}
        values.update(updates)
        return ' '.join(k+'='+v for k,v in values.items())
    def retarget(self):return bs.retarget_pending_smokes(self.plan,self.state,self.policy,self.queue)
    def test_prepare_reuses_validated_profiles_and_never_expands_60(self):
        value=self.prepare(qos={'status':'VERIFIED','actual_limit':100,'effective_limit':100})
        self.assertEqual(value['shared_limit'],60)
        policy=load_policy(self.plan)
        self.assertEqual(policy['formal_partitions'],'h200p,l40sp')
        self.assertEqual(policy['worker_source'],str(self.source))
        before=(self.path.parent/'scheduling/policy.json').read_bytes()
        self.prepare()
        self.assertEqual((self.path.parent/'scheduling/policy.json').read_bytes(),before)
    def test_all_partitions_validated_before_first_policy_write(self):
        with patch.object(bs,'qos_limit',return_value=self.qos), patch.object(bs,'partition_evidence',side_effect=[self.parts,ValueError('not UP')]):
            with self.assertRaisesRegex(ValueError,'not UP'):bs.prepare_policies(self.parent,self.controller)
        self.assertFalse(any(self.parent.glob('scopes/*/scheduling/policy.json')))
    def test_exact_pending_smoke_is_updated_in_place_only(self):
        self.prepare(); task=self.accepted(); original=copy.deepcopy(task)
        with patch.object(rq,'_command',side_effect=[self.record(),self.record(partition='l40sp')]), \
             patch.object(bs.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')) as update:
            events=self.retarget()
        self.assertEqual(events[0]['status'],'UPDATED')
        self.assertEqual(update.call_args.args[0],['scontrol','update','JobId=123','Partition=l40sp'])
        self.assertEqual(task,original)
        files=list((self.path.parent/'scheduling/updates').glob('*.json'));self.assertEqual(len(files),1)
        audit=json.loads(files[0].read_text());self.assertEqual(audit['before']['TresPerNode'],audit['after']['TresPerNode'])
        self.assertEqual(audit['plan_fingerprint'],self.plan['fingerprint'])
        self.assertEqual(audit['intent_sha256'],task['batch2_intent_sha256'])
    def test_running_job_and_pending_to_running_race_never_cancel(self):
        self.prepare();self.accepted()
        with patch.object(rq,'_command',return_value=self.record(state='RUNNING')), patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget(),[]);update.assert_not_called()
        with patch.object(rq,'_command',side_effect=[self.record(),self.record(state='RUNNING')]), \
             patch.object(bs.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','job no longer pending')) as update:
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED')
        self.assertEqual(update.call_count,1);self.assertEqual(self.task['job_id'],'123')
    def test_wrong_scheduler_identity_and_claim_never_update(self):
        self.prepare();self.accepted()
        with patch.object(rq,'_command',return_value=self.record(Account='other')),patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED');update.assert_not_called()
        Path(self.task['batch2_intent']).with_suffix('.claim.json').write_text('{}')
        with patch.object(rq,'_command') as command,patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED');command.assert_not_called();update.assert_not_called()
    def test_tampered_intent_or_script_refused(self):
        self.prepare();self.accepted()
        path=Path(self.task['batch2_intent']);path.write_text(path.read_text()+' ')
        with patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED');update.assert_not_called()
    def test_uncertain_or_duplicate_job_name_never_updates(self):
        self.prepare();self.accepted()
        self.queue['124'] = dict(self.queue['123'])
        with patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED');update.assert_not_called()
        self.task['status'] = 'UNCERTAIN'
        with patch.object(rq,'_command') as query,patch.object(bs.subprocess,'run') as update:
            self.assertEqual(self.retarget(),[]);query.assert_not_called();update.assert_not_called()
    def test_timeout_retains_durable_deferred_receipt_no_replacement(self):
        self.prepare();self.accepted()
        with patch.object(rq,'_command',return_value=self.record()), \
             patch.object(bs.subprocess,'run',side_effect=subprocess.TimeoutExpired('scontrol',30)):
            self.assertEqual(self.retarget()[0]['status'],'DEFERRED')
        audit=json.loads(next((self.path.parent/'scheduling/updates').glob('*.json')).read_text())
        self.assertEqual(audit['status'],'DEFERRED');self.assertEqual(self.task['job_id'],'123')
    def test_future_formal_same_dataset_l40s_gate_and_two_gpu_exclusions(self):
        self.prepare();policy=load_policy(self.plan)
        task=next(t for t in self.state['tasks'].values() if t['kind']=='experiment' and t['resource']=='cgics_mllm')
        gate=self.state['tasks'][task['dependencies'][0]];gate.update(status='PASS',job_id='555')
        hardware={'555':{'partition':'l40sp','state':'COMPLETED','exit_code':'0:0'}}
        got=effective_profile(self.plan,self.state,task,policy,hardware)
        expected=rq._profile(self.plan,task)
        self.assertEqual({k:v for k,v in got.items() if k!='partition'},{k:v for k,v in expected.items() if k!='partition'})
        self.assertEqual(got['partition'],'h200p,l40sp');self.assertEqual(got['gpus'],2)
        self.assertEqual(set(got['exclude'].split(',')),{'2FS9B74','A484652X3B10601'})
        self.assertEqual(effective_profile(self.plan,self.state,task,policy,{})['partition'],'h200p')
    def test_parent_explicit_opt_in_passes_scheduling_with_shared_reservations(self):
        self.prepare()
        with patch.object(rq,'tick') as tick:batch2.tick_batch(self.parent,scheduling=True)
        self.assertEqual(tick.call_count,3)
        self.assertTrue(all(c.kwargs['scheduling'] is True for c in tick.call_args_list))
    def test_tick_calls_batch2_intent_retarget_not_legacy(self):
        self.prepare()
        with patch.object(rq,'_scheduler_snapshot',return_value=({},{})), \
             patch('ics.scheduling.hardware_evidence',return_value=({},None)), \
             patch('ics.scheduling.retarget_pending_smokes') as old, \
             patch.object(bs,'retarget_pending_smokes',return_value=[]) as new, patch.object(rq,'_submit'):
            rq.tick(self.path,scheduling=True,external_active=[None]*60)
        old.assert_not_called();new.assert_called_once()

if __name__=='__main__':unittest.main()
