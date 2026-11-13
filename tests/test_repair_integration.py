"""Real CLI/worker subprocess tests against synthetic Slurm; no GPU/model execution."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest

from test_scaling_integration import create_fixture,atomic,wait_until,unlocked

PROJECT=Path(__file__).resolve().parents[1]
BASELINE=Path(os.environ.get('ICS_TEST_LEGACY_SOURCE',str(PROJECT))).resolve()
BAD_NODE='A484652X3B10601'
STUB_RUNNER=r'''import json,sys,time
from pathlib import Path
args=sys.argv[1:]
assert '--resume' in args
out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
# Existing records are untouched; this fixture represents the frozen scientific runner.
(out/'fixture_runner.json').write_text(json.dumps({'script':__file__,'argv':args,'cwd':str(Path.cwd())}))
time.sleep(.35)
'''

class RepairSubprocessIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ics-repair-integration-')
        self.root=Path(self.temp.name)
        self.fixture=create_fixture(self.root,BASELINE,PROJECT)
        for key,value in self.fixture.items():setattr(self,key,value)
        self.addCleanup(self.cleanup)
        # The temporary original worker is independently frozen before any repair.
        (self.source/'scripts/run_experiment.py').write_text(STUB_RUNNER)
        snapshot=json.loads((self.source/'snapshot.json').read_text())
        snapshot['files']['scripts/run_experiment.py']=hashlib.sha256(STUB_RUNNER.encode()).hexdigest()
        atomic(self.source/'snapshot.json',snapshot)
        state=self.state();jobs=[]
        self.preserved_formal=self.plan['tasks'][-1]['id']
        self.failed_formal=next(t['id'] for t in self.plan['tasks'] if t['id'].startswith('insid3_full--coco'))
        self.preserved_smoke='smoke-insid3_full-coco'
        definitions={t['id']:t for t in self.plan['tasks']+self.plan['smoke']}
        for n,(key,task) in enumerate(state['tasks'].items()):
            if key=='collect':continue
            success=key in {'data','preflight',self.preserved_formal,self.preserved_smoke}
            submitted=success or task['kind']=='smoke' or key==self.failed_formal
            if not submitted:
                task.update(status='BLOCKED');task.pop('job_id',None);continue
            jid=str(200+n)
            task.update(status='SUBMITTED',job_id=jid,attempts=1)
            actual='COMPLETED' if success else 'FAILED'
            jobs.append({'job_id':jid,'job_name':task['job_name'],'state':actual,
                'exit_code':'0:0' if success else '1:0','qos':'highpriority',
                'partition':'cpup' if key in {'data','preflight'} else 'l40sp',
                'node':'HEALTHY' if success else BAD_NODE,'key':key})
            atomic(self.run/'rolling_markers'/f'{key}.json',{'status':'PASS' if success else 'FAIL',
                'key':key,'kind':task['kind'],'plan_fingerprint':self.plan['fingerprint'],'slurm_job_id':jid})
            if task['kind'] in {'smoke','experiment'}:
                jobdir=self.run/'jobs'/key;jobdir.mkdir(parents=True,exist_ok=True)
                if success:
                    atomic(jobdir/'complete.json',{'status':'PASS','task_id':key,
                        'plan_fingerprint':self.plan['fingerprint'],'slurm_job_id':jid})
                else:
                    (jobdir/'runner.log').write_text('CUDA is unavailable; run inside an allocated GPU job\n')
                original=definitions[key]
                atomic(Path(original['output'])/'episodes/existing.json',{'status':'ok','episode_id':'fixture-already-complete'})
            script=self.run/'rolling_slurm'/f'{key}.sh';script.parent.mkdir(exist_ok=True)
            script.write_text('# original attempt evidence\n')
        atomic(self.run/'rolling_state.json',state)
        atomic(self.run/'preflight.complete.json',{'status':'PASS'})
        atomic(self.scheduler/'scheduler.json',{'jobs':jobs,'submissions':[],'updates':[]})
        self.cli_scale_prepare()
        self.before_source=(self.source/'snapshot.json').read_bytes()
        self.before_plan=(self.run/'plan.json').read_bytes()
        self.before_preserved={str(p):p.read_bytes() for key in (self.preserved_smoke,self.preserved_formal)
          for p in [self.run/'rolling_markers'/f'{key}.json',self.run/'jobs'/key/'complete.json']}

    def cleanup(self):
        path=self.run/'manager.json'
        if path.exists():
            record=json.loads(path.read_text());pid=record.get('pid')
            if pid and record.get('source')==str(self.run/'repair/source'):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:wait_until(lambda:unlocked(self.run/'manager.lock'),timeout=8)
                except AssertionError:os.kill(pid,signal.SIGKILL)
        self.temp.cleanup()

    def state(self):return json.loads((self.run/'rolling_state.json').read_text())
    def scheduler_state(self):return json.loads((self.scheduler/'scheduler.json').read_text())
    def cli_scale_prepare(self):
        result=subprocess.run([sys.executable,str(PROJECT/'scripts/scale_ics.py'),'prepare',
            '--run-dir',str(self.run),'--expected-plan-fingerprint',self.plan['fingerprint']],
            env=self.env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    def cli(self,verb,*args,success=True):
        result=subprocess.run([sys.executable,str(PROJECT/'scripts/repair_ics.py'),verb,
            '--run-dir',str(self.run),*args],env=self.env,capture_output=True,text=True,timeout=40)
        if success:self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:self.assertNotEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result
    def prepare(self,success=True):
        return self.cli('prepare','--expected-plan-fingerprint',self.plan['fingerprint'],
            '--expected-successful-experiments','1','--expected-failed','6','--expected-blocked','22',success=success)
    def assert_preserved(self):
        self.assertEqual((self.source/'snapshot.json').read_bytes(),self.before_source)
        self.assertEqual((self.run/'plan.json').read_bytes(),self.before_plan)
        for p,value in self.before_preserved.items():self.assertEqual(Path(p).read_bytes(),value)

    def test_cli_prepare_apply_activate_and_original_worker_without_mutable_state(self):
        self.prepare();self.prepare()
        self.cli('apply');after_apply=(self.run/'rolling_state.json').read_bytes()
        self.cli('apply');self.assertEqual((self.run/'rolling_state.json').read_bytes(),after_apply)
        self.cli('activate','--interval','10')
        wait_until(lambda:len(self.scheduler_state()['submissions'])==5)
        wait_until(lambda:json.loads((self.run/'manager.json').read_text()).get('phase')=='waiting')
        jobs_before=self.scheduler_state()['submissions'][:]
        self.cli('activate','--interval','10')
        self.assertEqual(self.scheduler_state()['submissions'],jobs_before)
        self.cli('pause')
        self.assertTrue(unlocked(self.run/'manager.lock'))
        # A repeated apply after new attempts have been accepted is a no-op.
        accepted_state=(self.run/'rolling_state.json').read_bytes()
        accepted_scripts={str(p):p.read_bytes() for p in (self.run/'rolling_slurm').glob('*.sh')}
        self.cli('apply')
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),accepted_state)
        for p,data in accepted_scripts.items():self.assertEqual(Path(p).read_bytes(),data)
        jobs=[j for j in self.scheduler_state()['jobs'] if j['job_id'] in jobs_before]
        self.assertTrue(all('--exclude' in j['argv'] and BAD_NODE in j['argv'][j['argv'].index('--exclude')+1] for j in jobs))
        for j in jobs:
            if 'cgics' in j['key']:self.assertIn('2FS9B74',j['argv'][j['argv'].index('--exclude')+1])
        job=next(j for j in jobs if j['key']=='smoke-insid3_full-lvis')
        intent=list((self.run/'repair/intents'/job['key']).glob('attempt-*.json'))[0]
        target=next(t for t in self.plan['smoke'] if t['id']==job['key'])
        existing=Path(target['output'])/'episodes/existing.json';existing_bytes=existing.read_bytes()
        state_file=self.run/'rolling_state.json';hidden=self.run/'fixture-state-temporarily-hidden.json'
        state_file.rename(hidden)
        env=dict(self.env,SLURM_JOB_ID=job['job_id'],SLURM_JOB_NAME=job['job_name'],CUDA_VISIBLE_DEVICES='0')
        try:
            first=subprocess.Popen(['bash',job['script']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            second=subprocess.Popen(['bash',job['script']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            output1=first.communicate(timeout=25);output2=second.communicate(timeout=25)
            self.assertEqual(sorted([first.returncode,second.returncode]),[0,1],repr((output1,output2)))
            marker=self.run/'rolling_markers'/f"{job['key']}.json";marker_bytes=marker.read_bytes()
            repeated=subprocess.run(['bash',job['script']],env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(repeated.returncode,0)
            self.assertEqual(marker.read_bytes(),marker_bytes)
            self.assertEqual(json.loads(marker_bytes)['status'],'PASS')
            # Incomplete/corrupt attempt claims must fail closed without marker writes.
            claim=intent.with_suffix('.claim.json');claim.write_text('{broken')
            repeated=subprocess.run(['bash',job['script']],env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(repeated.returncode,0)
            self.assertEqual(marker.read_bytes(),marker_bytes)
            intent_bytes=intent.read_bytes();intent.write_text('{}')
            repeated=subprocess.run(['bash',job['script']],env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(repeated.returncode,0)
            self.assertIn('hash mismatch',repeated.stderr)
            self.assertEqual(marker.read_bytes(),marker_bytes)
            intent.write_bytes(intent_bytes)
        finally:hidden.rename(state_file)
        proof=json.loads((Path(target['output'])/'fixture_runner.json').read_text())
        self.assertEqual(Path(proof['script']).resolve(),self.source/'scripts/run_experiment.py')
        self.assertEqual(Path(proof['cwd']),self.source)
        self.assertIn('--resume',proof['argv'])
        self.assertEqual(existing.read_bytes(),existing_bytes)
        self.assert_preserved()

    def test_live_job_refuses_preparation_without_archiving(self):
        scheduler=self.scheduler_state();scheduler['jobs'][0]['state']='RUNNING'
        atomic(self.scheduler/'scheduler.json',scheduler)
        before=(self.run/'rolling_state.json').read_bytes()
        result=self.prepare(success=False)
        self.assertIn('queued/running',result.stderr)
        self.assertFalse((self.run/'repair/policy.json').exists())
        self.assertFalse((self.run/'repair/archives').exists())
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),before)

    def test_interrupted_apply_rolls_forward_once_and_preserves_evidence(self):
        self.prepare();before=(self.run/'rolling_state.json').read_bytes()
        controller=self.run/'repair/source'
        injected=r'''import copy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/scripts')
import repair_ics as m
original=m.archive_failed
def interrupted(run,proof):
    first=copy.deepcopy(proof);first['failed']=first['failed'][:1]
    original(run,first)
    raise RuntimeError('fixture interruption after first archived task')
m.archive_failed=interrupted
m.main(['apply','--run-dir',sys.argv[2]])
'''
        result=subprocess.run([sys.executable,'-c',injected,str(controller),str(self.run)],
            env=self.env,capture_output=True,text=True,timeout=25)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('fixture interruption',result.stderr)
        self.assertTrue((self.run/'repair/transaction.json').exists())
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),before)
        archived=list((self.run/'repair/archives').rglob('runner.log'));self.assertEqual(len(archived),1)
        first_hash=hashlib.sha256(archived[0].read_bytes()).hexdigest()
        self.cli('apply')
        self.assertEqual(hashlib.sha256(archived[0].read_bytes()).hexdigest(),first_hash)
        all_archived=list((self.run/'repair/archives').rglob('runner.log'));self.assertEqual(len(all_archived),6)
        applied=(self.run/'rolling_state.json').read_bytes();self.cli('apply')
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),applied)
        self.assertEqual(len(list((self.run/'repair/archives').rglob('runner.log'))),6)
        self.assert_preserved()

    def test_existing_error_episode_stops_before_policy_or_archive(self):
        target=next(t for t in self.plan['tasks'] if t['id']==self.failed_formal)
        error=Path(target['output'])/'episodes/error.json'
        atomic(error,{'status':'error','episode_id':'failed-episode','error':'model error'})
        before=error.read_bytes();result=self.prepare(success=False)
        self.assertIn('error episodes',result.stderr)
        self.assertFalse((self.run/'repair/policy.json').exists())
        self.assertFalse((self.run/'repair/archives').exists())
        self.assertEqual(error.read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
