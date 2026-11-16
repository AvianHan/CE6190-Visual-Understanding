"""Independent CLI integration with the exported, byte-identical deployed controller.

Boundary: fixture paths/fingerprints, manifests/predictions and Slurm are synthetic.
Recovery CLI logic, all 8 deployed controller files, real process/PID/locks, and
pause/activate/watch are executed without mocks. No model/GPU process executes.
"""
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT if (ROOT/'scripts/resume_ics_tail.py').exists() else ROOT/'tail_update_20261116'
ENGINE=PACKAGE/'scripts/resume_ics_tail.py'
spec=importlib.util.spec_from_file_location('tail_test_helpers', PACKAGE/'tests/test_resume_ics_tail.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
M=H.M

FAKE_SLURM='''#!/usr/bin/env python3
import json,os,re,sys
from pathlib import Path
p=Path(os.environ['TAIL_FAKE_DB']);d=json.loads(p.read_text());a=sys.argv[1:];cmd=Path(sys.argv[0]).name
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d));t.replace(p)
if cmd=='scontrol':
 name=a[-1];print('PartitionName='+name+' State=UP AllowQos=ALL AllowAccounts=ALL');sys.exit(0)
if cmd=='sbatch':
 jid=str(900001+len(d['submissions']));name=a[a.index('--job-name')+1]
 row={'job_id':jid,'job_name':name,'state':'PENDING','exit_code':'0:0','qos':'highpriority','node':'fixture','submit':'2026-11-16T11:00:00','start':'Unknown','end':'Unknown','partition':a[a.index('--partition')+1]}
 d['submissions'].append({'argv':a,'job_id':jid});d['history'][jid]=row;save();print(jid);sys.exit(0)
if cmd=='squeue':
 for r in d['history'].values():
  if r['state'] in ('PENDING','RUNNING'):print('|'.join(r[k] for k in ('job_id','job_name','state','qos')))
 sys.exit(0)
if cmd=='sacct':
 fields=a[a.index('--format')+1].split(',');ids=set(a[a.index('--jobs')+1].split(',')) if '--jobs' in a else None
 names={'JobIDRaw':'job_id','JobName':'job_name','State':'state','ExitCode':'exit_code','QOS':'qos','NodeList':'node','Submit':'submit','Start':'start','End':'end','Partition':'partition'}
 for jid,r in d['history'].items():
  if ids is None or jid in ids:print('|'.join(r[names[f.split('%')[0]]] for f in fields))
 sys.exit(0)
raise SystemExit('Unexpected fake Slurm command '+cmd)
'''


class TailControllerIntegration(unittest.TestCase):
    def setUp(self):
        self.base=H.TailTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.base.marker_patch.stop()  # Integration uses real evidence checks.
        self.run=self.base.run;self.plan=copy.deepcopy(self.base.plan);self.state=copy.deepcopy(self.base.state)
        self.tmp=self.run.parent
        self.env=dict(os.environ);self.bin=self.tmp/'bin';self.bin.mkdir()
        fake=self.bin/'fake';fake.write_text(FAKE_SLURM);fake.chmod(0o755)
        for name in ('squeue','sacct','scontrol','sbatch'):(self.bin/name).symlink_to(fake)
        self.env['PATH']=str(self.bin)+':'+self.env.get('PATH','')
        self.db=self.tmp/'slurm.json';self.env['TAIL_FAKE_DB']=str(self.db)
        # Reuse synthetic episodes, retaining actual task ids, variants and dependencies.
        ep_plan=self.base.episode_fixture();ep_defs={t['id']:t for t in ep_plan['tasks']+ep_plan['smoke']}
        for task in self.plan['tasks']+self.plan['smoke']:
            if task['id'] in ep_defs:
                ep=ep_defs[task['id']]
                for k in ('manifest','output','max_episodes','num_shards','shard_index'):task[k]=ep[k]
                for p in (Path(task['output'])/'episodes').glob('*.json'):
                    r=json.loads(p.read_text());r['experiment_id']=task['config_data']['experiment_id'];H.put(p,r)
        self.plan['fingerprint']=M.fingerprint({k:v for k,v in self.plan.items() if k!='fingerprint'})
        self.fp=self.plan['fingerprint'];self.state['plan_fingerprint']=self.fp
        H.put(self.run/'plan.json',self.plan)
        exported=self.tmp/'exported'
        with gzip.open(Path(__file__).parent/'fixtures/deployed_tail_controller.json.gz','rt') as f:exported_files=json.load(f)
        for name,content in exported_files.items():
            p=exported/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        self.controller_hashes={}
        for prefix in ('source','repair/source'):
            files={}
            for p in (exported/prefix).rglob('*.py'):
                relative=p.relative_to(exported/prefix);dest=self.run/prefix/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                files[str(relative)]=M.sha(dest.read_bytes())
                if prefix=='repair/source':self.controller_hashes[str(relative)]=files[str(relative)]
            H.put(self.run/prefix/'snapshot.json',{'schema_version':1,'selection':{'runtime':str(self.tmp/'runtime')},'files':files})
        self.assertEqual(len(self.controller_hashes),8)
        def relocate(value):
            if isinstance(value,str):return value.replace('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-recovery-20261113',str(self.run))
            if isinstance(value,list):return [relocate(x) for x in value]
            if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
            return value
        scheduling=relocate(json.loads((exported/'scheduling/policy.json').read_text()));scheduling['plan_fingerprint']=self.fp
        scheduling['fingerprint']=M.fingerprint({k:v for k,v in scheduling.items() if k!='fingerprint'})
        H.put(self.run/'scheduling/policy.json',scheduling)
        repair=relocate(json.loads((exported/'repair/policy.json').read_text()));repair.update(plan_fingerprint=self.fp,scheduling_policy_fingerprint=scheduling['fingerprint'],worker_snapshot_sha256=M.sha((self.run/'source/snapshot.json').read_bytes()))
        repair['fingerprint']=M.fingerprint({k:v for k,v in repair.items() if k!='fingerprint'});self.rp=repair['fingerprint']
        H.put(self.run/'repair/policy.json',repair);H.put(self.run/'repair/applied.json',{'fixture':True})
        for key in M.SELECTED:
            task=self.state['tasks'][key];p=Path(task['repair_attempt']['intent_path']);v=json.loads(p.read_text())
            v.update(plan_fingerprint=self.fp,repair_policy_fingerprint=self.rp,worker_snapshot_sha256=repair['worker_snapshot_sha256']);H.put(p,v)
            task['repair_attempt'].update(intent_sha256=M.sha(p.read_bytes()),policy_fingerprint=self.rp)
            if key==M.INTERRUPTED:H.put(p.with_suffix('.claim.json'),{'slurm_job_id':'201500','intent_sha256':M.sha(p.read_bytes())})
        for key,task in self.state['tasks'].items():
            if task['status']=='PASS':self.marker(task)
        p=self.run/'rolling_markers'/(M.INTERRUPTED+'.json');v=json.loads(p.read_text());v['plan_fingerprint']=self.fp;H.put(p,v)
        H.put(self.run/'rolling_state.json',self.state)
        history=copy.deepcopy(self.base.history)
        for r in history.values():r.update(submit='2026-11-13T14:00:00',start='2026-11-13T14:01:00',end='2026-11-13T15:00:00',partition='l40sp')
        H.put(self.db,{'history':history,'submissions':[]})
        pins={k:M.sha((self.run/k).read_bytes()) for k in M.PINS}
        self.fixture_pins=self.tmp/'fixture-pins.json';H.put(self.fixture_pins,{'PLAN':self.fp,'REPAIR':self.rp,'PINS':pins,'SCIENTIFIC_FAILURES':M.SCIENTIFIC_FAILURES})
        self.harness=self.tmp/'tail_cli_fixture.py';self.harness.write_text("import importlib.util,json,sys\nfrom pathlib import Path\nspec=importlib.util.spec_from_file_location('tail_engine',"+repr(str(ENGINE))+")\nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\npins=json.loads(Path("+repr(str(self.fixture_pins))+").read_text())\nfor k,v in pins.items():setattr(m,k,v)\nraise SystemExit(m.main())\n")
        self.addCleanup(self.stop_manager)

    def marker(self,task):
        key=task['key'];H.put(self.run/'rolling_markers'/(key+'.json'),{'key':key,'status':'PASS','plan_fingerprint':self.fp,'slurm_job_id':task['job_id']})
        if task['kind'] in ('smoke','experiment'):H.put(self.run/'jobs'/key/'complete.json',{'task_id':key,'status':'PASS','plan_fingerprint':self.fp,'slurm_job_id':task['job_id']})

    def command(self,script,*args,expect=0):
        r=subprocess.run([sys.executable,'-I','-B','-u',str(script),*args,'--run-dir',str(self.run)],env=self.env,capture_output=True,text=True,timeout=45)
        self.assertEqual(r.returncode,expect,r.stdout[-5000:]+'\n'+r.stderr[-5000:]);return r

    def control(self,*args,expect=0):return self.command(self.run/'repair/source/scripts/repair_ics.py',*args,expect=expect)

    def stop_manager(self):
        if (self.run/'manager.json').exists():self.control('pause')

    def state_now(self):return json.loads((self.run/'rolling_state.json').read_text())

    def wait_submissions(self,count):
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            d=json.loads(self.db.read_text())
            if len(d['submissions'])==count and not any(t['status']=='SUBMITTING' for t in self.state_now()['tasks'].values()):return d
            time.sleep(.05)
        self.fail('Submissions did not reach '+str(count)+'\n'+(self.run/'manager.log').read_text()[-6000:])

    def finish_current_jobs(self):
        state=self.state_now();db=json.loads(self.db.read_text())
        for task in state['tasks'].values():
            if task['status']=='SUBMITTED':
                db['history'][task['job_id']].update(state='COMPLETED',end='2026-11-16T12:00:00');self.marker(task)
        H.put(self.db,db)

    def test_actual_cli_pause_activate_full_tail_and_idempotence(self):
        original_predictions=H.tree(self.run/'results')
        self.control('activate');first_pid=json.loads((self.run/'manager.json').read_text())['pid']
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if json.loads((self.run/'manager.json').read_text()).get('phase')=='waiting':break
            time.sleep(.05)
        before=copy.deepcopy(self.state_now())
        self.command(self.harness,'audit')
        self.assertEqual(json.loads(self.db.read_text())['submissions'],[])
        self.command(self.harness,'resume');db=self.wait_submissions(10)
        second_pid=json.loads((self.run/'manager.json').read_text())['pid'];self.assertNotEqual(first_pid,second_pid)
        after=self.state_now()
        for key,task in before['tasks'].items():
            if task['status']=='PASS' or key in M.SCIENTIFIC_FAILURES:self.assertEqual(after['tasks'][key],task)
        for key in M.SELECTED:
            self.assertEqual(after['tasks'][key]['status'],'SUBMITTED');self.assertEqual(after['tasks'][key]['attempts'],before['tasks'][key]['attempts']+1)
            intent=json.loads(Path(after['tasks'][key]['repair_attempt']['intent_path']).read_text())
            self.assertEqual(intent['worker_source'],str(self.run/'source'))
            self.assertEqual(intent['plan_fingerprint'],self.fp)
        self.assertEqual(after['scheduling']['effective_policy']['max_inflight'],60)
        attempts={key:after['tasks'][key]['attempts'] for key in M.SELECTED}
        self.command(self.harness,'resume')
        self.assertEqual(len(json.loads(self.db.read_text())['submissions']),10)
        self.assertEqual({key:self.state_now()['tasks'][key]['attempts'] for key in M.SELECTED},attempts)
        self.control('pause');self.finish_current_jobs();self.control('watch','--once');self.wait_submissions(24)
        self.finish_current_jobs();self.control('watch','--once');self.wait_submissions(25)
        self.assertEqual(self.state_now()['tasks']['collect']['status'],'SUBMITTED')
        self.finish_current_jobs();result=self.control('watch','--once',expect=2)
        final=self.state_now();self.assertEqual(sum(t['status']=='PASS' for t in final['tasks'].values()),1451)
        self.assertEqual(sum(t['status']=='FAILED' for t in final['tasks'].values()),4)
        self.assertIn('COMPLETE_WITH_FAILURES',result.stdout)
        self.assertEqual(H.tree(self.run/'results'),original_predictions)
        for name,digest in self.controller_hashes.items():self.assertEqual(M.sha((self.run/'repair/source'/name).read_bytes()),digest)

    def test_live_manager_wrong_pid_record_refuses_without_signal_or_reset(self):
        self.control('activate')
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            record=json.loads((self.run/'manager.json').read_text())
            if record.get('phase')=='waiting':break
            time.sleep(.05)
        original_record=copy.deepcopy(record);original_state=(self.run/'rolling_state.json').read_bytes()
        record['pid']=os.getpid();H.put(self.run/'manager.json',record)
        try:
            result=self.command(self.harness,'resume',expect=1)
            self.assertIn('Manager command is not the prepared repair watcher',result.stderr)
            os.kill(original_record['pid'],0)
            self.assertEqual((self.run/'rolling_state.json').read_bytes(),original_state)
            self.assertFalse((self.run/M.TAG).exists())
            self.assertEqual(json.loads(self.db.read_text())['submissions'],[])
        finally:H.put(self.run/'manager.json',original_record)

    def test_damaged_prediction_refuses_before_pausing_live_controller(self):
        self.control('activate');pid=json.loads((self.run/'manager.json').read_text())['pid']
        pred=next((self.run/'results'/M.INTERRUPTED/'predictions').glob('*.png'))
        pred.write_bytes(b'changed-success-output')
        result=self.command(self.harness,'resume',expect=1)
        self.assertIn('Successful prediction changed',result.stderr)
        os.kill(pid,0)
        self.assertFalse((self.run/M.TAG).exists())
        self.assertEqual(json.loads(self.db.read_text())['submissions'],[])


if __name__=='__main__':unittest.main(verbosity=2)
