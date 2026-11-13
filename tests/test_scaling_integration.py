"""Synthetic subprocess fixture support; never allocates GPUs or invokes a model."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import time

FAKE_SLURM = r'''
import fcntl,json,os,pwd,sys
from pathlib import Path
root=Path(os.environ['ICS_TEST_SCHEDULER'])
root.mkdir(exist_ok=True)
path=root/'scheduler.json'
with (root/'lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=json.loads(path.read_text())
 name=Path(sys.argv[0]).name
 args=sys.argv[1:]
 def flag(key,default=None):
  if key in args: return args[args.index(key)+1]
  return next((v.split('=',1)[1] for v in args if v.startswith(key+'=')),default)
 if name=='squeue':
  for j in state['jobs']:
   if j['state'] in ('PENDING','RUNNING'):
    print('|'.join(j[k] for k in ('job_id','job_name','state','qos')))
 elif name=='sacct':
  fmt=flag('--format','JobIDRaw,JobName,State,ExitCode,QOS').split(',')
  names={'JobIDRaw':'job_id','JobID':'job_id','JobName':'job_name','State':'state','ExitCode':'exit_code','QOS':'qos','Partition':'partition','NodeList':'node'}
  for j in state['jobs']:
   print('|'.join(j.get(names.get(f.split('%')[0],f),'') for f in fmt))
 elif name=='sbatch':
  jid=str(1000+len(state['jobs']))
  script=args[-1]
  j=dict(job_id=jid,job_name=flag('--job-name'),state='PENDING',exit_code='0:0',qos=flag('--qos'),
   partition=flag('--partition'),time=flag('--time'),script=script,key=Path(script).stem,
   node='None assigned',argv=args,uid=str(os.getuid()))
  state['jobs'].append(j)
  state['submissions'].append(jid)
  print('accepted-unknown' if (root/'lose-ack').exists() else jid)
 elif name=='scontrol':
  if 'update' in args:
   options=dict(arg.split('=',1) for arg in args if '=' in arg)
   job=next(j for j in state['jobs'] if j['job_id']==options['JobId'])
   if job['state']!='PENDING':
    print('Job is no longer pending',file=sys.stderr);sys.exit(1)
   if 'Partition' in options: job['partition']=options['Partition']
   if 'TimeLimit' in options: job['time']=options['TimeLimit']
   state['updates'].append(options)
  elif 'job' in args:
   jid=args[args.index('job')+1]
   job=next(j for j in state['jobs'] if j['job_id']==jid)
   print('JobId='+jid+' JobName='+job['job_name']+' UserId='+pwd.getpwuid(os.getuid()).pw_name+'('+str(os.getuid())+') Account=prj0000000262 QOS='+job['qos']+' JobState='+job['state']+' Partition='+job['partition']+' TimeLimit='+job['time']+' Command='+job['script']+' NodeList='+job['node'])
  elif 'partition' in args:
   for p in ('h200p','l40sp'):
    print('PartitionName='+p+' AllowAccounts=ALL AllowGroups=ALL AllowQos=highpriority State=UP MaxTime=3-00:00:00')
  else:
   print('Unexpected scontrol args: '+repr(args),file=sys.stderr);sys.exit(2)
 elif name=='sacctmgr':
  fmt=flag('format','Name,MaxSubmitJobsPU')
  columns=fmt.split(',')
  values={'Name':'highpriority','MaxSubmitJobsPU':'60','MaxJobsPU':'60','GrpSubmitJobs':'','MaxTRES':'','MaxTRESPU':'','GrpTRES':'','Flags':'','MaxWall':'3-00:00:00','Account':'prj0000000262','User':pwd.getpwuid(os.getuid()).pw_name,'QOS':'highpriority','DefaultQOS':'highpriority'}
  print('|'.join(values.get(f,'') for f in columns))
 else:
  print('Unexpected fake command: '+name,file=sys.stderr);sys.exit(2)
 temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(state));temp.replace(path)
'''


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value)); temp.replace(path)


def create_fixture(root:Path,baseline:Path,project:Path):
    """Construct six accepted-smoke candidates under an independently frozen worker."""
    runtime=root/'runtime';run=runtime/'runs/recovery';source=run/'source'
    source.mkdir(parents=True)
    for directory in ('ics','scripts','configs'):
        shutil.copytree(baseline/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(baseline/'ics_eval.py',source/'ics_eval.py')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}
    atomic(source/'snapshot.json',{'files':hashes,'selection':{'runtime':str(runtime),
           'suite':'configs/suite.initial.lvisfix.json','cluster':'configs/cluster.orion.recovery.json'}})
    sys.path.insert(0,str(project))
    from ics.orchestration import fingerprint,write_plan
    from ics.rolling_queue import initialize_queue
    profiles={r:{'account':'prj0000000262','partition':'h200p','qos':'highpriority','time':'12:00:00',
                'gpus':g,'python':sys.executable,'cpus':4,'memory':'32G','max_parallel':cap}
             for r,g,cap in [('insid3',1,4),('cgics_mllm',2,2)]}
    profiles['cgics_mllm']['exclude']='2FS9B74'
    preflight=dict(profiles['insid3'],partition='cpup',gpus=0)
    cluster={'schema_version':1,'launcher_python':sys.executable,'max_parallel':8,
             'resources':{'preflight':preflight,**profiles}}
    manifests={d:{'id':d,'path':str(runtime/f'{d}.json'),'dataset_root':str(runtime/'data'),
                  'cohort':'standard','prepare':{'dataset':d}}
               for d in ('coco','lvis')}
    smoke=[];formal=[]
    for name,resource in [('cgics_full','cgics_mllm'),('insid3_full','insid3'),('insid3_component_full','insid3')]:
        cfg={'experiment_id':name,'adapter':{'name':'dummy'}}
        for dataset in manifests:
            base=dict(resource=resource,config=str(run/'configs'/f'{name}.json'),config_data=cfg,
                      manifest_id=dataset,manifest=manifests[dataset]['path'],dataset_root=str(runtime/'data'),
                      shard_index=0,num_shards=1,max_episodes=None,phase='experiment')
            smoke.append(dict(base,id=f'smoke-{name}-{dataset}',phase='smoke',max_episodes=2,
                              output=str(run/'smoke'/f'{name}-{dataset}')))
            for number in range(4):
                formal.append(dict(base,id=f'{name}--{dataset}--{number}',shard_index=number,num_shards=4,
                                   output=str(run/'results'/f'{name}-{dataset}-{number}')))
    plan={'schema_version':1,'project_root':str(source),'run_dir':str(run),'cluster':cluster,
          'resources':profiles,'manifests':manifests,'smoke':smoke,'tasks':formal,
          'recovery':{'parent_run':str(runtime/'runs/old'),'parent_plan_fingerprint':'synthetic-parent'},
          'prepare_missing_manifests':False}
    plan['fingerprint']=fingerprint(plan)
    write_plan(plan)
    state=initialize_queue(run/'plan.json',max_inflight=8,shared_limit=60)
    for key in ('data','preflight'):
        state['tasks'][key].update(status='PASS',job_id='9'+str(len(key)))
    # Completed result remains authoritative even when its task is not ready again.
    state['tasks'][formal[-1]['id']].update(status='PASS',job_id='99')
    atomic(run/'rolling_state.json',state)
    fakebin=root/'bin';fakebin.mkdir()
    scheduler=root/'fake';scheduler.mkdir()
    atomic(scheduler/'scheduler.json',{'jobs':[],'submissions':[],'updates':[]})
    for name in ('squeue','sacct','sbatch','scontrol','sacctmgr'):
        path=fakebin/name;path.write_text('#!'+sys.executable+'\n'+FAKE_SLURM);path.chmod(0o755)
    env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ['PATH'],ICS_TEST_SCHEDULER=str(scheduler),
             ICS_RUNTIME=str(runtime),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return {'run':run,'source':source,'runtime':runtime,'scheduler':scheduler,'env':env,'plan':plan}


def wait_until(predicate,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            if predicate():return
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(.05)
    raise AssertionError('Timed out waiting for synthetic subprocess state')


def unlocked(path):
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get('ICS_TEST_LEGACY_SOURCE', str(PROJECT))).resolve()


class ScalingProcessIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ics-scaling-integration-')
        self.root = Path(self.temp.name)
        self.fixture = create_fixture(self.root, BASELINE, PROJECT)
        for key, value in self.fixture.items(): setattr(self, key, value)
        self.processes = []
        self.manager_pids = set()
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        # Every PID comes from a child process or this fixture's own metadata.
        for pid in self.manager_pids:
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
        for process in self.processes:
            if process.poll() is None: process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        try: wait_until(lambda: unlocked(self.run/'manager.lock'), timeout=10)
        finally: self.temp.cleanup()

    def scheduler_state(self):
        return json.loads((self.scheduler/'scheduler.json').read_text())

    def queue_state(self):
        return json.loads((self.run/'rolling_state.json').read_text())

    def legacy(self, lose_ack=False):
        if lose_ack: (self.scheduler/'lose-ack').touch()
        stream=(self.root/'legacy-manager.log').open('w')
        self.addCleanup(stream.close)
        process=subprocess.Popen([sys.executable, str(self.source/'scripts/queue_ics.py'),
            'watch','--run-dir',str(self.run),'--interval','10'],env=self.env,stdout=stream,stderr=subprocess.STDOUT)
        self.processes.append(process)
        wait_until(lambda: len(self.scheduler_state()['jobs']) == 6)
        wait_until(lambda: json.loads((self.run/'manager.json').read_text()).get('phase') == 'waiting')
        return process

    def cli(self, verb, *arguments, success=True):
        command=[sys.executable, str(PROJECT/'scripts/scale_ics.py'),verb,'--run-dir',str(self.run),*arguments]
        result=subprocess.run(command,env=self.env,capture_output=True,text=True,timeout=30)
        if success: self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else: self.assertNotEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        if (self.run/'manager.json').exists():
            metadata=json.loads((self.run/'manager.json').read_text())
            if metadata.get('source') == str(self.run/'scheduling/source'):
                self.manager_pids.add(metadata['pid'])
        return result

    def prepare(self):
        return self.cli('prepare','--expected-plan-fingerprint',self.plan['fingerprint'])

    def activate(self):
        self.cli('activate','--interval','10')
        wait_until(lambda: json.loads((self.run/'manager.json').read_text()).get('source') == str(self.run/'scheduling/source'))
        metadata=json.loads((self.run/'manager.json').read_text())
        self.manager_pids.add(metadata['pid'])
        wait_until(lambda: json.loads((self.run/'manager.json').read_text()).get('phase') == 'waiting')
        return metadata['pid']

    def complete_smokes(self):
        with (self.scheduler/'lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.scheduler_state()
            for job in state['jobs']:
                if not job['key'].startswith('smoke-'): continue
                job.update(state='COMPLETED',partition='l40sp',node='L40S-TEST-NODE',exit_code='0:0')
                atomic(self.run/'rolling_markers'/f"{job['key']}.json",{
                    'status':'PASS','key':job['key'],'plan_fingerprint':self.plan['fingerprint'],
                    'slurm_job_id':job['job_id']})
                atomic(self.run/'jobs'/job['key']/'complete.json',{
                    'status':'PASS','task_id':job['key'],'plan_fingerprint':self.plan['fingerprint'],
                    'slurm_job_id':job['job_id']})
            atomic(self.scheduler/'scheduler.json',state)

    def test_real_handoff_and_kill_resume_preserve_jobs_pass_and_worker_source(self):
        previous=self.legacy()
        before_plan=(self.run/'plan.json').read_bytes()
        before_snapshot=(self.source/'snapshot.json').read_bytes()
        previous_jobs={j['key']:j['job_id'] for j in self.scheduler_state()['jobs']}
        completed=self.plan['tasks'][-1]['id']
        self.prepare()
        self.assertIsNone(previous.poll(), 'prepare must not stop the existing manager')
        manager=self.activate()
        previous.wait(timeout=10)
        self.assertEqual(previous.returncode,0)
        self.assertNotEqual(manager,previous.pid)
        self.assertEqual({j['key']:j['job_id'] for j in self.scheduler_state()['jobs']},previous_jobs)
        self.assertEqual(self.queue_state()['tasks'][completed]['status'],'PASS')
        self.assertTrue(all(j['partition']=='l40sp' for j in self.scheduler_state()['jobs']))
        # Kill the scaled controller only. Accepted worker jobs survive unchanged.
        os.kill(manager,signal.SIGKILL)
        wait_until(lambda: unlocked(self.run/'manager.lock'))
        resumed=self.activate()
        self.assertNotEqual(resumed,manager)
        self.assertEqual({j['key']:j['job_id'] for j in self.scheduler_state()['jobs']},previous_jobs)
        self.assertEqual(len(self.scheduler_state()['submissions']),6)
        # CPU fixture marks six synthetic hardware checks complete; no inference ran.
        self.complete_smokes()
        wait_until(lambda: len(self.scheduler_state()['jobs']) > 6, timeout=20)
        wait_until(lambda: json.loads((self.run/'manager.json').read_text()).get('phase') == 'waiting')
        state=self.scheduler_state()
        jobs=[j for j in state['jobs'] if not j['key'].startswith('smoke-')]
        self.assertEqual(len(jobs),len(self.plan['tasks'])-1)
        self.assertNotIn(completed,[j['key'] for j in jobs])
        self.assertTrue(all(j['partition']=='h200p,l40sp' for j in jobs))
        self.assertTrue(all(j['time']=='12:00:00' for j in jobs))
        for job in jobs:
            script=Path(job['script']).read_text()
            self.assertIn(str(self.source/'scripts/queue_ics.py'),script)
            self.assertNotIn(str(self.run/'scheduling/source/scripts/queue_ics.py'),script)
        self.assertEqual((self.run/'plan.json').read_bytes(),before_plan)
        self.assertEqual((self.source/'snapshot.json').read_bytes(),before_snapshot)
        for name,digest in json.loads(before_snapshot)['files'].items():
            self.assertEqual(hashlib.sha256((self.source/name).read_bytes()).hexdigest(),digest)

    def test_scaled_submission_budget_counts_other_shared_jobs(self):
        previous=self.legacy()
        self.cli('prepare','--expected-plan-fingerprint',self.plan['fingerprint'],'--shared-limit','8')
        self.activate()
        previous.wait(timeout=10)
        with (self.scheduler/'lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.scheduler_state()
            for number in range(7):
                state['jobs'].append(dict(job_id=str(9000+number),job_name='unrelated-'+str(number),
                    state='RUNNING',exit_code='0:0',qos='highpriority',partition='h200p',time='12:00:00',
                    script='/unrelated/job.sh',key='unrelated-'+str(number),node='OTHER-NODE'))
            atomic(self.scheduler/'scheduler.json',state)
        self.complete_smokes()
        wait_until(lambda: len(self.scheduler_state()['submissions']) == 7, timeout=20)
        wait_until(lambda: json.loads((self.run/'manager.json').read_text()).get('phase') == 'waiting')
        state=self.scheduler_state()
        ours=[job for job in state['jobs'] if not job['key'].startswith(('smoke-','unrelated-'))]
        self.assertEqual(len(ours),1, 'seven unrelated highpriority jobs leave one of eight slots')
        self.assertEqual(self.queue_state()['scheduling']['effective_policy']['shared_limit'],8)
        self.assertEqual(sum(job['state'] in {'PENDING','RUNNING'} for job in state['jobs']),8)
        self.assertEqual(len(state['updates']),6)
        self.assertTrue(all(event['JobId'] not in {str(9000+i) for i in range(7)} for event in state['updates']))

    def test_failed_prepare_or_tampered_controller_never_stops_legacy_manager(self):
        previous=self.legacy()
        jobs=self.scheduler_state()['submissions']
        self.cli('prepare','--expected-plan-fingerprint','wrong-fingerprint',success=False)
        self.assertIsNone(previous.poll())
        self.prepare()
        target=self.run/'scheduling/source/ics/rolling_queue.py'
        target.write_text(target.read_text()+'\n# corruption for rejection test\n')
        self.cli('activate','--interval','10',success=False)
        self.assertIsNone(previous.poll())
        self.assertFalse(unlocked(self.run/'manager.lock'))
        self.assertEqual(self.scheduler_state()['submissions'],jobs)

    def test_uncertain_accepted_jobs_are_reconciled_without_duplicate_submission(self):
        previous=self.legacy(lose_ack=True)
        self.assertTrue(any(t['status']=='UNCERTAIN' for t in self.queue_state()['tasks'].values()))
        (self.scheduler/'lose-ack').unlink()
        self.prepare(); self.activate()
        previous.wait(timeout=10)
        smokes=[t for t in self.queue_state()['tasks'].values() if t['kind']=='smoke']
        self.assertEqual(len(smokes),6)
        self.assertTrue(all(t.get('job_id') for t in smokes))
        self.assertEqual(len(self.scheduler_state()['jobs']),6)
        self.assertEqual(len(set(t['job_id'] for t in smokes)),6)


if __name__ == '__main__':
    unittest.main()
