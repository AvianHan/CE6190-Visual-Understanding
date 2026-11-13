import copy
import errno
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ics import repair as R
from ics.orchestration import fingerprint

class RepairPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        (self.run/'source').mkdir();(self.run/'source/snapshot.json').write_text('{"files":{}}')
        self.plan={'run_dir':str(self.run),'project_root':str(self.run/'source'),'fingerprint':'p'*64,'cluster':{'launcher_python':'python3'}}
        self.policy={'plan_fingerprint':self.plan['fingerprint'],'worker_source':self.plan['project_root'],
            'controller_source':str(self.run/'repair/source'),'exclude_nodes':[R.BAD_NODE],
            'worker_snapshot_sha256':R.digest(self.run/'source/snapshot.json')}
        self.policy['fingerprint']=fingerprint(self.policy)
        R._atomic_json(self.run/'repair/policy.json',self.policy)
        self.task={'key':'model--coco--0','kind':'experiment','resource':'insid3','index':0,'dependencies':['smoke'],'attempts':1,'status':'PENDING'}
        self.state={'policy':{'user':'hansy'},'tasks':{self.task['key']:self.task}}
    def test_immutable_record_idempotent_and_changed_rejected(self):
        path=self.run/'immutable.json';sha=R.immutable_json(path,{'a':1})
        self.assertEqual(sha,R.immutable_json(path,{'a':1}))
        with self.assertRaises(ValueError):R.immutable_json(path,{'a':2})
        self.assertEqual(json.loads(path.read_text()),{'a':1})
        self.assertFalse(list(self.run.glob('immutable.json.*')))
    def test_unique_attempt_names_are_stable_length_and_sealed(self):
        first=R.make_worker_launch(self.run/'plan.json',self.plan,self.state,self.task)
        first_name=self.task['job_name'];first_intent=self.task['repair_attempt']['intent_path']
        self.assertIn('repair_worker.py',first[1])
        self.assertEqual(first,R.make_worker_launch(self.run/'plan.json',self.plan,self.state,self.task))
        self.task['attempts']=2
        R.make_worker_launch(self.run/'plan.json',self.plan,self.state,self.task)
        self.assertNotEqual(first_name,self.task['job_name']);self.assertEqual(len(first_name),len(self.task['job_name']))
        self.assertNotEqual(first_intent,self.task['repair_attempt']['intent_path']);self.assertTrue(Path(first_intent).exists())
    def test_exclusions_union_and_cpu_unchanged(self):
        self.assertEqual(R.effective_exclusions({'gpus':2,'exclude':'2FS9B74'},self.policy)['exclude'],'2FS9B74,'+R.BAD_NODE)
        self.assertEqual(R.effective_exclusions({'gpus':1},self.policy)['exclude'],R.BAD_NODE)
        self.assertEqual(R.effective_exclusions({'gpus':0,'partition':'cpup'},self.policy),{'gpus':0,'partition':'cpup'})
    def test_policy_cannot_change_scientific_source(self):
        bad=copy.deepcopy(self.policy);bad['worker_source']='somewhere';bad['fingerprint']=fingerprint({k:v for k,v in bad.items() if k!='fingerprint'})
        R._atomic_json(self.run/'repair/policy.json',bad)
        with self.assertRaisesRegex(ValueError,'identity'):R.load_repair_policy(self.plan)
    def test_reset_preserves_pass_and_failure_history(self):
        failed=copy.deepcopy(self.task);failed.update(status='FAILED',job_id='4',exit_code='1:0',attempts=1)
        passed=copy.deepcopy(self.task);passed.update(key='pass',status='PASS',job_id='3',ended_at=4)
        blocked=copy.deepcopy(self.task);blocked.update(key='blocked',status='BLOCKED',attempts=0)
        state={'tasks':{'fail':failed,'pass':passed,'blocked':blocked},'last_error':'old','capacity':{},'repair_circuit_breaker':{'old':1}}
        new=R.reset_state(state,{'failed':['fail'],'preserved':['pass'],'blocked':['blocked']},self.policy)
        self.assertEqual(new['tasks']['pass'],passed);self.assertNotIn('job_id',new['tasks']['fail'])
        self.assertEqual(new['tasks']['fail']['attempt_history'][0]['job_id'],'4');self.assertEqual(new['tasks']['fail']['attempts'],1)
        self.assertEqual(new['tasks']['blocked']['status'],'PENDING');self.assertNotIn('repair_circuit_breaker',new)
        self.assertEqual(state['tasks']['fail']['job_id'],'4')
    def test_archive_is_idempotent_and_does_not_move_results(self):
        key=self.task['key'];job=self.run/'jobs'/key;job.mkdir(parents=True);(job/'runner.log').write_text('old failure')
        result=self.run/'results/model/episodes';result.mkdir(parents=True);(result/'ok.json').write_text('keep')
        R._atomic_json(self.run/'rolling_markers'/(key+'.json'),{'status':'FAIL'})
        proof={'failed':[key],'history_job_ids':{key:'123'}}
        first=R.archive_failed(self.run,proof);second=R.archive_failed(self.run,proof)
        self.assertEqual((result/'ok.json').read_text(),'keep');self.assertTrue(any(r['status']=='MOVED' for r in first))
        self.assertTrue(any(r['status']=='ALREADY_ARCHIVED' for r in second))
        self.assertEqual((self.run/'repair/archives'/key/'job-123/job/runner.log').read_text(),'old failure')
    def test_unknown_or_other_node_failure_is_not_auto_retried(self):
        key=self.task['key'];p=self.run/'jobs'/key/'runner.log';p.parent.mkdir(parents=True);task={**self.task,'job_id':'7'}
        p.write_text('CUDA is unavailable; run inside an allocated GPU job')
        self.assertEqual(R.failure_category(self.run,task,R.BAD_NODE),'CUDA_UNAVAILABLE_ON_EXCLUDED_NODE')
        with self.assertRaisesRegex(ValueError,'another node'):R.failure_category(self.run,task,'other-node')
        p.write_text('ValueError: different failure')
        with self.assertRaisesRegex(ValueError,'unreviewed'):R.failure_category(self.run,task,R.BAD_NODE)
    def test_episode_errors_are_exposed_without_altering_rows(self):
        key=self.task['key'];output=self.run/'results/model';episodes=output/'episodes';episodes.mkdir(parents=True)
        ok={'status':'ok','episode_id':'ok'};bad={'status':'error','episode_id':'bad','error':'bad input'}
        (episodes/'ok.json').write_text(json.dumps(ok));(episodes/'error.json').write_text(json.dumps(bad))
        report=R.inspect_episode_errors({'smoke':[],'tasks':[{'id':key,'output':str(output)}]},[key])
        self.assertEqual(len(report['error_records']),1);self.assertEqual(report['outputs'][0]['existing_records'],2)
        self.assertEqual(json.loads((episodes/'ok.json').read_text()),ok)
    def test_publication_visibility_retry_does_not_swallow_permission(self):
        script=Path(__file__).resolve().parents[1]/'scripts/repair_worker.py'
        spec=importlib.util.spec_from_file_location('repair_worker_visibility',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module.time,'sleep'),patch.object(Path,'read_bytes',side_effect=[FileNotFoundError(errno.ENOENT,'late'),b'ready']):
            self.assertEqual(module.visible_read('file'),b'ready')
        with patch.object(module.time,'sleep'),patch.object(Path,'read_bytes',side_effect=PermissionError(errno.EACCES,'denied')) as read:
            with self.assertRaises(PermissionError):module.visible_read('file')
            self.assertEqual(read.call_count,1)

if __name__=='__main__':unittest.main()
