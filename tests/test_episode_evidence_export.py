import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('episodes',Path(__file__).resolve().parents[1]/'scripts/export_episode_evidence.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class EpisodeExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.home=Path(self.tmp.name);self.run=self.home/'run';self.run.mkdir()
        self.out=self.home/'export'
        self.ok=self.run/'results/model/coco_standard'
        self.failed=self.run/'results/model/lvis_reference'
        self.smoke=self.run/'smoke/model/coco'
        tasks=[]
        for i,(output,manifest) in enumerate(((self.ok,'coco_standard'),(self.ok,'coco_standard'),(self.failed,'lvis_reference'))):
            tasks.append({'id':'formal'+str(i),'config_data':{'experiment_id':'model'},'output':str(output),
                          'manifest_id':manifest,'manifest':'/external/'+manifest+'.json','shard_index':i if i<2 else 0,
                          'num_shards':2 if i<2 else 1,'max_episodes':None})
        smoke={'id':'smoke-model-coco','config_data':{'experiment_id':'model'},'output':str(self.smoke),
               'manifest_id':'coco_standard','manifest':'/external/coco_standard.json','shard_index':0,'num_shards':1,'max_episodes':2}
        self.plan={'fingerprint':M.EXPECTED,'tasks':tasks,'smoke':[smoke]}
        self.state={'plan_fingerprint':M.EXPECTED,'policy':{'user':'hansy'},'tasks':{
            'formal0':{'status':'PASS','job_id':'10'},'formal1':{'status':'FAILED','job_id':'11'},
            'formal2':{'status':'FAILED','job_id':'12'},'smoke-model-coco':{'status':'PASS','job_id':'13'}}}
        M.dump(self.run/'plan.json',self.plan);M.dump(self.run/'rolling_state.json',self.state)
        M.dump(self.run/'manager.json',{'phase':'stopped'})
        self.raw=b'{\r\n "status":"ok", "episode_id":"same-id", "iou":0.5, "shard_index":0, "num_shards":2\r\n}\r\n'
        self.put(self.ok/'episodes/same.json',self.raw)
        self.put(self.failed/'episodes/same.json',b'{"status":"ok","episode_id":"same-id","iou":0.7}\n')
        self.put(self.smoke/'episodes/smoke.json',b'{"status":"ok","episode_id":"smoke-id"}\n')
        self.put(self.ok/'predictions/huge/nested/mask.png',b'NEVER_READ_MASK')
        self.put(self.ok/'locks/metadata.json',b'NEVER_READ_LOCK')
        self.put(self.ok/'attempts/old.json',b'OMITTED_ATTEMPT_HISTORY')
        self.put(self.run/'data/external.json',b'NEVER_READ_DATA')
        self.put(self.run/'source/worker.py',b'print("frozen")\n')
        M.dump(self.run/'source/snapshot.json',{'files':{'worker.py':hashlib.sha256(b'print("frozen")\n').hexdigest()}})
        M.dump(self.ok/'run.json',{'expected_episodes':100})
        M.dump(self.ok/'adapter.json',{'model':'same'})

    def put(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

    def collect(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return M.collect(self.run,self.out,self.home,with_scheduler=False,**kwargs)

    def envelopes(self):
        return [json.loads(line) for line in (self.out/'episodes.jsonl').read_text().splitlines()]

    def test_direct_scan_once_and_never_recursive_prescan(self):
        original=os.scandir; scanned=[]
        def limited(path):
            path=Path(path)
            if self.run==path or self.run in path.parents:
                self.assertIn(path.name,('episodes','shards'),'Unexpected recursive result scan')
                if path.name=='episodes': scanned.append(str(path))
            return original(path)
        with patch.object(M.os,'scandir',side_effect=limited): self.collect()
        self.assertEqual(len(scanned),3)
        self.assertEqual(len(scanned),len(set(scanned)))
        self.assertEqual(len(self.envelopes()),3)

    def test_old_jsonl_prescan_deadline_trap_regression(self):
        # A read of predictions or a whole result root would stall and consume the deadline.
        original=os.scandir
        def stall_if_old_scan(path):
            p=Path(path)
            if self.run==p or self.run in p.parents:
                if p.name!='episodes': raise AssertionError('Old recursive scan would exhaust 300-second budget here')
            return original(path)
        with patch.object(M.os,'scandir',side_effect=stall_if_old_scan): self.collect(metadata_seconds=0)
        self.assertEqual(len(self.envelopes()),3)
        self.assertTrue((self.out/'state_after.json').exists())

    def test_original_bytes_hash_and_cross_output_duplicate_ids_retained(self):
        self.collect()
        rows=self.envelopes()
        same=[row for row in rows if json.loads(row['raw_utf8'])['episode_id']=='same-id']
        self.assertEqual(len(same),2);self.assertNotEqual(same[0]['output_id'],same[1]['output_id'])
        row=next(row for row in same if 'coco_standard' in row['path'])
        self.assertEqual(row['raw_utf8'].encode(),self.raw)
        self.assertEqual(row['sha256'],hashlib.sha256(self.raw).hexdigest())
        mapping=json.loads((self.out/'output_task_mapping.json').read_text())
        self.assertEqual(len(next(item for item in mapping if item['output']==str(self.ok))['tasks']),2)

    def test_source_manifest_exact_copy_and_run_unchanged(self):
        before={str(p.relative_to(self.run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}
        archive=self.collect()
        after={str(p.relative_to(self.run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        with tarfile.open(archive) as tar:
            names=tar.getnames()
            self.assertIn('run/source/worker.py',names)
            self.assertNotIn('run/data/external.json',names)
            self.assertFalse(any(name.endswith('.png') for name in names))
        meta=json.loads((self.out/'metadata_inventory.json').read_text())
        self.assertTrue(next(row for row in meta if row.get('expected_sha256'))['snapshot_hash_matches'])

    def test_symlink_episode_never_followed(self):
        self.put(self.home/'secret.json',b'EXTERNAL_SECRET')
        (self.ok/'episodes/symlink.json').symlink_to(self.home/'secret.json')
        self.collect()
        self.assertNotIn('EXTERNAL_SECRET',(self.out/'episodes.jsonl').read_text())
        omissions=json.loads((self.out/'omissions.json').read_text())
        self.assertTrue(any('symlink.json' in row['path'] for row in omissions))

    def test_time_budget_reports_unscanned_not_empty_success(self):
        self.collect(seconds=0)
        report=json.loads((self.out/'EXPORT_REPORT.json').read_text())
        self.assertFalse(report['all_plan_episode_directories_scanned'])
        self.assertEqual(report['episode_files'],0)
        self.assertEqual(report['directory_status_counts'],{'UNSCANNED_TIME_OR_ENTRY_BUDGET':3})
        self.assertTrue(report['state_after_captured'])

    def test_size_budget_preserves_omissions(self):
        self.collect(raw_budget=len(self.raw))
        self.assertEqual(len(self.envelopes()),1)
        report=json.loads((self.out/'EXPORT_REPORT.json').read_text())
        self.assertFalse(report['all_plan_episode_directories_scanned'])
        self.assertEqual(len(json.loads((self.out/'omissions.json').read_text())),2)

    def test_invalid_json_and_non_utf8_retained(self):
        self.put(self.ok/'episodes/invalid.json',b'{broken')
        self.put(self.ok/'episodes/nonutf8.json',b'\xff\x00')
        self.collect()
        rows=self.envelopes()
        self.assertEqual(len(rows),5)
        self.assertEqual(sum('parse_error' in row for row in rows),2)
        self.assertTrue(any('raw_base64' in row for row in rows))

    def test_absent_directory_is_explicit(self):
        task=dict(self.plan['tasks'][0],id='missing',output=str(self.run/'results/model/missing'))
        self.plan['tasks'].append(task);M.dump(self.run/'plan.json',self.plan)
        self.collect()
        coverage=json.loads((self.out/'directory_coverage.json').read_text())
        self.assertEqual(next(row for row in coverage if 'missing' in row['path'])['status'],'DIRECTORY_ABSENT')

    def test_collision_fingerprint_and_tag_guards(self):
        self.out.mkdir();self.put(self.out/'keep',b'KEEP')
        with self.assertRaises(ValueError):self.collect()
        self.assertEqual((self.out/'keep').read_bytes(),b'KEEP')
        for value in ('../x','', 'a/b','x'*65,'a;rm'):
            with self.assertRaises(ValueError):M.safe_tag(value)
        self.assertEqual(M.safe_tag('20261113-123'),'20261113-123')
        with self.assertRaises(ValueError):M.collect(self.run,self.home/'other',self.home,with_scheduler=False,expected_fingerprint='wrong')

    def test_completed_accounting_prioritized_over_stale_state(self):
        groups=M.output_groups(self.plan,self.state,{'12':'COMPLETED'})
        self.assertEqual(groups[0]['output'],str(self.failed))
        self.assertEqual(groups[0]['tasks'][0]['accounting_state'],'COMPLETED')

    def test_unsafe_snapshot_path_never_copied(self):
        M.dump(self.run/'source/snapshot.json',{'files':{'../../outside':'bad'}})
        self.collect()
        omissions=json.loads((self.out/'omissions.json').read_text())
        self.assertTrue(any(row['reason']=='INVALID_SOURCE_MANIFEST' for row in omissions))


    def test_cost_metadata_and_repair_snapshot_exact_files(self):
        M.dump(self.ok/'shards/attempt-1.json',{'wall_seconds':10})
        M.dump(self.run/'jobs/formal0/gpu_memory.json',{'peak':123})
        M.dump(self.run/'jobs/formal0/status.json',{'status':'PASS'})
        M.dump(self.run/'repair/archives/formal2/job-12/job/failure.json',{'error':'CUDA unavailable'})
        self.state['tasks']['formal2']['attempt_history']=[{'job_id':'12','status':'FAILED'}]
        M.dump(self.run/'rolling_state.json',self.state)
        for name in ('policy.json','prepare.json','applied.json','archive_inventory.json','transaction.json'):
            M.dump(self.run/'repair'/name,{'name':name})
        self.put(self.run/'repair/source/worker.py',b'print("controller")\n')
        M.dump(self.run/'repair/source/snapshot.json',{'files':{'worker.py':hashlib.sha256(b'print("controller")\n').hexdigest()}})
        self.collect()
        self.assertTrue((self.out/'run/results/model/coco_standard/shards/attempt-1.json').exists())
        self.assertTrue((self.out/'run/jobs/formal0/gpu_memory.json').exists())
        self.assertTrue((self.out/'run/repair/archives/formal2/job-12/job/failure.json').exists())
        self.assertTrue((self.out/'run/repair/source/worker.py').exists())
        self.assertTrue((self.out/'run/repair/archive_inventory.json').exists())

    def test_external_path_metadata_does_not_copy_content(self):
        self.put(self.home/'external-weight',b'PRIVATE_WEIGHT_BYTES')
        self.plan['resources']={'insid3':{'python':str(self.home/'external-weight')}}
        self.plan['manifests']={'coco':{'path':str(self.home/'external-weight')}}
        M.dump(self.run/'plan.json',self.plan)
        self.collect()
        rows=json.loads((self.out/'external_paths.json').read_text())
        row=next(row for row in rows if row['path']==str(self.home/'external-weight'))
        self.assertEqual(row['size'],20)
        self.assertFalse(row['content_copied'])
        self.assertTrue(all(b'PRIVATE_WEIGHT_BYTES' not in p.read_bytes() for p in self.out.rglob('*') if p.is_file()))

if __name__=='__main__':unittest.main(verbosity=2)
