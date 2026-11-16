"""Evidence-based and synthetic filesystem checks; no GPU or Slurm mutations."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('resume_ics_tail', ROOT/'scripts/resume_ics_tail.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FIXTURE = ROOT/'tests/fixtures/tail_recovery_evidence.json.gz'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.encoded(value))


def tree(root):
    return {str(p.relative_to(root)): M.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class TailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)/'run'
        self.run.mkdir()
        with gzip.open(FIXTURE, 'rt') as handle:
            evidence = json.load(handle)
        self.state = evidence['state']
        self.plan = evidence['plan']
        old = self.plan['run_dir']
        def relocate(v):
            if isinstance(v, str): return v.replace(old, str(self.run))
            if isinstance(v, dict): return {k:relocate(x) for k,x in v.items()}
            if isinstance(v, list): return [relocate(x) for x in v]
            return v
        self.state = relocate(self.state)
        self.plan = relocate(self.plan)
        self.history = {}
        for key, task in self.state['tasks'].items():
            if task['status'] in ('PASS','FAILED'):
                self.history[str(task['job_id'])] = {
                    'job_id':str(task['job_id']), 'job_name':task['job_name'],
                    'state':'COMPLETED' if task['status']=='PASS' else 'FAILED',
                    'exit_code':'0:0' if task['status']=='PASS' else '1:0',
                    'qos':self.state['policy']['qos'], 'node':'fixture'}
        for key in M.SELECTED:
            task=self.state['tasks'][key]
            original=evidence['intents'][key]
            intent=relocate(original)
            path=Path(task['repair_attempt']['intent_path'])
            put(path,intent)
            task['repair_attempt']['intent_sha256']=M.sha(path.read_bytes())
            if key==M.INTERRUPTED:
                put(path.with_suffix('.claim.json'), {'slurm_job_id':'201500','intent_sha256':task['repair_attempt']['intent_sha256']})
                put(self.run/'rolling_markers'/(key+'.json'), {'key':key,'status':'FAIL','plan_fingerprint':M.PLAN,'slurm_job_id':'201500','error_type':'KeyboardInterrupt','error':'Termination signal 15'})
        science={}
        defs={t['id']:t for t in self.plan['tasks']+self.plan['smoke']}
        for key,(jid,name,_) in M.SCIENTIFIC_FAILURES.items():
            path=Path(defs[key]['output'])/'episodes'/(name+'.json')
            put(path,{'status':'error','error':'Reference mask is empty at the CG-ICS working resolution'})
            science[key]=(jid,name,M.sha(path.read_bytes()))
        self.science_patch=mock.patch.object(M,'SCIENTIFIC_FAILURES',science)
        self.science_patch.start();self.addCleanup(self.science_patch.stop)
        self.original_marker=M.marker_checks
        def checked_marker(run, task, status='PASS'):
            if status=='FAIL':return self.original_marker(run,task,status)
            return {'status':'PASS'}
        self.marker_patch=mock.patch.object(M,'marker_checks',side_effect=checked_marker)
        self.markers=self.marker_patch.start();self.addCleanup(self.marker_patch.stop)

    def audit(self):
        return M.audit_state(self.plan,self.state,{},self.history,self.run)

    def test_audit_selects_only_ten_and_preserves_scientific_failures(self):
        before=tree(self.run); state_before=copy.deepcopy(self.state)
        proof=self.audit()
        self.assertEqual(set(proof['selected']),set(M.SELECTED))
        self.assertEqual(len(proof['pending_gate_tasks']),14)
        self.assertEqual(len(proof['preserved_scientific_failures']),4)
        self.assertEqual(self.state,state_before)
        self.assertEqual(tree(self.run),before)

    def test_fresh_queue_new_allocation_or_claim_refuses_without_mutation(self):
        for scenario in ('queue','history','claim','marker','complete','next_intent'):
            with self.subTest(scenario=scenario):
                h=copy.deepcopy(self.history);q={}
                key=M.UNCERTAIN[0];claim=Path(self.state['tasks'][key]['repair_attempt']['intent_path']).with_suffix('.claim.json')
                if scenario=='queue':q={'123':{'job_name':'anything'}}
                if scenario=='history':h['999999']={'job_name':self.state['tasks'][key]['job_name']}
                created=None
                if scenario=='claim':created=claim
                if scenario=='marker':created=self.run/'rolling_markers'/(key+'.json')
                if scenario=='complete':created=self.run/'jobs'/key/'complete.json'
                if scenario=='next_intent':created=claim.with_name('attempt-'+str(self.state['tasks'][key]['attempts']+1)+'.json')
                if created is not None:put(created,{'slurm_job_id':'unknown'})
                before=tree(self.run)
                with self.assertRaises(ValueError):M.audit_state(self.plan,self.state,q,h,self.run)
                self.assertEqual(tree(self.run),before)
                if created is not None:created.unlink()

    def test_alias_real_success_cannot_be_reassigned(self):
        self.state['tasks'][M.ALIASES['201529'][1]]['job_id']='888888'
        with self.assertRaises(ValueError):self.audit()

    def test_changed_intent_refuses(self):
        path=Path(self.state['tasks'][M.UNCERTAIN[0]]['repair_attempt']['intent_path'])
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Intent SHA'):self.audit()

    def test_changed_scientific_error_refuses_without_mutation(self):
        key=next(iter(M.SCIENTIFIC_FAILURES))
        name=M.SCIENTIFIC_FAILURES[key][1]
        definition=next(t for t in self.plan['tasks'] if t['id']==key)
        path=Path(definition['output'])/'episodes'/(name+'.json')
        put(path,{'status':'ok'})
        before=tree(self.run)
        with self.assertRaisesRegex(ValueError,'Scientific failure record changed'):self.audit()
        self.assertEqual(tree(self.run),before)

    def test_build_after_changes_only_selection_and_never_decrements_attempts(self):
        original=copy.deepcopy(self.state)
        after=M.build_after_state(self.state,'txn-fixture')
        for key,task in original['tasks'].items():
            if key not in M.SELECTED:self.assertEqual(after['tasks'][key],task)
            else:
                changed=after['tasks'][key]
                self.assertEqual(changed['status'],'PENDING')
                self.assertEqual(changed['attempts'],task['attempts'])
                self.assertNotIn('job_id',changed)
                self.assertNotIn('repair_attempt',changed)
                self.assertEqual(changed['attempt_history'][-1]['job_name'],task['job_name'])
        self.assertEqual(self.state,original)

    def episode_fixture(self):
        defs=[]
        for key in M.SELECTED:
            n=250 if key==M.INTERRUPTED else 2
            manifest=self.run/'manifests'/(key+'.json')
            eps=[{'episode_id':key+'-ep-'+str(i)} for i in range(n)]
            put(manifest,{'episodes':eps})
            output=self.run/'results'/key
            defs.append({'id':key,'output':str(output),'manifest':str(manifest),'max_episodes':None,'num_shards':1,'shard_index':0,'config_data':{'experiment_id':'fixture'}})
            if key==M.INTERRUPTED:
                put(output/'run.json',{'binding':{'run_id':'fixture-run'}})
                for ep in eps[:155]:
                    token=M.sha(ep['episode_id'].encode())
                    pred=output/'predictions'/(token+'.png');pred.parent.mkdir(parents=True,exist_ok=True);pred.write_bytes(b'stable-fixture-prediction')
                    put(output/'episodes'/(token+'.json'),{'episode_id':ep['episode_id'],'status':'ok','run_id':'fixture-run','experiment_id':'fixture','prediction_path':'predictions/'+token+'.png','prediction_sha256':M.sha(pred.read_bytes())})
        return {'tasks':defs[:-1],'smoke':defs[-1:]}

    def test_episode_inspection_keeps_155_successes_and_finds_95_missing(self):
        plan=self.episode_fixture();before=tree(self.run)
        rows=M.inspect_selected_episodes(plan,self.state,self.run)
        self.assertEqual(rows[0]['counts'],{'ok':155,'missing':95})
        self.assertEqual(sum(len(r['preserved_successes']) for r in rows),155)
        self.assertEqual(tree(self.run),before)

    def test_existing_error_or_changed_prediction_refuses(self):
        plan=self.episode_fixture()
        out=Path(plan['tasks'][0]['output']);pred=next((out/'predictions').glob('*.png'))
        pred.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'prediction changed'):M.inspect_selected_episodes(plan,self.state,self.run)
        pred.write_bytes(b'stable-fixture-prediction')
        task=plan['tasks'][1];ep=json.loads(Path(task['manifest']).read_text())['episodes'][0]
        path=Path(task['output'])/'episodes'/(M.sha(ep['episode_id'].encode())+'.json')
        put(path,{'status':'error','episode_id':ep['episode_id']})
        with self.assertRaisesRegex(ValueError,'Existing error'):M.inspect_selected_episodes(plan,self.state,self.run)

    def transaction_fixture(self, publish=True):
        put(self.run/'rolling_state.json',self.state)
        for key in M.SELECTED:
            p=self.run/'rolling_slurm'/(key+'.sh');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
        put(self.run/'jobs'/M.INTERRUPTED/'failure.json',{'old':'failure'})
        before_raw=(self.run/'rolling_state.json').read_bytes()
        before_sha=M.sha(before_raw)
        transaction_id=M.sha((M.PLAN+before_sha+M.TAG).encode())
        after=M.build_after_state(self.state,transaction_id)
        self.txn_episode_plan=self.episode_fixture()
        episodes=M.inspect_selected_episodes(self.txn_episode_plan,self.state,self.run)
        proof=self.audit()
        txn={'schema_version':1,'transaction_id':transaction_id,'selected':list(M.SELECTED),'plan_fingerprint':M.PLAN,'before_state':copy.deepcopy(self.state),'after_state':after,'before_sha256':before_sha,'after_sha256':M.sha(M.encoded(after)),'archives':M.archive_selection(self.run,self.state),'history':self.history,'episode_inspection':episodes,'proof':{k:v for k,v in proof.items() if k!='history'}}
        if publish:M.publish_transaction(self.run,txn,before_raw)
        return txn

    def test_atomic_publication_keeps_partial_staging_outside_transaction(self):
        txn=self.transaction_fixture(publish=False)
        before_raw=(self.run/'rolling_state.json').read_bytes()
        for phase in ('before_state','transaction'):
            with self.subTest(phase=phase):
                def fail(point,index):
                    if point==phase:raise RuntimeError('simulated preparation interruption')
                with self.assertRaises(RuntimeError):M.publish_transaction(self.run,txn,before_raw,fault=fail)
                self.assertFalse((self.run/M.TAG).exists())
                self.assertEqual((self.run/'rolling_state.json').read_bytes(),before_raw)
        partials=list(self.run.glob(M.TAG+'-building-*'))
        self.assertEqual(len(partials),2)
        M.publish_transaction(self.run,txn,before_raw)
        M.validate_transaction(self.run,txn)
        self.assertEqual(list(self.run.glob(M.TAG+'-building-*')),partials)
        M.apply_transaction(self.run,txn)

    def test_postprepare_intent_claim_and_scientific_evidence_changes_refuse(self):
        txn=self.transaction_fixture()
        uncertain_intent=Path(self.state['tasks'][M.UNCERTAIN[0]]['repair_attempt']['intent_path'])
        interrupted_claim=Path(self.state['tasks'][M.INTERRUPTED]['repair_attempt']['intent_path']).with_suffix('.claim.json')
        scientific_error=Path(txn['proof']['preserved_scientific_failures'][0]['path'])
        paths=(uncertain_intent,uncertain_intent.with_suffix('.claim.json'),interrupted_claim,scientific_error)
        for path in paths:
            with self.subTest(path=str(path)):
                original=path.read_bytes() if path.exists() else None
                path.write_bytes(original+b' ' if original is not None else b'{"slurm_job_id":"999999"}\n')
                before=tree(self.run)
                with self.assertRaises(ValueError):M.apply_transaction(self.run,txn)
                self.assertEqual(tree(self.run),before)
                if original is None:path.unlink()
                else:path.write_bytes(original)

    def test_coherently_rewritten_success_still_refuses_transaction(self):
        txn=self.transaction_fixture()
        saved=txn['episode_inspection'][0]['preserved_successes'][0]
        prediction=Path(saved['prediction_path']);episode=Path(saved['path'])
        prediction.write_bytes(b'changed-prediction-with-consistent-row-hash')
        row=json.loads(episode.read_text());row['prediction_sha256']=M.sha(prediction.read_bytes())
        put(episode,row)
        # Self-consistency alone cannot protect the previously captured result.
        inspected=M.inspect_selected_episodes(self.txn_episode_plan,self.state,self.run)
        self.assertEqual(inspected[0]['counts'],{'ok':155,'missing':95})
        before=tree(self.run)
        with self.assertRaises(ValueError):M.apply_transaction(self.run,txn)
        self.assertEqual(tree(self.run),before)

    def test_transaction_corruption_refuses_before_archive_or_state_mutation(self):
        txn=self.transaction_fixture();before=tree(self.run)
        for scenario in ('integrity','identity','before_digest','unselected_task','archive_path','frozen_field'):
            with self.subTest(scenario=scenario):
                changed=copy.deepcopy(txn)
                if scenario=='integrity':changed['transaction_id']='incorrect'
                if scenario=='identity':changed['transaction_id']='incorrect'
                if scenario=='before_digest':changed['before_sha256']='0'*64
                if scenario=='unselected_task':
                    key=next(iter(M.SCIENTIFIC_FAILURES))
                    changed['after_state']['tasks'][key]['status']='PENDING'
                if scenario=='archive_path':changed['archives'][0]['source']=str(self.run/'results')
                if scenario=='frozen_field':changed['after_state']['policy']['max_inflight']=999
                if scenario!='integrity':
                    changed['after_sha256']=M.sha(M.encoded(changed['after_state']))
                    changed['integrity_sha256']=M.fingerprint({k:v for k,v in changed.items() if k!='integrity_sha256'})
                with self.assertRaises(ValueError):M.apply_transaction(self.run,changed)
                self.assertEqual(tree(self.run),before)

    def test_interrupted_archive_rolls_forward_and_preserves_results(self):
        txn=self.transaction_fixture()
        result=self.run/'results'/'keep.json';put(result,{'ok':True});result_sha=M.sha(result.read_bytes())
        def fail(phase,index):
            if phase=='archive' and index==2:raise RuntimeError('simulated interruption')
        with self.assertRaises(RuntimeError):M.apply_transaction(self.run,txn,fault=fail)
        M.apply_transaction(self.run,txn)
        self.assertEqual(json.loads((self.run/'rolling_state.json').read_text()),txn['after_state'])
        self.assertEqual(M.sha(result.read_bytes()),result_sha)
        self.assertTrue((self.run/M.TAG/'applied.json').exists())
        for key in M.SCIENTIFIC_FAILURES:self.assertEqual(txn['before_state']['tasks'][key],txn['after_state']['tasks'][key])

    def test_interruption_after_state_swap_is_idempotent(self):
        txn=self.transaction_fixture()
        def fail(phase,index):
            if phase=='state':raise RuntimeError('simulated post-swap interruption')
        with self.assertRaises(RuntimeError):M.apply_transaction(self.run,txn,fault=fail)
        self.assertFalse((self.run/M.TAG/'applied.json').exists())
        M.apply_transaction(self.run,txn)
        self.assertEqual(json.loads((self.run/'rolling_state.json').read_text()),txn['after_state'])

    def test_applied_resume_does_not_reset_new_submission(self):
        txn=self.transaction_fixture();M.apply_transaction(self.run,txn)
        after=json.loads((self.run/'rolling_state.json').read_text())
        after['tasks'][M.INTERRUPTED].update(status='SUBMITTED',job_id='999999',attempts=3)
        put(self.run/'rolling_state.json',after)
        original=(self.run/'rolling_state.json').read_bytes()
        with mock.patch.object(M,'verify_static',return_value=self.plan),mock.patch.object(M,'controller') as control:
            M.resume(self.run)
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),original)
        control.assert_called_once_with(self.run,'activate')
        with self.assertRaisesRegex(ValueError,'State changed outside'):M.apply_transaction(self.run,txn)

    def test_failed_preflight_does_not_pause_or_modify_state(self):
        put(self.run/'rolling_state.json',self.state)
        before=(self.run/'rolling_state.json').read_bytes()
        with mock.patch.object(M,'verify_static',return_value=self.plan),mock.patch.object(M,'scheduler_evidence',return_value=({'live':{}},self.history)),mock.patch.object(M,'controller') as control:
            with self.assertRaises(ValueError):M.resume(self.run)
        control.assert_not_called()
        self.assertEqual((self.run/'rolling_state.json').read_bytes(),before)
        self.assertFalse((self.run/M.TAG/'transaction.json').exists())

    def test_source_pin_guard_and_source_digest(self):
        root=self.run/'pin-fixture';root.mkdir()
        plan={'run_dir':str(root),'project_root':str(root/'source')}
        fp=M.fingerprint(plan);plan['fingerprint']=fp;put(root/'plan.json',plan)
        for prefix in ('source','repair/source'):
            p=root/prefix/'module.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x=1\n')
            put(root/prefix/'snapshot.json',{'files':{'module.py':M.sha(p.read_bytes())}})
        put(root/'repair/policy.json',{'fingerprint':M.REPAIR,'controller_source':str(root/'repair/source')})
        put(root/'repair/applied.json',{})
        names=('source/snapshot.json','repair/source/snapshot.json','repair/policy.json')
        pins={n:M.sha((root/n).read_bytes()) for n in names}
        with mock.patch.object(M,'PLAN',fp),mock.patch.object(M,'PINS',pins):
            M.verify_static(root)
            (root/'source/module.py').write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):M.verify_static(root)
            (root/'source/module.py').write_text('x=1\n')
            (root/'repair/policy.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned evidence changed'):M.verify_static(root)


if __name__=='__main__':unittest.main()
