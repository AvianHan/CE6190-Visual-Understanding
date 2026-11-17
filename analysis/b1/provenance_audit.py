#!/usr/bin/env python3
"""Read-only B1 evidence audit. Original evidence bytes are never rewritten."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, default=Path('b1_evidence'))
    ap.add_argument('--out', type=Path, default=Path('b1_analysis/provenance_audit.json'))
    args = ap.parse_args()
    root = args.evidence.resolve()
    original = Path('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime')
    run = root / 'runtime/runs/ics-batch2-20261116'
    core = run / 'scopes/core'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / 'source'))
    from ics.runner import canonical_sha256, scientific_config, project_code_digest
    from ics.batch2_analysis import verify_score_pool, verify_counts, _planned_config
    from ics.batch2_data import _identities
    from ics_eval import manifest_fingerprint, validate_manifest
    errors = []
    counts = Counter()
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def read(p):
        return json.loads(p.read_text())
    def mapped(p):
        result = root / 'runtime' / Path(p).relative_to(original)
        require(result.resolve().is_relative_to(root), 'Escaped evidence path')
        require(not result.is_symlink(), 'Symlink in evidence')
        return result
    def sha(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576), b''):
                h.update(b)
        return h.hexdigest()
    def checkfile(p, digest):
        require(sha(p) == digest, f'File SHA mismatch: {p}')
    def checked(label, fn):
        try:
            fn()
        except Exception as exc:
            errors.append({'context':label, 'error':str(exc)})
    report = read(root / 'EXPORT_REPORT.json')
    plan, state = read(core/'plan.json'), read(core/'rolling_state.json')
    receipt = read(core/'data/data_receipt.json')
    history = receipt['history']
    def inventory():
        names = set()
        for record in report['files']:
            p = root / record['archive']
            require(p.resolve().is_relative_to(root) and not p.is_symlink(), 'Unsafe inventory path')
            require(record['archive'] not in names, 'Duplicate inventory entry')
            names.add(record['archive'])
            require(p.stat().st_size == record['bytes'], f'Size mismatch: {p}')
            checkfile(p, record['sha256'])
            counts['inventory_files'] += 1
            counts['inventory_bytes'] += record['bytes']
        require(counts['inventory_bytes'] == report['uncompressed_bytes'], 'Inventory byte total differs')
    checked('export inventory', inventory)
    prior_sha, prior_ids = set(), set()
    def prior():
        require(canonical_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'}) == receipt['receipt_sha256'], 'Receipt hash')
        require(canonical_sha256(receipt['request']) == receipt['request_sha256'], 'Preparation request hash')
        require(canonical_sha256({k:v for k,v in history.items() if k!='exclusion_sha256'}) == history['exclusion_sha256'], 'Exclusion receipt hash')
        task_manifests = set()
        for rec in history['plans']:
            checkfile(mapped(rec['path']), rec['file_sha256'])
            old_plan = read(mapped(rec['path']))
            require(old_plan['fingerprint']==rec['plan_fingerprint'], 'Prior plan fingerprint')
            require(len(old_plan['tasks'])==rec['formal_tasks'], 'Prior plan formal count')
            task_manifests.update(t['manifest'] for t in old_plan['tasks'])
            counts['prior_plans'] += 1
        require(task_manifests=={r['path'] for r in history['manifests']}, 'Prior manifest coverage')
        for rec in history['manifests']:
            checkfile(mapped(rec['path']), rec['file_sha256'])
            m = read(mapped(rec['path']))
            validate_manifest(m)
            require(manifest_fingerprint(m)==rec['manifest_sha256'], 'Prior canonical manifest hash')
            ids = {e['episode_id'] for e in m['episodes']}
            require(bool(rec['bindings']), 'No retained prior binding')
            for b in rec['bindings']:
                checkfile(mapped(b['path']), b['file_sha256'])
                binding=read(mapped(b['path']))['binding']
                require(binding['manifest_sha256']==rec['manifest_sha256'] and binding['protocol_id']==m['protocol_id'], 'Prior manifest/run binding')
                selected=binding['selected_episode_ids']
                require(len(selected)==len(set(selected))==b['selected_episode_count'] and set(selected)<=ids, 'Prior selected IDs')
                counts['prior_bindings'] += 1
            for ep in m['episodes']:
                qids=_identities(ep['query'],ep['dataset'])
                prior_sha.update(x[7:] for x in qids if x.startswith('sha256:'))
                prior_ids.update(x for x in qids if x.startswith('coco-image:'))
            counts['prior_manifests'] += 1
        require(sorted(prior_sha)==history['query_image_sha256'], 'Reconstructed prior image hashes')
        require(sorted(prior_ids)==history['canonical_source_image_ids'], 'Reconstructed prior source IDs')
    checked('history and receipt', prior)
    grouped=defaultdict(list)
    for t in plan['tasks']:
        if t['config_data']['experiment_id']=='batch2_cgics_score_pool':
            grouped[t['output']].append(t)
    code_digest=project_code_digest()
    excluded={'sha256:'+x for x in prior_sha}|prior_ids
    global_queries=set()
    cohort_records=[]
    def output(out, tasks):
        path=mapped(out)
        meta=read(path/'run.json'); binding=meta['binding']
        cfg=scientific_config(meta['config'],validate_model_path=False)
        require(canonical_sha256(cfg)==binding['config_sha256'], 'Scientific config hash')
        identity={k:v for k,v in binding.items() if k!='run_id'}
        require(binding['run_id']==cfg['experiment_id']+'-'+canonical_sha256(identity)[:16], 'Run ID hash')
        require(binding['code_sha256']==code_digest, 'Frozen scientific code hash')
        require(len(tasks)==4 and {t['shard_index'] for t in tasks}==set(range(4)), 'Four unique shards required')
        for t in tasks:
            t=dict(t); t['_resource_profile']=plan['resources'][t['resource']]
            require(_planned_config(t)==cfg, 'Plan/output scientific config differs')
            require(t['manifest']==meta['manifest_path'], 'Plan/output manifest path differs')
            require(state['tasks'][t['id']]['status']=='PASS', 'Task not PASS in saved state')
            for filename,taskkey in [('rolling_markers/'+t['id']+'.json','key'),('jobs/'+t['id']+'/complete.json','task_id')]:
                marker=read(core/filename)
                require(marker[taskkey]==t['id'] and marker['plan_fingerprint']==plan['fingerprint'] and marker['status']=='PASS','Invalid task completion binding')
            counts['passed_shards']+=1
        entry=receipt['manifests'][tasks[0]['manifest_id']]
        checkfile(mapped(entry['path']),entry['file_sha256'])
        checkfile(mapped(entry['protocol_path']),entry['protocol_sha256'])
        m=read(mapped(entry['path'])); side=read(mapped(entry['protocol_path']))
        validate_manifest(m)
        require(manifest_fingerprint(m)==binding['manifest_sha256']==entry['manifest_sha256']==side['manifest_sha256'],'Manifest fingerprint binding')
        require(m['protocol_id']==binding['protocol_id']==side['protocol_id'],'Protocol binding')
        require(side['batch2']['prior_exclusion_sha256']==history['exclusion_sha256'],'Exclusion sidecar binding')
        require(mapped(side['batch2']['exclusion_receipt_path'])==core/'data/data_receipt.json','Receipt path binding')
        require(side['batch2']['cohort']==entry['cohort']=='fresh','Cohort binding')
        ids=[e['episode_id'] for e in m['episodes']]
        require(len(ids)==len(set(ids))==300 and ids==binding['selected_episode_ids'],'Complete selected manifest IDs')
        episodes={e['episode_id']:e for e in m['episodes']}; positions={eid:i for i,eid in enumerate(ids)}
        local={}; refs=defaultdict(set)
        for ep in m['episodes']:
            qids=_identities(ep['query'],ep['dataset'])
            require(not (qids & excluded),'Prior query overlaps fresh cohort')
            qsha=ep['query']['image_sha256']; val=(ep['concept_id'],tuple(sorted(qids)))
            require(qsha not in local or local[qsha]==val,'Multiple query targets on same image')
            local[qsha]=val
            for ref in ep['references']:
                rids=_identities(ref,ep['dataset'])
                require(not (rids&qids) and not (rids&refs[qsha]),'Reference/query overlap or repeated reference')
                refs[qsha].update(rids)
        current=set().union(*(set(v[1]) for v in local.values()))
        require(not(current & global_queries),'Cross-fold/dataset query overlap')
        global_queries.update(current)
        require(len(local)==entry['unique_query_images']==entry['requested_queries']==100,'Unique query count')
        seen=set()
        for p in sorted((path/'episodes').glob('*.json')):
            row=read(p);eid=row['episode_id'];ep=episodes[eid]
            require(eid not in seen,'Duplicate row');seen.add(eid)
            require(row['status']=='ok','Non-ok episode')
            require(row['config_sha256']==binding['config_sha256'],'Row config hash')
            expected={k:ep[k] for k in ('dataset','split','fold','track','concept_id')}
            expected.update(query_image_sha256=ep['query']['image_sha256'],query_mask_sha256=ep['query']['mask_sha256'],reference_image_sha256=[r['image_sha256'] for r in ep['references']])
            require(all(row[k]==v for k,v in expected.items()),'Episode manifest identity')
            require(row['num_shards']==4 and row['shard_index']==positions[eid]%4,'Modulo shard assignment')
            require(row['base_seed']==cfg['seed'] and row['episode_seed']==int(canonical_sha256([cfg['seed'],ep['query']['image_sha256']])[:8],16),'Frozen episode seed')
            verify_counts(row)
            require(set(row['alternative_predictions'])=={'ref_only','query_only'},'Alternative coverage')
            for alternative in row['alternative_predictions'].values():verify_counts(alternative)
            audit=verify_score_pool(row)
            prompts=[c['prompt'] for c in audit['candidates']]
            require(all(p.strip() for p in prompts) and len(prompts)==len(set(prompts))==audit['unique_candidates'], 'Duplicate/empty candidate or pool size mismatch')
            require(audit['fingerprint_payload']['source_pin']==binding['configured_provenance']['adapter.upstream_dir']['commit'], 'Pool upstream source pin mismatch')
            require(audit['requested_candidates']==cfg['adapter']['BON'] and audit['generated_candidates']>=len(prompts), 'Candidate budget mismatch')
            require(audit['tie_policy']=='first_candidate_in_generated_order', 'Tie metadata mismatch')
            counts['episodes_and_pools']+=1
        require(seen==set(ids),'Missing episode rows')
        cohort_records.append({'manifest':tasks[0]['manifest_id'],'episodes':len(ids),'unique_queries':len(local),'represented_classes':len({e['concept_id'] for e in m['episodes']}),'class_universe_count':entry['class_universe_count']})
        counts['outputs']+=1
    for out,tasks in sorted(grouped.items()):
        checked(out,lambda out=out,tasks=tasks:output(out,tasks))
    checked('total coverage',lambda:require(counts['passed_shards']==56 and counts['outputs']==14 and counts['episodes_and_pools']==4200,'Unexpected B1 coverage'))
    result={'status':'PASS' if not errors else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'counts':dict(counts),'error_count':len(errors),'errors':errors,'code_sha256':code_digest,'prior_unique_image_hashes':len(prior_sha),'prior_unique_source_ids':len(prior_ids),'cohorts':cohort_records,'boundaries':['The audit verifies recorded identities, hash-bound source code, manifests, history and ranking arithmetic.','Raw query/reference RGB images, reference masks, model weights and original dataset metadata are absent; their bytes and upstream GT conversion cannot be independently reverified.','Pool candidates and component scores are checked for recorded consistency; no MLLM or SAM inference is rerun.','PNG pixel and query-GT checks are separately recorded in pixel_audit.json.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cohorts'},ensure_ascii=False,indent=2))
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
