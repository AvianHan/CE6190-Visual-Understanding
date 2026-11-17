#!/usr/bin/env python3
"""Read-only B2 evidence audit. Original evidence bytes are never rewritten."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, default=Path('b2_evidence'))
    ap.add_argument('--out', type=Path, default=Path('b2_analysis/provenance_audit.json'))
    args = ap.parse_args()
    root = args.evidence.resolve()
    original = Path('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime')
    run = root / 'runtime/runs/ics-batch2-20261116'
    core = run / 'scopes/core'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / 'source'))
    from ics.runner import canonical_sha256, scientific_config, project_code_digest
    from ics.batch2_analysis import verify_counts, _planned_config
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
    configs={'batch2_insid3_full','batch2_cgics_full','batch2_cgics_single','batch2_cgics_bon1'}
    def verify_plan():
        raw={k:v for k,v in plan.items() if k!='fingerprint'}
        actual=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(actual==plan['fingerprint']==state['plan_fingerprint'],'Plan/state fingerprint mismatch')
        require(set(report['configurations'])==configs and report['queue_pass_tasks']==224,'Export configuration/task count mismatch')
    checked('plan and report configuration',verify_plan)
    grouped=defaultdict(list)
    for t in plan['tasks']:
        if t['config_data']['experiment_id'] in configs:
            grouped[t['output']].append(t)
    code_digest=project_code_digest()
    excluded={'sha256:'+x for x in prior_sha}|prior_ids
    global_queries=set()
    cohort_records=[]
    audited_manifests=set()
    config_manifest_sets=defaultdict(set)
    per_config_counts=Counter()
    provenance_by_config={}
    expected_layout={x['manifest_id']:x for x in receipt['request']['cohort_layout']}
    def output(out, tasks):
        path=mapped(out)
        meta=read(path/'run.json'); binding=meta['binding']
        cfg=scientific_config(meta['config'],validate_model_path=False)
        require(canonical_sha256(cfg)==binding['config_sha256'], 'Scientific config hash')
        identity={k:v for k,v in binding.items() if k!='run_id'}
        require(binding['run_id']==cfg['experiment_id']+'-'+canonical_sha256(identity)[:16], 'Run ID hash')
        require(binding['code_sha256']==code_digest, 'Frozen scientific code hash')
        require(binding['dataset_root']==str(original/'data'),'Unexpected dataset root')
        require(cfg['experiment_id'] in configs,'Unexpected experiment')
        prov=binding['configured_provenance']
        if cfg['experiment_id'] in provenance_by_config:
            require(prov==provenance_by_config[cfg['experiment_id']],
                    'Recorded upstream/model provenance varies within configuration')
        else:
            provenance_by_config[cfg['experiment_id']]=prov
        require(meta['config']['method']==tasks[0]['config_data']['method'],'Method identity')
        require(len(tasks)==4 and {t['shard_index'] for t in tasks}==set(range(4)), 'Four unique shards required')
        for t in tasks:
            t=dict(t); t['_resource_profile']=plan['resources'][t['resource']]
            require(_planned_config(t)==cfg, 'Plan/output scientific config differs')
            require(t['manifest']==meta['manifest_path'], 'Plan/output manifest path differs')
            require(state['tasks'][t['id']]['status']=='PASS', 'Task not PASS in saved state')
            for filename,taskkey in [('rolling_markers/'+t['id']+'.json','key'),('jobs/'+t['id']+'/complete.json','task_id')]:
                marker=read(core/filename)
                require(marker[taskkey]==t['id'] and marker['plan_fingerprint']==plan['fingerprint'] and marker['status']=='PASS','Invalid task completion binding')
                require(str(marker['slurm_job_id'])==str(state['tasks'][t['id']]['job_id']),'Job ID differs from completion record')
            counts['passed_shards']+=1
        entry=receipt['manifests'][tasks[0]['manifest_id']]
        layout=expected_layout[tasks[0]['manifest_id']]
        require(all(entry[k]==layout[k] for k in ('dataset','fold','cohort')),
                'Receipt differs from predeclared cohort layout')
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
            require((ep['dataset'],ep['fold'])==(entry['dataset'],entry['fold']),
                    'Manifest episode dataset/fold differs from receipt')
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
        if tasks[0]['manifest_id'] not in audited_manifests:
            require(not(current & global_queries),'Cross-fold/dataset query overlap')
            global_queries.update(current)
        require(len(local)==entry['unique_query_images']==entry['requested_queries']==100,'Unique query count')
        require(len(local)==layout['queries'] and len(ids)==entry['episodes'],
                'Predeclared query or episode coverage mismatch')
        seen=set()
        for p in sorted((path/'episodes').glob('*.json')):
            row=read(p);eid=row['episode_id'];ep=episodes[eid]
            require(eid not in seen,'Duplicate row');seen.add(eid)
            require(row['status']=='ok','Non-ok episode')
            require(row['config_sha256']==binding['config_sha256'],'Row config hash')
            require(row['manifest_sha256']==binding['manifest_sha256'],'Row manifest hash')
            require(row['run_id']==binding['run_id'],'Row run identity')
            require(row['experiment_id']==cfg['experiment_id'] and row['method']==cfg['method'] and row['variant']==cfg['variant'],'Row scientific method identity')
            require(row['protocol_id']==binding['protocol_id'] and row['evaluation_protocol']==binding['evaluation_protocol']=='unified_original_resolution','Row protocol identity')
            require(row['nshot']==len(ep['references'])==1 and row['reference_seed']==ep['reference_seed'],'One-shot reference protocol')
            require(row['dataset_version']==ep['dataset_version'] and row['mask_semantics']==ep['mask_semantics'],'Dataset/mask semantics mismatch')
            expected={k:ep[k] for k in ('dataset','split','fold','track','concept_id')}
            expected.update(query_image_sha256=ep['query']['image_sha256'],query_mask_sha256=ep['query']['mask_sha256'],reference_image_sha256=[r['image_sha256'] for r in ep['references']])
            require(all(row[k]==v for k,v in expected.items()),'Episode manifest identity')
            require(row['num_shards']==4 and row['shard_index']==positions[eid]%4,'Modulo shard assignment')
            require(row['base_seed']==cfg['seed'] and row['episode_seed']==int(canonical_sha256([cfg['seed'],ep['query']['image_sha256']])[:8],16),'Frozen episode seed')
            verify_counts(row)
            require(not row.get('alternative_predictions'),'Unexpected nested alternatives in B2')
            counts['episodes']+=1
            per_config_counts[cfg['experiment_id']]+=1
        require(seen==set(ids),'Missing episode rows')
        config_manifest_sets[cfg['experiment_id']].add(tasks[0]['manifest_id'])
        if tasks[0]['manifest_id'] not in audited_manifests:
            cohort_records.append({'manifest':tasks[0]['manifest_id'],'episodes':len(ids),'unique_queries':len(local),'represented_classes':len({e['concept_id'] for e in m['episodes']}),'class_universe_count':entry['class_universe_count']})
            audited_manifests.add(tasks[0]['manifest_id'])
        counts['outputs']+=1
    for out,tasks in sorted(grouped.items()):
        checked(out,lambda out=out,tasks=tasks:output(out,tasks))
    def total_coverage():
        require(counts['passed_shards']==224 and counts['outputs']==56 and counts['episodes']==16800,'Unexpected B2 coverage')
        require(set(config_manifest_sets)==configs and all(v==set(receipt['manifests']) for v in config_manifest_sets.values()),'Configurations do not share all14 exact manifests')
        require(all(per_config_counts[c]==4200 for c in configs),'Unexpected per-configuration coverage')
        require(len(audited_manifests)==14 and len(global_queries)==2800,'Unexpected unique query/source identity coverage')
        require(set(receipt['manifests'])==set(expected_layout),'Receipt manifest layout differs')
        require(receipt['episodes']==4200 and receipt['query_images']==1400,
                'Receipt total episode/query counts differ')
        cg_prov=[provenance_by_config[c] for c in sorted(configs) if c.startswith('batch2_cgics_')]
        require(all(p==cg_prov[0] for p in cg_prov),
                'Recorded CG-ICS upstream/model provenance differs across variants')
    checked('total coverage',total_coverage)
    result={'status':'PASS' if not errors else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'counts':dict(counts),'error_count':len(errors),'errors':errors,'code_sha256':code_digest,'prior_unique_image_hashes':len(prior_sha),'prior_unique_source_ids':len(prior_ids),'cohorts':cohort_records,'per_config_episodes':dict(per_config_counts),'configurations':sorted(config_manifest_sets),'recorded_provenance_sha256_by_config':{c:canonical_sha256(p) for c,p in provenance_by_config.items()},'unique_query_images':len(global_queries)//2,'export_created_at_utc':report['created_at_utc'],'boundaries':['The audit verifies recorded identities, hash-bound source code, manifests, history and paired configuration coverage.','Raw query/reference RGB images, reference masks, model weights and original dataset metadata are absent; their bytes and upstream GT conversion cannot be independently reverified.','The four configurations share the exact same 14 manifest bytes and 4200 query-reference episodes; no MLLM, SAM or INSID3 inference is rerun.','Recorded upstream/model hashes are consistent across all folds per configuration and across the three CG-ICS variants; this is not a new hash of absent model files.','PNG pixel and query-GT checks are separately recorded in pixel_audit.json.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cohorts'},ensure_ascii=False,indent=2))
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
