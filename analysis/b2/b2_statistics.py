#!/usr/bin/env python3
"""Read-only B2 matched-system analysis; no cluster or model calls.

Python 3.10+, NumPy. The separate provenance and pixel audits establish
source correctness. This script treats all four frozen configurations equally
and does not choose an accuracy- or speed-favourable subset.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS=('insid3_full','cgics_full','cgics_single','cgics_bon1')
PAIRS=(('cgics_full','insid3_full'),('cgics_full','cgics_single'),
       ('cgics_full','cgics_bon1'),('cgics_bon1','cgics_single'),
       ('cgics_bon1','insid3_full'),('cgics_single','insid3_full'))
PRIMARY_PAIRS=PAIRS[:3]
ORIGINAL=Path('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime')

def avg(v):
    v=list(v)
    return float(np.mean(v)) if v else None

def distribution(v):
    v=[float(x) for x in v if x is not None and np.isfinite(x)]
    return {'n':len(v),'mean':avg(v),'median':float(np.median(v)) if v else None,
            'p95':float(np.percentile(v,95)) if v else None,'sum':sum(v)}

def boot(values, clusters, reps, seed, strata=None):
    grouped=defaultdict(list)
    for v,c in zip(values,clusters):grouped[c].append(float(v))
    keys=sorted(grouped)
    sums=np.array([sum(grouped[k]) for k in keys]);counts=np.array([len(grouped[k]) for k in keys])
    result={'mean':float(sums.sum()/counts.sum()),'query_targets':len(values),'query_image_clusters':len(keys),'ci95':None}
    if len(keys)<2:return result
    rng=np.random.default_rng(seed);draws=[]
    if strata is not None:
        sk=defaultdict(set)
        for c,s in zip(clusters,strata):sk[s].add(c)
        pos={k:i for i,k in enumerate(keys)}
        groups=[[pos[k] for k in sorted(kk)] for _,kk in sorted(sk.items())]
    for start in range(0,reps,128):
        size=min(128,reps-start)
        if strata is None:
            idx=rng.integers(0,len(keys),size=(size,len(keys)))
            draws.extend((sums[idx].sum(1)/counts[idx].sum(1)).tolist())
        else:
            folds=[]
            for g in groups:
                idx=np.array(g)[rng.integers(0,len(g),size=(size,len(g)))]
                folds.append(sums[idx].sum(1)/counts[idx].sum(1))
            draws.extend(np.mean(folds,axis=0).tolist())
    result['ci95']=np.percentile(draws,[2.5,97.5]).tolist()
    return result

def target(row):
    return (row['dataset'],row['split'],row['fold'],row['track'],row['query_image_sha256'],row['query_mask_sha256'],row['concept_id'],row['base_seed'])

def identity(row):
    return target(row)+(tuple(zip(row['reference_image_sha256'],row['_reference_mask_sha256'])),)

def gpu(row):
    names={v.get('device_name') for v in row.get('cuda_memory',{}).values() if isinstance(v,dict)}
    return next(iter(names)) if len(names)==1 and None not in names else None

def pooled(rows):
    classes=defaultdict(Counter)
    for row in rows:classes[(row['fold'],row['concept_id'])].update(row['counts'])
    folds=defaultdict(list)
    for (f,_),c in classes.items():
        u=c['tp']+c['fp']+c['fn']
        if u:folds[f].append(c['tp']/u)
    return avg(avg(v) for v in folds.values())

def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze(evidence,output,reps,seed):
    core=evidence/'runtime/runs/ics-batch2-20261116/scopes/core'
    plan=json.loads((core/'plan.json').read_text())
    def mapped(p):return evidence/'runtime'/Path(p).relative_to(ORIGINAL)
    rows_by_method={};by_input={};failures={};files=[];jobs=[];outputs=[]
    for method in METHODS:
        rows=[];tasks=[t for t in plan['tasks'] if t['config_data']['experiment_id']=='batch2_'+method]
        assert len(tasks)==56
        groups=defaultdict(list)
        for task in tasks:groups[task['output']].append(task)
        assert len(groups)==14
        failure_attempts=[]
        for orig,tt in sorted(groups.items()):
            out=mapped(orig);meta=json.loads((out/'run.json').read_text())
            manifest=json.loads(mapped(meta['manifest_path']).read_text())
            episodes={e['episode_id']:e for e in manifest['episodes']}
            rr=[]
            for p in sorted((out/'episodes').glob('*.json')):
                row=json.loads(p.read_text());files.append(p)
                row['_reference_mask_sha256']=[x['mask_sha256'] for x in episodes[row['episode_id']]['references']]
                row['_method']=method;rr.append(row)
            if len(rr)!=300 or len({r['episode_id'] for r in rr})!=300:
                raise ValueError(f'Incomplete/duplicate canonical rows: {out}')
            rows.extend(rr);outputs.append(str(out.relative_to(evidence)))
            for p in (out/'attempts').rglob('*.json') if (out/'attempts').exists() else []:
                failure_attempts.append(json.loads(p.read_text()))
            shard_info={}
            for p in (out/'shards').glob('*.json'):
                v=json.loads(p.read_text());shard_info[v['shard_index']]=v
            for t in tt:
                status=json.loads((core/'jobs'/t['id']/'status.json').read_text())
                shard=shard_info.get(t['shard_index'],{})
                rs=[r for r in rr if r['shard_index']==t['shard_index']]
                names={gpu(r) for r in rs};name=next(iter(names)) if len(names)==1 else 'MIXED_OR_UNKNOWN'
                sampling=status.get('gpu_sampling',{})
                start,end=status.get('start'),status.get('end')
                wall=end-start if start is not None and end is not None and end>=start else None
                ms,mr=status.get('mllm_server_start'),status.get('mllm_server_ready')
                startup=mr-ms if ms is not None and mr is not None and mr>=ms else None
                assignment=status.get('gpu_assignment')
                n_gpu=len(assignment) if isinstance(assignment,list) else None
                peak=sampling.get('owned_simultaneous_total_peak_mib')
                jobs.append({'method':method,'dataset':rs[0]['dataset'],'fold':rs[0]['fold'],
                    'task_id':t['id'],'slurm_job_id':status.get('slurm_job_id'),'status':status.get('status'),
                    'runner_gpu_model':name,'canonical_episodes':len(rs),'allocated_gpu_count':n_gpu,
                    'worker_wall_seconds':wall,'allocated_gpu_seconds_proxy':wall*n_gpu if wall is not None and n_gpu is not None else None,
                    'mllm_startup_seconds':startup,'runner_model_load_seconds':shard.get('model_load_seconds'),
                    'sampled_owned_simultaneous_peak_gib':peak/1024 if peak is not None else None,
                    'sampling_status':sampling.get('status'),'memory_samples':sampling.get('samples'),
                    'memory_failed_samples':sampling.get('failed_samples')})
        statuses=Counter(r['status'] for r in rows)
        failures[method]={'expected':4200,'canonical_observed':len(rows),'canonical_statuses':dict(statuses),
            'archived_attempt_records':len(failure_attempts),'archived_statuses':dict(Counter(a.get('status') for a in failure_attempts)),
            'note':'Exported canonical and archived records only; absence of archived errors does not prove no historical scheduler/infrastructure failures.'}
        if len(rows)!=4200 or statuses!={'ok':4200} or any(r.get('iou') is None for r in rows):
            raise ValueError('This B2 analysis requires all 4200 canonical successful rows per configuration; inspect failure summary first')
        mapping={identity(r):r for r in rows}
        if len(mapping)!=4200:raise ValueError('Duplicate exact input within configuration')
        rows_by_method[method]=rows;by_input[method]=mapping
    keyset=set(by_input[METHODS[0]])
    if any(set(by_input[m])!=keyset for m in METHODS):raise ValueError('Four configurations lack identical exact input coverage')
    queries=[];episode_csv=[]
    for key in sorted(keyset):
        base=by_input[METHODS[0]][key]
        item={k:base[k] for k in ('dataset','fold','episode_id','concept_id','query_image_sha256','query_mask_sha256')}
        item['reference_image_sha256']=base['reference_image_sha256'][0];item['reference_mask_sha256']=base['_reference_mask_sha256'][0]
        for m in METHODS:
            r=by_input[m][key];item[m+'_iou']=r['iou'];item[m+'_inference_seconds']=r.get('inference_seconds');item[m+'_runner_gpu']=gpu(r)
            for metric in ('sam_calls','mllm_requests','candidate_score_evaluations','search_nodes','search_restarts'):
                item[m+'_'+metric]=r.get('telemetry',{}).get(metric)
        episode_csv.append(item)
    by_target=defaultdict(list)
    for key in sorted(keyset):by_target[key[:-1]].append(key)
    for key,inputs in sorted(by_target.items()):
        if len(inputs)!=3 or len({x[-1] for x in inputs})!=3:raise ValueError('Expected three distinct references per query')
        q=dict(zip(('dataset','split','fold','track','query_image_sha256','query_mask_sha256','concept_id','base_seed'),key));q['references']=3
        for m in METHODS:
            vals=[by_input[m][i]['iou'] for i in inputs]
            q[m+'_iou']=avg(vals);q[m+'_reference_population_sd']=float(np.std(vals));q[m+'_worst_reference_iou']=min(vals)
        for a,b in PAIRS:q[a+'_minus_'+b+'_pp']=100*(q[a+'_iou']-q[b+'_iou'])
        queries.append(q)
    datasets={};summary_csv=[];paired_csv=[];fold_csv=[];cost_csv=[];telemetry_csv=[];memory_csv=[]
    for ds in ('coco','lvis'):
        qq=[q for q in queries if q['dataset']==ds];clusters=[q['query_image_sha256'] for q in qq];folds=[q['fold'] for q in qq]
        if len(set(clusters))!=len(qq):raise ValueError('Unexpected multiple targets per query image')
        result={'unique_query_images':len(qq),'episodes_per_method':3*len(qq),'folds':len(set(folds)),
                'references_per_query':3,'methods':{},'paired_comparisons':{},'matched_cost_strata':[],'fold_results':[]}
        for m in METHODS:
            rr=[r for r in rows_by_method[m] if r['dataset']==ds]
            est=boot([q[m+'_iou'] for q in qq],clusters,reps,seed)
            est['fold_stratified_ci95_sensitivity']=boot([q[m+'_iou'] for q in qq],clusters,reps,seed,folds)['ci95']
            est['mean_fold_class_pooled_iou']=pooled(rr)
            est['mean_reference_population_sd']=avg(q[m+'_reference_population_sd'] for q in qq)
            est['mean_worst_reference_iou']=avg(q[m+'_worst_reference_iou'] for q in qq)
            est['runner_gpu_counts']=dict(Counter(gpu(r) for r in rr));result['methods'][m]=est
            summary_csv.append({'dataset':ds,'method':m,'episodes':len(rr),'unique_queries':len(qq),
                'equal_query_iou_pct':100*est['mean'],'ci95_low_pct':100*est['ci95'][0],'ci95_high_pct':100*est['ci95'][1],
                'mean_fold_class_pooled_iou_pct':100*est['mean_fold_class_pooled_iou'],
                'mean_reference_sd_pp':100*est['mean_reference_population_sd'],'mean_worst_reference_iou_pct':100*est['mean_worst_reference_iou']})
            tel={'dataset':ds,'method':m,'episodes':len(rr)}
            for metric in ('sam_calls','mllm_requests','candidate_score_evaluations','search_nodes','search_restarts'):
                vals=[r.get('telemetry',{}).get(metric) for r in rr];tel[metric+'_observed_n']=sum(x is not None for x in vals);tel[metric+'_mean']=avg(x for x in vals if x is not None)
            telemetry_csv.append(tel)
            for name in sorted({j['runner_gpu_model'] for j in jobs if j['method']==m and j['dataset']==ds},key=str):
                jj=[j for j in jobs if j['method']==m and j['dataset']==ds and j['runner_gpu_model']==name]
                mem={'dataset':ds,'method':m,'runner_gpu_model':name,'tasks':len(jj)}
                for metric in ('mllm_startup_seconds','runner_model_load_seconds','sampled_owned_simultaneous_peak_gib','worker_wall_seconds','allocated_gpu_seconds_proxy'):
                    val=distribution(j[metric] for j in jj);mem[metric+'_observed_n']=val['n'];mem[metric+'_median']=val['median'];mem[metric+'_sum']=val['sum']
                memory_csv.append(mem)
        for a,b in PAIRS:
            label=a+'_minus_'+b;vals=[q[label+'_pp'] for q in qq];est=boot(vals,clusters,reps,seed)
            est['fold_stratified_ci95_sensitivity']=boot(vals,clusters,reps,seed,folds)['ci95']
            est.update(direction=a+' minus '+b,unit='percentage points',primary=(a,b) in PRIMARY_PAIRS,
                queries_positive=sum(v>1e-12 for v in vals),queries_negative=sum(v< -1e-12 for v in vals),queries_zero=sum(abs(v)<=1e-12 for v in vals))
            result['paired_comparisons'][label]=est
            paired_csv.append({'dataset':ds,'comparison':label,'primary':est['primary'],'unique_queries':len(qq),
                'mean_delta_pp':est['mean'],'ci95_low_pp':est['ci95'][0],'ci95_high_pp':est['ci95'][1],
                'positive_queries':est['queries_positive'],'negative_queries':est['queries_negative'],'zero_queries':est['queries_zero']})
            gpu_pairs=defaultdict(list)
            for key in sorted(keyset):
                ar,br=by_input[a][key],by_input[b][key]
                if ar['dataset']!=ds:continue
                name=gpu(ar)
                if name and name==gpu(br) and all(isinstance(r.get('inference_seconds'),(int,float)) and np.isfinite(r['inference_seconds']) and r['inference_seconds']>0 for r in (ar,br)):
                    gpu_pairs[name].append((ar,br))
            for name,pairs in sorted(gpu_pairs.items()):
                aa=distribution(ar['inference_seconds'] for ar,_ in pairs);bb=distribution(br['inference_seconds'] for _,br in pairs)
                tq=defaultdict(list)
                for ar,br in pairs:tq[target(ar)].append((ar,br))
                deltas=[avg(100*(ar['iou']-br['iou']) for ar,br in refs) for refs in tq.values()]
                cl=[k[4] for k in tq]
                sensitivity=boot(deltas,cl,reps,seed)
                row={'dataset':ds,'comparison':a+'_over_'+b,'runner_gpu_model':name,'exact_input_pairs':len(pairs),
                    'matched_query_targets':len(tq),'queries_with_all_three_matched_references':sum(len(v)==3 for v in tq.values()),
                    'first_mean_seconds':aa['mean'],'second_mean_seconds':bb['mean'],'first_median_seconds':aa['median'],'second_median_seconds':bb['median'],
                    'first_p95_seconds':aa['p95'],'second_p95_seconds':bb['p95'],'ratio_of_mean_inference_seconds_first_over_second':aa['mean']/bb['mean'],
                    'first_allocated_gpus':1 if a=='insid3_full' else 2,'second_allocated_gpus':1 if b=='insid3_full' else 2,
                    'same_subset_query_accuracy_delta_pp':sensitivity['mean'],'same_subset_accuracy_ci95_low_pp':sensitivity['ci95'][0],
                    'same_subset_accuracy_ci95_high_pp':sensitivity['ci95'][1]}
                result['matched_cost_strata'].append(row);cost_csv.append(row)
        for f in sorted(set(folds)):
            qf=[q for q in qq if q['fold']==f];r={'dataset':ds,'fold':f,'unique_queries':len(qf),'episodes_per_method':3*len(qf),'observed_classes':len({q['concept_id'] for q in qf})}
            for m in METHODS:
                r[m+'_equal_query_iou_pct']=100*avg(q[m+'_iou'] for q in qf)
                r[m+'_class_pooled_iou_pct']=100*pooled(x for x in rows_by_method[m] if x['dataset']==ds and x['fold']==f)
            for a,b in PAIRS:r[a+'_minus_'+b+'_pp']=avg(q[a+'_minus_'+b+'_pp'] for q in qf)
            fold_csv.append(r);result['fold_results'].append(r)
        datasets[ds]=result
    files.sort()
    report={'schema_version':1,'analysis':'B2 fresh-query matched systems and CG-ICS search budgets',
       'source_outputs':len(outputs),'canonical_episodes':len(files),'methods':list(METHODS),'comparisons':[a+'_minus_'+b for a,b in PAIRS],
       'bootstrap_replicates':reps,'bootstrap_seed':seed,
       'source_episode_inventory_sha256':hashlib.sha256(''.join(str(p.relative_to(evidence))+':'+hashlib.sha256(p.read_bytes()).hexdigest()+'\n' for p in files).encode()).hexdigest(),
       'accuracy_estimand':'Mean IoU over three distinct references per query-target, then equal query-target mean within dataset. All four configurations cover identical exact inputs. 100 query images per fold.',
       'bootstrap_scope':'Query-image cluster percentile bootstrap, with all references paired; fixed-fold stratification is a separately labelled sensitivity. Exploratory, no multiplicity correction, one model seed.',
       'cost_estimand':'Same recorded runner GPU model and exact query/reference inputs only; ratio of arithmetic mean inference_seconds. Each stratum reports accuracy on its own matched-input subset. Startup, contention and MLLM GPU model are not controlled by runner matching.',
       'cost_limitations':['INSID3 allocates one GPU; CG-ICS variants allocate two including the owned MLLM. Runner-time ratios are not total GPU-efficiency ratios.','No cross-GPU pooled speedup is reported. Different strata may have different query/fold composition.','Memory values are sampled owned-process simultaneous peaks and include vLLM reserved/cache memory, not minimum model requirements.','Worker cost records refer to exported current attempts; they do not establish total historical retry/queue cost.','BON1 changes initial candidate budget to one while retaining full-search expansion/restarts; it is not the single-concept no-search route.'],
       'failure_coverage':failures,'datasets':datasets,'telemetry_summary':telemetry_csv,'worker_cost_summary':memory_csv,
       'scientific_acceptance':'Statistics only; require separate protocol and pixel audits before final report acceptance.'}
    output.mkdir(parents=True,exist_ok=True);(output/'statistics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for name,rr in [('b2_summary.csv',summary_csv),('b2_paired_comparisons.csv',paired_csv),('b2_fold_results.csv',fold_csv),
        ('b2_matched_cost.csv',cost_csv),('b2_telemetry.csv',telemetry_csv),('b2_worker_cost_summary.csv',memory_csv),
        ('b2_worker_costs.csv',jobs),('b2_query_results.csv',queries),('b2_episode_results.csv',episode_csv)]:csvwrite(output/name,rr)
    print(json.dumps({'accuracy':summary_csv,'primary_comparisons':[x for x in paired_csv if x['primary']],
        'primary_matched_cost':[x for x in cost_csv if tuple(x['comparison'].split('_over_')) in PRIMARY_PAIRS]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,default=Path('b2_evidence'))
    p.add_argument('--output',type=Path,default=Path('b2_analysis'));p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=6190)
    a=p.parse_args();analyze(a.evidence.resolve(),a.output.resolve(),a.replicates,a.seed)
