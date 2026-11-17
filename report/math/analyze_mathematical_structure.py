#!/usr/bin/env python3
"""Recompute descriptive ICS interactions and exact metric identities.

Read-only use of the audited first-batch ledger. No model execution, source
mutation, test-set tuning, or second-batch performance estimation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

IDS = ('manifest_sha256','protocol_id','evaluation_protocol','dataset','fold',
       'concept_id','query_image_sha256','query_mask_sha256',
       'reference_image_sha256','reference_seed','base_seed','episode_seed','nshot')
FACTORS = [
 ('Debias benefit with self/cross minus benefit with seed-only',
  ['insid3_component_full','insid3_no_debias','insid3_no_aggregation','insid3_no_debias_seed_only']),
 ('Visual-box benefit with search minus benefit with single concept',
  ['cgics_component_full','cgics_text_only','cgics_single_concept_full','cgics_single_concept_text']),
]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def qkey(row):
    return tuple(row[k] for k in ('query_image_sha256','query_mask_sha256','concept_id'))

def cluster_mean_ci(query_values, seed_text, reps=2000):
    # The target is a mean over query targets. Resampling image clusters retains
    # all targets/references from each selected image; it does not weight each
    # image equally when one image contains multiple target classes.
    images=defaultdict(list)
    for q,v in query_values.items(): images[q[0]].append(float(np.mean(v)))
    keys=sorted(images)
    sums=np.array([sum(images[k]) for k in keys])
    counts=np.array([len(images[k]) for k in keys])
    rng=np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16],16))
    indices=rng.integers(0,len(keys),size=(reps,len(keys)))
    draw=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return {'query_targets':int(counts.sum()),'query_images':len(keys),
            'equal_query_mean_pp':float(100*sums.sum()/counts.sum()),
            'image_cluster_ci95_pp':(100*np.quantile(draw,[.025,.975])).tolist(),
            'bootstrap_replicates':reps}

def paired_rows(conn, configs, dataset):
    placeholders=','.join('?' for _ in configs)
    rows=conn.execute(f'SELECT * FROM formal WHERE config IN ({placeholders}) AND dataset=?',[*configs,dataset])
    byconfig={c:{} for c in configs}
    for row in rows:
        r=dict(row); assert r['manifest_id'].endswith('_component')
        assert r['status']=='ok' and r['iou'] is not None
        key=(r['manifest_id'],r['episode_id'])
        assert key not in byconfig[r['config']]
        byconfig[r['config']][key]=r
    keys=set(byconfig[configs[0]])
    assert all(set(byconfig[c])==keys for c in configs)
    for key in sorted(keys):
        group=[byconfig[c][key] for c in configs]
        for other in group[1:]:
            assert all(group[0][k]==other[k] for k in IDS)
            assert group[0]['tp']+group[0]['fn']==other['tp']+other['fn']
            assert group[0]['tn']+group[0]['fp']==other['tn']+other['fp']
        yield group

def interactions(conn):
    output=[]
    for dataset in ('coco','lvis'):
        for question,configs in FACTORS:
            values=defaultdict(list);same_device=defaultdict(lambda:defaultdict(list));n=0
            for rows in paired_rows(conn,configs,dataset):
                a,b,c,d=rows
                values[qkey(a)].append(a['iou']-b['iou']-c['iou']+d['iou']);n+=1
                if len({r['gpu_device'] for r in rows})==1:
                    same_device[a['gpu_device']][qkey(a)].append(a['iou']-b['iou']-c['iou']+d['iou'])
            output.append({'question':question,'dataset':dataset,'configs':configs,
                           'weights':[1,-1,-1,1],'episodes':n,
                           **cluster_mean_ci(values,question+dataset),
                           'all_four_same_recorded_gpu_sensitivity':[
                              {'gpu':gpu,'episodes':sum(len(v) for v in q.values()),
                               **cluster_mean_ci(q,question+'|same-device|'+dataset+'|'+gpu)}
                              for gpu,q in sorted(same_device.items())],
                           'sensitivity_note':'Same-device subsets can have different folds and composition; absence of an interval effect on a small subset is not evidence that hardware caused the full-cohort effect.'})
    return output

def search_heterogeneity(conn):
    output=[]
    configs=['cgics_single_concept_full','cgics_component_full']
    for dataset in ('coco','lvis'):
        values=defaultdict(list);devices=defaultdict(int)
        for early,search in paired_rows(conn,configs,dataset):
            values[qkey(early)].append((early['iou'],search['iou']))
            devices[(early['gpu_device'],search['gpu_device'])]+=1
        means=np.array([np.mean(v,axis=0) for _,v in sorted(values.items())])
        delta=means[:,1]-means[:,0]
        output.append({'dataset':dataset,'query_targets':len(values),
          'early_equal_query_iou_percent':float(100*means[:,0].mean()),
          'search_equal_query_iou_percent':float(100*means[:,1].mean()),
          'mean_search_gain_pp':float(100*delta.mean()),
          'search_better_queries':int((delta>1e-12).sum()),
          'equal_queries':int((np.abs(delta)<=1e-12).sum()),
          'early_better_queries':int((delta< -1e-12).sum()),
          'search_gain_over_5pp_queries':int((delta>.05).sum()),
          'search_loss_over_5pp_queries':int((delta< -.05).sum()),
          'hindsight_max_of_query_means_iou_percent':float(100*means.max(axis=1).mean()),
          'hindsight_gain_over_always_search_pp':float(100*np.maximum(-delta,0).mean()),
          'observed_device_pairs':[{ 'early':a,'search':b,'episodes':n} for (a,b),n in sorted(devices.items())],
          'interpretation':'Hindsight selection uses ground truth after averaging references; descriptive upper envelope, not an inference policy or expected gate accuracy.'})
    return output

def metric_decomposition(conn):
    output=[]
    for config in ('insid3_full','cgics_full'):
        for dataset in ('coco','lvis'):
            classes=defaultdict(list)
            for row in conn.execute('SELECT * FROM formal WHERE config=? AND dataset=? AND manifest_id LIKE ?',
                                    (config,dataset,'%_standard')):
                r=dict(row);assert r['status']=='ok' and r['iou'] is not None
                classes[(r['fold'],r['concept_id'])].append(r)
            perclass=[]
            for (fold,cid),rows in sorted(classes.items()):
                tp=np.array([r['tp'] for r in rows],dtype=float)
                fp=np.array([r['fp'] for r in rows],dtype=float)
                gt=tp+np.array([r['fn'] for r in rows],dtype=float)
                union=gt+fp;assert (union>0).all()
                j=tp/union
                assert np.max(np.abs(j-np.array([r['iou'] for r in rows])))<1e-12
                cov=lambda x:float(np.mean((x-x.mean())*(j-j.mean()))/union.mean())
                pooled=float(tp.sum()/union.sum());mean=float(j.mean())
                covgt,covfp=cov(gt),cov(fp)
                error=abs((pooled-mean)-(covgt+covfp))
                assert error<1e-12
                perclass.append({'fold':fold,'concept':cid,'episodes':len(rows),
                    'pooled':pooled,'mean':mean,'gt_covariance':covgt,
                    'fp_covariance':covfp,'identity_error':error})
            # Preserve exactly the report's equal-fold, then equal-class order.
            folds=defaultdict(list)
            for row in perclass:folds[row['fold']].append(row)
            avg=lambda key:float(np.mean([np.mean([c[key] for c in group]) for group in folds.values()]))
            output.append({'config':config,'dataset':dataset,'classes':len(classes),
              'episodes':sum(x['episodes'] for x in perclass),
              'mean_fold_class_episode_iou_percent':100*avg('mean'),
              'mean_fold_class_pooled_iou_percent':100*avg('pooled'),
              'pooled_minus_class_episode_mean_pp':100*(avg('pooled')-avg('mean')),
              'gt_covariance_contribution_pp':100*avg('gt_covariance'),
              'fp_covariance_contribution_pp':100*avg('fp_covariance'),
              'max_identity_error':max(x['identity_error'] for x in perclass),
              'interpretation':'An exact weighting identity; covariance components are not causal contributions.'})
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--db',type=Path,default=Path(__file__).resolve().parents[2]/'audit_20261116/full_analysis/episodes.sqlite')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
    conn=sqlite3.connect('file:'+str(args.db.resolve())+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    result={'status':'PASS','source_db':str(args.db.resolve()),'source_read_only':True,
      'component_interactions':interactions(conn),'search_heterogeneity':search_heterogeneity(conn),
      'metric_decomposition':metric_decomposition(conn),
      'second_batch_results_used':False,'new_model_runs':0}
    previous=json.loads((args.db.parent/'components_analysis.json').read_text())
    for item in result['component_interactions']:
        old=next(x for x in previous['factor_interactions'] if x['question']==item['question'] and x['dataset']==item['dataset'])
        assert abs(item['equal_query_mean_pp']-100*old['mean_query_target_delta'])<1e-10
        assert np.max(np.abs(np.array(item['image_cluster_ci95_pp'])-100*np.array(old['query_image_cluster_ci95'])))<1e-10
    result['verification']={'prior_component_effects_and_cis_reproduced':True,
      'metric_identity_error_below':1e-12,
      'source_summaries_sha256':{p.name:digest(p) for p in [args.db.parent/'components_analysis.json',args.db.parent/'parameter_cost_size.json']},
      'script_sha256':digest(__file__)}
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/'mathematical_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','interactions':len(result['component_interactions']),
       'metric_decompositions':len(result['metric_decomposition']),
       'search_heterogeneity':result['search_heterogeneity']},indent=2))

if __name__=='__main__':main()
