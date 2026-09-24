"""Exhaustive frozen-fold score/rank comparisons, not biological validation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.portable_io import load_bundle,load_gate,sha256
from src.response_pipeline import build_random_framework_inputs
from src.release_runtime import score_gate,evaluate_scores
from src.unified_dstar_ensemble import rank_descending
from src.release_protocol import assert_runtime_compatibility


def verify_fold(path):
    path=Path(path);source,meta=load_bundle(path)
    if source.stage!='test':raise ValueError('Frozen manuscript comparison requires test stage')
    inputs=build_random_framework_inputs(source,beta=meta['beta'])
    final=score_gate(inputs,load_gate(path.parent/'gate'),beta=meta['beta'])
    channels={'S':inputs.S,'C':inputs.C,'Dstar':inputs.Dstar,'B':inputs.backbone,'F':final}
    ref=json.loads((path.parent/'reference.json').read_text())
    if sha256(path.parent/'reference.npz')!=ref['reference_sha256']:
        raise ValueError('Reference score checksum mismatch')
    features_hash=hashlib.sha256(inputs.features.tobytes()).hexdigest()
    if features_hash!=ref['features_sha256']:raise ValueError('Reference feature tensor differs')
    report={'split':source.split,'fold':source.fold,'queries':len(source.query_ids),
            'candidate_count':len(source.candidate_hashes),'features_equal':True,
            'bundle_sha256':sha256(path),'features_sha256':features_hash,'channels':{}}
    with np.load(path.parent/'reference.npz',allow_pickle=False) as archive:
        for name,scores in channels.items():
            reference=archive[name]
            np.testing.assert_allclose(scores,reference,rtol=1e-10,atol=1e-10,err_msg=name)
            ranks=rank_descending(np.where(inputs.eligible_mask,scores,np.nan),inputs.candidate_hashes)
            ref_ranks=rank_descending(np.where(inputs.eligible_mask,reference,np.nan),inputs.candidate_hashes)
            np.testing.assert_array_equal(ranks,ref_ranks,err_msg=name)
            metrics,rows=evaluate_scores(inputs,scores)
            report['channels'][name]={'max_score_error':float(np.max(np.abs(scores-reference))),
                                      'all_active_ranks_equal':True,**asdict(metrics)}
            if name=='F':
                old=pd.read_csv(path.parent/'frozen_final_metrics.tsv',sep='\t').set_index('query_id')
                aligned=old.loc[list(source.query_ids),'best_positive_rank'].to_numpy()
                np.testing.assert_array_equal(rows.best_positive_rank.to_numpy(),aligned,
                                              err_msg='Published final target ranks differ')
    report['frozen_final_target_ranks_equal']=True
    return report


def verify_all(root):
    root=Path(root);paths=sorted(root.glob('*/fold_*/test/bundle.json'))
    if not paths:raise ValueError('No test bundles found')
    reports=[]
    for p in paths:
        report=verify_fold(p);reports.append(report)
        print(f"Verified {report['split']}/{report['fold']}: all channel scores and ranks match",flush=True)
    aggregates=[]
    expected=pd.read_csv(root/'expected_metrics.tsv',sep='\t')
    expected_folds={(split,fold) for split in ('Random-edge','TF-50','Ligand-Morgan-0.5') for fold in range(5)}
    if len(reports)!=15 or {(r['split'],r['fold']) for r in reports}!=expected_folds:
        raise ValueError('Verification requires the exact benchmark set of 15 folds')
    for split in sorted({r['split'] for r in reports}):
        folds=[r for r in reports if r['split']==split]
        for name in ('S','C','Dstar','B','F'):
            n=sum(r['queries'] for r in folds)
            row={'split':split,'model':name,'query_count':n}
            for k in ('h10','h50','mrr'):
                row[k]=sum(r['queries']*r['channels'][name][k] for r in folds)/n
            old=expected[(expected['split']==split)&(expected['model']==name)]
            if len(old)!=1:raise ValueError(f'Missing frozen summary: {split}/{name}')
            for k,col in [('h10','H10'),('h50','H50'),('mrr','MRR')]:
                np.testing.assert_allclose(row[k],float(old.iloc[0][col]),rtol=0,atol=1e-12)
            aggregates.append(row)
    return {'schema':'ligand2tf-verification-1','all_15_folds':True,'folds':reports,
            'runtime':assert_runtime_compatibility(),
            'aggregates':aggregates,'scope':'Frozen-score/checkpoint replay; no encoder feature generation or retraining.'}
