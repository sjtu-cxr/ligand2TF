"""Export local verification fixtures from a trusted frozen research workspace.

This is a maintainer migration tool, not a runtime dependency. Artifacts remain
under ignored data/. Original pickle-based torch checkpoints are loaded with
weights_only=True. No training or reference data mutation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['Random-edge','TF-50','Ligand-Morgan-0.5'])
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--stage',choices=['validation','test'],default='test')
    args=p.parse_args();root=args.reference_root.resolve();out=args.output.resolve()
    if out==root or out in root.parents:raise ValueError('Unsafe output directory')
    if args.split is None or args.fold is None:
        splits=[args.split] if args.split else ['Random-edge','TF-50','Ligand-Morgan-0.5']
        folds=[args.fold] if args.fold is not None else range(5)
        for split in splits:
            for fold in folds:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--reference-root',str(root),
                                '--output',str(out),'--split',split,'--fold',str(fold),'--stage',args.stage],check=True)
        return
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(root))
    from scripts.pipeline.build_random_framework_three_split_inputs import (
        load_fold_evidence_source,apply_dstar_seed_override,build_random_framework_inputs,
        _load_sequence_similarity)
    from scripts.pipeline.run_efit_refit_three_method_replay import esm2_archive_paths
    from scripts.pipeline.run_structure_augmented_scheme3 import _load_dense_seed_archives
    from scripts.pipeline.run_candidate_gated_top10_ranking import (
        CandidateGateConfig,_new_model,predict_candidate_gate,mean_rank_ensemble,
        active_backbone_rank_data,ranking_metrics)
    from scripts.pipeline.run_random_framework_three_split import adapt_trainer_inputs
    base=root/'data/model_training/v66/results/efit_refit_three_method_replay_20260830'
    dsroot=root/'data/model_training/v66/results/structure_aware_dstar_ablation_20260828'
    selections=[json.loads(s) for s in (base/'candidate_gate/validation_selections.jsonl').read_text().splitlines()]
    out.mkdir(parents=True,exist_ok=True)
    protein=out/'protein_similarity.npz'
    for selection in selections:
        split=selection['split'];fold=selection['fold']
        if args.split and split!=args.split:continue
        if args.fold is not None and fold!=args.fold:continue
        target=out/split/f'fold_{fold}'/args.stage
        if (target/'bundle.json').exists():
            print(f'Skipping complete bundle: {split}/{fold}/{args.stage}',flush=True);continue
        target.mkdir(parents=True,exist_ok=True)
        source=load_fold_evidence_source(split=split,fold=fold,stage=args.stage)
        paths=esm2_archive_paths(dsroot,split=split,fold=fold,stage=args.stage)
        source=apply_dstar_seed_override(source,_load_dense_seed_archives(paths,source))
        if not protein.exists():
            cache=_load_sequence_similarity(source)
            np.savez_compressed(protein,candidate_hashes=np.asarray(cache.candidate_hashes),scores=cache.scores)
        fit=source.fit_edges[['ligand_key','sequence_md5']].drop_duplicates()
        fit.to_csv(target/'fit_edges.tsv',sep='\t',index=False)
        np.savez_compressed(target/'dstar_scores.npz',scores=source.dstar_seed_scores)
        files={name:{'path':path,'sha256':digest(target/path)} for name,path in
               [('fit_edges','fit_edges.tsv'),('dstar_scores','dstar_scores.npz'),
                ('protein_similarity','../../../protein_similarity.npz')]}
        meta={'schema':'ligand2tf-bundle-1','split':split,'fold':fold,'stage':args.stage,
              'query_ids':source.query_ids,'split_units':source.split_units,
              'candidate_hashes':source.candidate_hashes,'relevant_hashes':source.relevant_hashes,
              'seeds':[42,20260717,20260718],'beta':selection['selected_beta'],'files':files,
              'reference_inputs':{str(path.relative_to(root)):digest(path) for path in paths}}
        inputs=build_random_framework_inputs(source,beta=selection['selected_beta'])
        arrays,metadata=adapt_trainer_inputs(inputs)
        model_path=base/'candidate_gate/models'/split/f'fold_{fold}/model_state.pt'
        payload=torch.load(model_path,map_location='cpu',weights_only=True)
        checkpoint=target/'gate';checkpoint.mkdir(exist_ok=True)
        np.savez_compressed(checkpoint/'weights.npz',**{f'{i}:{k}':v.detach().numpy()
                           for i,state in enumerate(payload['state_dicts']) for k,v in state.items()})
        cm={k:v for k,v in payload.items() if k!='state_dicts'}
        cm.update(schema='ligand2tf-gate-1',state_count=len(payload['state_dicts']),
                  weights_sha256=digest(checkpoint/'weights.npz'),beta=selection['selected_beta'])
        (checkpoint/'checkpoint.json').write_text(json.dumps(cm,indent=2)+'\n')
        if payload['selected_config'] is None:
            final=active_backbone_rank_data(inputs.backbone,inputs.eligible_mask,inputs.candidate_hashes)[0]
        else:
            predictions=[]
            for state in payload['state_dicts']:
                model=_new_model(CandidateGateConfig(**payload['selected_config']),torch.device('cpu'))
                model.load_state_dict(state)
                predictions.append(predict_candidate_gate(model,arrays,indexes=np.arange(len(inputs.query_ids)),
                                                         candidate_hashes=inputs.candidate_hashes,device=torch.device('cpu')))
            final=mean_rank_ensemble(predictions,inputs.candidate_hashes,inputs.eligible_mask)
        np.savez_compressed(target/'reference.npz',S=inputs.S,C=inputs.C,Dstar=inputs.Dstar,
                            B=inputs.backbone,F=final)
        golden={'features_sha256':hashlib.sha256(inputs.features.tobytes()).hexdigest(),
                'features_shape':list(inputs.features.shape),'reference_sha256':digest(target/'reference.npz'),
                'model_source_sha256':digest(model_path)}
        rows=pd.read_csv(base/'candidate_gate/per_query_metrics.tsv',sep='\t')
        if args.stage=='test':
            rows[(rows['split']==split)&(rows['fold']==fold)].to_csv(target/'frozen_final_metrics.tsv',sep='\t',index=False)
        (target/'reference.json').write_text(json.dumps(golden,indent=2)+'\n')
        (target/'bundle.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'Exported {split}/{fold}/{args.stage}: {len(inputs.query_ids)} queries',flush=True)
    summary=base/'manuscript_refresh_20260831/manuscript_main_metrics.tsv'
    pd.read_csv(summary,sep='\t').to_csv(out/'expected_metrics.tsv',sep='\t',index=False)

if __name__=='__main__':
    main()
