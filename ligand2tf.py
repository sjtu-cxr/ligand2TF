"""Portable train, rank, evaluate and frozen-replay commands."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    for name in ['predict','evaluate','train-gate']:
        c=commands.add_parser(name);c.add_argument('--bundle',type=Path,required=True)
        c.add_argument('--output',type=Path,required=True);c.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        if name!='train-gate':c.add_argument('--checkpoint',type=Path,required=True)
        if name=='predict':c.add_argument('--top-k',type=int,default=10)
        if name=='train-gate':c.add_argument('--smoke',action='store_true',help='One epoch; explicitly not formal training')
    c=commands.add_parser('verify');c.add_argument('--bundles',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('example');c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('prepare-protein');c.add_argument('--candidates',type=Path,required=True);c.add_argument('--alignments',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('bundle')
    for name in ['fit-edges','protein','queries','output']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--seed-scores',nargs=3,type=Path,required=True);c.add_argument('--split',required=True)
    c.add_argument('--fold',type=int,required=True);c.add_argument('--stage',choices=['validation','test','predict'],required=True);c.add_argument('--beta',type=float,required=True)
    c=commands.add_parser('train-dstar')
    for name in ['train','validation','features','output']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--seed',type=int,choices=[42,20260717,20260718],required=True);c.add_argument('--device',choices=['cpu','cuda'],default='cpu');c.add_argument('--smoke',action='store_true')
    c=commands.add_parser('score-dstar')
    for name in ['features','fit-edges','queries','weights','output']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args(argv)
    if args.output.exists():p.error('Output exists; choose a new destination')
    import torch
    torch.set_num_threads(1)
    if args.command in {'example','prepare-protein','bundle'}:
        from src.artifact_preparation import create_example,prepare_protein,create_bundle
        if args.command=='example':create_example(args.output)
        elif args.command=='prepare-protein':prepare_protein(args.candidates,args.alignments,args.output)
        else:create_bundle(args.fit_edges,args.protein,args.seed_scores,args.queries,args.output,split=args.split,fold=args.fold,stage=args.stage,beta=args.beta)
        return
    if args.command=='train-dstar':
        from src.dstar_workflow import train_dstar
        train_dstar(args.train,args.validation,args.features,args.output,seed=args.seed,device=args.device,smoke=args.smoke);return
    if args.command=='score-dstar':
        from src.dstar_workflow import score_dstar
        score_dstar(args.features,args.fit_edges,args.queries,args.weights,args.output,device=args.device);return
    if args.command=='verify':
        from src.release_verification import verify_all
        report=verify_all(args.bundles);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n');return
    from src.portable_io import load_bundle,load_gate,save_gate
    from src.response_pipeline import build_random_framework_inputs
    from src.release_runtime import fit_gate,score_gate,evaluate_scores
    source,meta=load_bundle(args.bundle)
    inputs=build_random_framework_inputs(source,beta=meta['beta'])
    if args.command=='train-gate':
        payload,report=fit_gate(inputs,beta=meta['beta'],epochs=1 if args.smoke else 30,device=args.device)
        save_gate(args.output,payload)
        (args.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');return
    scores=score_gate(inputs,load_gate(args.checkpoint),beta=meta['beta'],device=args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='evaluate':
        metrics,rows=evaluate_scores(inputs,scores)
        rows.to_csv(args.output,sep='\t',index=False)
        print(json.dumps(vars(metrics)));return
    if args.top_k<=0:p.error('top-k must be positive')
    import numpy as np
    import pandas as pd
    from src.unified_dstar_ensemble import rank_descending
    ranks=rank_descending(np.where(inputs.eligible_mask,scores,np.nan),inputs.candidate_hashes)
    records=[]
    for i,q in enumerate(inputs.query_ids):
        selected=np.flatnonzero(ranks[i]<=args.top_k)
        for j in sorted(selected,key=lambda j:ranks[i,j]):
            records.append({'query_id':q,'candidate_hash':inputs.candidate_hashes[j],
                            'rank':int(ranks[i,j]),'score':float(scores[i,j]),
                            'protein_transfer_available':bool(inputs.s_available[i,j]),
                            'ligand_transfer_available':bool(inputs.c_available[i,j])})
    pd.DataFrame(records).to_csv(args.output,sep='\t',index=False)

if __name__=='__main__':main()
