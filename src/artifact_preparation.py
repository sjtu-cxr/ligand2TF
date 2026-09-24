"""Explicit portable input construction; no implicit research-directory access."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.portable_io import sha256,load_bundle
from src.release_protocol import PROTOCOL


def prepare_protein(candidates_path,alignments_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    frame=pd.read_csv(candidates_path,sep='\t',dtype=str,keep_default_na=False)
    if not {'sequence_md5','protein_sequence'}<=set(frame):raise ValueError('Candidate table requires exact identities and sequences')
    ids=tuple(frame.sequence_md5);index={v:i for i,v in enumerate(ids)}
    if not ids or len(ids)!=len(index):raise ValueError('Candidate identities must be unique')
    lengths=dict(zip(ids,frame.protein_sequence.str.len()))
    scores=np.zeros((len(ids),len(ids)),dtype=np.float32)
    with Path(alignments_path).open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            fields=line.rstrip('\n').split('\t')
            if len(fields)<12:raise ValueError('Expected standard 12-column MMseqs m8 alignment')
            q,t=fields[:2]
            if q not in index or t not in index or q==t:continue
            aligned=int(float(fields[3]));qlen=lengths[q];tlen=lengths[t]
            if min(qlen,tlen,aligned)<=0:continue
            value=np.float32(float(fields[2])/100*min(aligned/qlen,aligned/tlen))
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid identity-coverage value')
            scores[index[q],index[t]]=max(scores[index[q],index[t]],value)
    np.fill_diagonal(scores,np.float32(1))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,candidate_hashes=np.asarray(ids),scores=scores)


def create_bundle(fit_edges,protein,seed_scores,query_manifest,output,*,split,fold,stage,beta):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    meta=json.loads(Path(query_manifest).read_text());queries=tuple(meta['query_ids'])
    with np.load(protein,allow_pickle=False) as a:candidates=tuple(a['candidate_hashes'].astype(str))
    if len(seed_scores)!=3:raise ValueError('Provide three Dstar archives in the frozen seed order')
    arrays=[]
    for path in seed_scores:
        with np.load(path,allow_pickle=False) as a:
            if tuple(a['query_ids'].astype(str))!=queries or tuple(a['candidate_hashes'].astype(str))!=candidates:
                raise ValueError('Dstar archive identity/order mismatch')
            arrays.append(a['scores'].copy())
    if any(a.shape!=(len(queries),len(candidates)) for a in arrays):raise ValueError('Dstar shape mismatch')
    output.mkdir(parents=True)
    np.savez_compressed(output/'dstar_scores.npz',scores=np.stack(arrays))
    files={name:{'path':os.path.relpath(Path(path).resolve(),output.resolve()),'sha256':sha256(path)}
           for name,path in [('fit_edges',fit_edges),('protein_similarity',protein),('dstar_scores',output/'dstar_scores.npz')]}
    meta.update(schema='ligand2tf-bundle-1',split=split,fold=fold,stage=stage,beta=beta,
                candidate_hashes=candidates,seeds=PROTOCOL['seeds'],files=files)
    manifest=output/'bundle.json';manifest.write_text(json.dumps(meta,indent=2)+'\n')
    load_bundle(manifest)
    return manifest


def create_example(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);rng=np.random.default_rng(42)
    candidates=[f'p{i:03d}' for i in range(140)]
    matrix=rng.uniform(0,.5,(140,140)).astype(np.float32);np.fill_diagonal(matrix,1)
    protein=output/'protein.npz';np.savez_compressed(protein,candidate_hashes=candidates,scores=matrix)
    fit=output/'fit.tsv'
    pd.DataFrame([{'ligand_key':'C'*i,'sequence_md5':'p000','split_unit':'C'*i} for i in range(1,13)]).to_csv(fit,sep='\t',index=False)
    for stage,counts in [('validation',range(1,13)),('test',range(13,17))]:
        queries=['C'*i for i in counts]
        query_path=output/f'{stage}_queries.json'
        query_path.write_text(json.dumps({'query_ids':queries,'split_units':queries,
                                         'relevant_hashes':[['p010'] for _ in queries]}))
        paths=[]
        for seed in PROTOCOL['seeds']:
            path=output/f'{stage}_{seed}.npz';values=rng.normal(size=(len(queries),140)).astype(np.float32)
            np.savez_compressed(path,scores=values,query_ids=queries,candidate_hashes=candidates);paths.append(path)
        create_bundle(fit,protein,paths,query_path,output/stage,split='synthetic',fold=0,stage=stage,beta=.5)
    (output/'README.txt').write_text('Synthetic interface smoke test only. No biological performance claim.\n')
