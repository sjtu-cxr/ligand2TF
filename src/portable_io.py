"""Portable, hash-checked fold bundles and safe NumPy gate checkpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.response_pipeline import FoldEvidenceSource
from src.unified_bscd_residual import FEATURE_NAMES
from src.release_protocol import PROTOCOL, assert_runtime_compatibility


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_file(manifest_path, record):
    path=(Path(manifest_path).parent/record['path']).resolve()
    if sha256(path)!=record['sha256']:
        raise ValueError(f'Artifact checksum mismatch: {path.name}')
    return path


def load_bundle(path):
    assert_runtime_compatibility()
    path=Path(path).resolve(); meta=json.loads(path.read_text())
    if meta.get('schema')!='ligand2tf-bundle-1':
        raise ValueError('Unsupported bundle schema')
    if meta.get('seeds')!=PROTOCOL['seeds'] or meta.get('stage') not in {'validation','test','predict'}:
        raise ValueError('Bundle seed order or fitting stage is invalid')
    if meta.get('beta') not in PROTOCOL['beta_grid']:
        raise ValueError('Transfer beta is outside the frozen grid')
    queries=tuple(meta['query_ids']); candidates=tuple(meta['candidate_hashes'])
    units=tuple(meta['split_units']); relevant=tuple(tuple(v) for v in meta.get('relevant_hashes',[[] for _ in queries]))
    if not queries or len(queries)!=len(set(queries)) or not candidates or len(candidates)!=len(set(candidates)):
        raise ValueError('Queries and candidates must be nonempty and unique')
    if len(units)!=len(queries) or len(relevant)!=len(queries):
        raise ValueError('Query metadata is misaligned')
    if any(not isinstance(v,str) or not v for v in (*queries,*candidates,*units)):
        raise ValueError('Identifiers must be nonempty strings')
    files=meta['files']; fit_path=checked_file(path,files['fit_edges'])
    protein_path=checked_file(path,files['protein_similarity'])
    dstar_path=checked_file(path,files['dstar_scores'])
    fit=pd.read_csv(fit_path,sep='\t',dtype=str,keep_default_na=False)
    if not {'ligand_key','sequence_md5'}<=set(fit.columns):
        raise ValueError('Fitting graph lacks required columns')
    pairs=set(fit[['ligand_key','sequence_md5']].itertuples(index=False,name=None))
    if any(p not in candidates for _,p in pairs):
        raise ValueError('Fitting candidate absent from library')
    masks=np.array([[(q,p) in pairs for p in candidates] for q in queries],dtype=bool)
    for q,responders in zip(queries,relevant,strict=True):
        if any((q,p) in pairs for p in responders):
            raise ValueError('Held-out responder overlaps fitting graph')
        if any(p not in candidates for p in responders):
            raise ValueError('Relevant candidate absent from library')
    with np.load(dstar_path,allow_pickle=False) as archive:
        scores=archive['scores'].copy()
    if scores.shape!=(3,len(queries),len(candidates)) or not np.all(np.isfinite(scores[:,~masks])):
        raise ValueError('Dstar scores must be three aligned finite active-library matrices')
    if np.isinf(scores).any():
        raise ValueError('Infinite Dstar score')
    source=FoldEvidenceSource(meta['split'],int(meta['fold']),meta['stage'],queries,units,candidates,
                              fit,masks,relevant,scores,protein_path,protein_path)
    return source,meta


def save_gate(path,payload):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    path.mkdir(parents=True)
    meta={k:v for k,v in payload.items() if k!='state_dicts'}
    states=payload['state_dicts']
    np.savez_compressed(path/'weights.npz',**{f'{i}:{k}':v.detach().cpu().numpy()
                                            for i,state in enumerate(states) for k,v in state.items()})
    meta['state_count']=len(states);meta['schema']='ligand2tf-gate-1'
    meta['weights_sha256']=sha256(path/'weights.npz')
    (path/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')


def load_gate(path):
    path=Path(path);meta=json.loads((path/'checkpoint.json').read_text())
    if meta.get('schema')!='ligand2tf-gate-1' or meta.get('feature_names')!=list(FEATURE_NAMES):
        raise ValueError('Checkpoint schema or feature order mismatch')
    if meta.get('training_seeds')!=PROTOCOL['seeds']:
        raise ValueError('Checkpoint seed order mismatch')
    expected=0 if meta['selected_config'] is None else 3
    if meta['state_count']!=expected or sha256(path/'weights.npz')!=meta['weights_sha256']:
        raise ValueError('Checkpoint count or checksum mismatch')
    states=[{} for _ in range(expected)]
    with np.load(path/'weights.npz',allow_pickle=False) as archive:
        for key in archive.files:
            index,name=key.split(':',1);values=archive[key]
            if not np.isfinite(values).all(): raise ValueError('Nonfinite checkpoint weight')
            states[int(index)][name]=torch.from_numpy(values.copy())
    return dict(meta,state_dicts=states)
