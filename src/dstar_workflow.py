"""Final dual-encoder validation training and E_fit refit from frozen features.

Raw pretrained encoder feature generation is intentionally separate from
training; this API consumes an explicit, portable numeric feature archive.
"""
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.release_protocol import PROTOCOL,make_dual_encoder
from src.portable_io import sha256
from src.unified_dstar_data import build_training_episodes,build_evaluation_queries
from src.unified_dstar_training import train_fold,_run_training_epoch


class PackedFeatures:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as a:self.values={k:a[k].copy() for k in a.files}
        self.ligands=tuple(self.values['ligand_keys'].astype(str))
        self.candidates=tuple(self.values['candidate_hashes'].astype(str))
        for ids in (self.ligands,self.candidates):
            if not ids or len(ids)!=len(set(ids)) or any(not v for v in ids):raise ValueError('Feature identities must be unique')
        for name,size,rows in [('molformer',768,len(self.ligands)),('ecfp',2048,len(self.ligands)),
                               ('ion',10,len(self.ligands)),('esm',1280,len(self.candidates))]:
            values=self.values[name]
            if values.shape!=(rows,size) or values.dtype!=np.float32 or not np.isfinite(values).all():
                raise ValueError(f'{name} features must be finite float32 of shape {(rows,size)}')
        if self.values['is_ion'].dtype!=np.bool_ or self.values['is_ion'].shape!=(len(self.ligands),):
            raise ValueError('Ion mask must be aligned boolean values')
        self.li={k:i for i,k in enumerate(self.ligands)};self.pi={k:i for i,k in enumerate(self.candidates)}

    def ligand_batch(self,keys):
        indexes=[self.li[k] for k in keys]
        return tuple(torch.from_numpy(self.values[k][indexes].copy()) for k in ('molformer','ecfp','ion','is_ion'))

    def protein_batch(self,keys):
        return torch.from_numpy(self.values['esm'][[self.pi[k] for k in keys]].copy())

    def audit(self):
        return {'format':'ligand2tf-feature-pack-1','sha256':sha256(self.path),
                'ligands':len(self.ligands),'candidates':len(self.candidates),'esm_dim':1280}


def set_seeds(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)


def read_edges(path, *, require_split_unit=True):
    frame=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False)
    required={'ligand_key','sequence_md5'}
    if require_split_unit:required.add('split_unit')
    if not required<=set(frame.columns):
        raise ValueError(f'Edges require columns: {sorted(required)}')
    if frame.empty or frame[['ligand_key','sequence_md5']].duplicated().any():
        raise ValueError('Edges must be nonempty and exact-pair deduplicated')
    return frame


def train_dstar(train_path,val_path,feature_path,output,*,seed,device='cpu',smoke=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if seed not in PROTOCOL['seeds']:raise ValueError('Seed is outside the frozen protocol')
    train=read_edges(train_path);val=read_edges(val_path);store=PackedFeatures(feature_path)
    pairs=lambda frame:set(frame[['ligand_key','sequence_md5']].itertuples(index=False,name=None))
    if pairs(train)&pairs(val):raise ValueError('Training/validation response overlap')
    settings=PROTOCOL['training'];set_seeds(seed)
    episodes=build_training_episodes(train,store.candidates,unknown_per_query=settings['unknown_per_query'],seed=seed)
    queries=build_evaluation_queries(train,val,store.candidates)
    model=make_dual_encoder()
    provenance={'protocol':PROTOCOL['schema'],'seed':seed,'smoke':smoke,
                'train_sha256':sha256(train_path),'val_sha256':sha256(val_path),'feature_sha256':sha256(feature_path)}
    selected=train_fold(model,episodes,queries,store,output/'validation',device=device,
                        max_epochs=1 if smoke else settings['max_epochs'],validation_interval=1 if smoke else settings['validation_interval'],
                        patience=settings['patience'],query_batch_size=settings['query_batch_size'],
                        unlabeled_per_query=settings['unknown_per_query'],unlabeled_weight=settings['unlabeled_weight'],
                        candidate_batch_size=settings['candidate_batch_size'],seed=seed,config=provenance)
    # train_fold restores the selected validation state before returning.
    np.savez_compressed(output/'validation_weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    provenance['validation_weights_sha256']=sha256(output/'validation_weights.npz')
    # Preserve the formal reference's RNG order: reset BEFORE construction,
    # then do not reset the torch RNG between construction and refit epochs.
    epoch=int(selected['selected_epoch']);fit=pd.concat([train,val],ignore_index=True)
    set_seeds(seed)
    episodes=build_training_episodes(fit,store.candidates,unknown_per_query=settings['unknown_per_query'],seed=seed)
    model=make_dual_encoder().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    generator=random.Random(seed)
    for step in range(1,epoch+1):
        _run_training_epoch(model,episodes,store,optimizer,device=torch.device(device),
                            query_batch_size=settings['query_batch_size'],unlabeled_weight=settings['unlabeled_weight'],
                            random_generator=generator,epoch=step)
        scheduler.step()
    np.savez_compressed(output/'refit_weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    provenance.update(selected_epoch=epoch,fit_roles=['train','val'],weights_sha256=sha256(output/'refit_weights.npz'))
    (output/'refit.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def score_dstar(feature_path,fit_path,query_path,weights_path,output,*,device='cpu'):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    store=PackedFeatures(feature_path);fit=read_edges(fit_path,require_split_unit=False)
    meta=json.loads(Path(query_path).read_text());queries=tuple(meta['query_ids'])
    if not queries or len(queries)!=len(set(queries)):raise ValueError('Query IDs must be nonempty and unique')
    pairs=set(fit[['ligand_key','sequence_md5']].itertuples(index=False,name=None))
    model=make_dual_encoder().to(device)
    with np.load(weights_path,allow_pickle=False) as a:
        model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files},strict=True)
    model.eval();scores=np.full((len(queries),len(store.candidates)),np.nan,dtype=np.float32)
    with torch.no_grad():
        for i,q in enumerate(queries):
            active=[j for j,p in enumerate(store.candidates) if (q,p) not in pairs]
            if not active:raise ValueError('Query has no active candidates')
            ligand=model.encode_ligand(*(v.to(device) for v in store.ligand_batch([q])))
            for start in range(0,len(active),1024):
                ix=active[start:start+1024]
                protein=model.encode_protein(store.protein_batch([store.candidates[j] for j in ix]).to(device))
                values=model.score(ligand,protein)[0].cpu().numpy()
                if not np.isfinite(values).all():raise ValueError('Nonfinite Dstar prediction')
                scores[i,ix]=values
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,scores=scores,query_ids=queries,candidate_hashes=store.candidates)
