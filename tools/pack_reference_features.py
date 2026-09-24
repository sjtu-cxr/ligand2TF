"""Convert trusted local frozen feature caches to a safe portable NPZ pack.

The input pickle caches must be trusted; this tool is for the original project
owner. Never use it on untrusted downloaded pickle files. It does not generate
new ESM2/MoLFormer embeddings or change feature values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--bundles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trust-local-pickle-caches',action='store_true',required=True)
    args=p.parse_args();root=args.reference_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    sys.path.insert(0,str(root))
    from src.unified_dstar_data import FeatureStore
    cfgpath=root/'data/model_training/v66/results/structure_aware_dstar_ablation_20260828/configs/esm2_650m/Random-edge/fold_0/seed_42.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());features=cfg['features']
    resolved={}
    for key in ['molformer_path','esm_path','ecfp_path','ion_descriptors_path']:
        value=Path(features[key])
        # Resolve original absolute paths by their repository-relative data suffix.
        if value.is_absolute():value=Path('data')/str(value).split('/data/',1)[1]
        resolved[key]=root/value
    store=FeatureStore(**resolved,esm_dim=1280,missing_ion_policy=features['missing_ion_policy'])
    ligands=set();candidates=None
    for path in sorted(args.bundles.glob('*/fold_*/test/bundle.json')):
        meta=json.loads(path.read_text());ligands.update(meta['query_ids'])
        fit=pd.read_csv(path.parent/meta['files']['fit_edges']['path'],sep='\t')
        ligands.update(fit.ligand_key.astype(str))
        if candidates is None:candidates=meta['candidate_hashes']
        if candidates!=meta['candidate_hashes']:raise ValueError('Candidate order differs across bundles')
    if candidates is None:raise ValueError('No test bundles available')
    ligands=sorted(ligands);mol,ecfp,ions,is_ion=store.ligand_batch(ligands)
    esm=store.protein_batch(candidates)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,ligand_keys=ligands,candidate_hashes=candidates,
                        molformer=mol.numpy(),ecfp=ecfp.numpy(),ion=ions.numpy(),
                        is_ion=is_ion.numpy(),esm=esm.numpy())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance={'source_sha256':{key:digest(path) for key,path in resolved.items()},
                'output_sha256':digest(output),'ligands':len(ligands),'candidates':len(candidates),
                'warning':'Trusted cache conversion only; not raw encoder feature generation.'}
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))

if __name__=='__main__':main()
