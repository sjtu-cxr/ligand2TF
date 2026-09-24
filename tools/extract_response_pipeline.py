"""Developer utility: extract reviewed numerical definitions from a reference tree."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import yaml


def segments(path, names):
    text = path.read_text(); lines = text.splitlines(keepends=True); found = {}
    for node in ast.parse(text).body:
        name = getattr(node, 'name', None)
        if name in names:
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            found[name] = ''.join(lines[first-1:node.end_lineno])
    if set(found) != set(names):
        raise ValueError(f'Missing definitions in {path.name}: {set(names)-set(found)}')
    return '\n\n'.join(found[n] for n in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args(); root = args.reference_root.resolve()
    dest = Path(__file__).resolve().parents[1]
    source = root/'scripts/pipeline/build_random_framework_three_split_inputs.py'
    names = ['QueryEvidence','CompactSequenceSimilarity','FoldEvidenceSource',
             'DStarSeedOverride','apply_dstar_seed_override','ActiveRandomFrameworkRow',
             'RandomFrameworkInputs','_edge_pairs','_similarity_value','_chemical_tables',
             'build_query_evidence','_vector','_boolean_vector','_calibrated_dstar',
             'build_active_row','protein_transfer_raw','_validate_structure_weight',
             '_align_structure_similarity','build_random_framework_inputs']
    sources = [(source,names),
               (root/'scripts/pipeline/build_unified_random_edge_full_inputs.py',['_pure_transfer_scores'])]
    header = '''"""Frozen V66 response reconstruction; extracted numerical functions.

The sole adapted I/O hook loads an explicit candidate-aligned similarity cache.
No original project imports or implicit data locations are used.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark_agnostic_transfer import transfer_first_auto, witness_availability
from src.transfer_distilled_residual import _robust_z, calibrated_dstar_fallback
from src.structure_response_transfer import CompactStructureSimilarity, combine_transfer_scores
from src.unified_bscd_residual import FEATURE_NAMES, build_bscd_residual_features
from src.unified_chemical_transfer import FINGERPRINT_NAMES, chemical_transfer_scores
from src.unified_random_edge_full_baselines import candidate_hash_tie_scores

def _label_matrix(relevant, candidates):
    # Labels are optional for prediction; never used to construct evidence.
    index = {value: i for i, value in enumerate(candidates)}
    labels = np.zeros((len(relevant), len(candidates)), dtype=bool)
    for row, values in enumerate(relevant):
        if any(value not in index for value in values):
            raise ValueError('Relevant candidate absent from candidate library')
        labels[row, [index[value] for value in values]] = True
    return labels

def _load_sequence_similarity(source):
    with np.load(source.sequence_similarity_path, allow_pickle=False) as archive:
        candidates = tuple(archive['candidate_hashes'].astype(str))
        scores = archive['scores'].copy()
    if candidates != source.candidate_hashes:
        raise ValueError('Protein similarity cache candidate order mismatch')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Protein similarities must be finite and in [0, 1]')
    return CompactSequenceSimilarity(candidates, scores)

'''
    out = dest/'src/response_pipeline.py'
    out.write_text((header+'\n\n'.join(segments(p,n) for p,n in sources)).rstrip()+'\n')
    manifest = {'output': str(out.relative_to(dest)),
                'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                'inputs': [{'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                            'definitions':n} for p,n in sources]}
    for name in ['structure_response_transfer','unified_random_edge_full_baselines']:
        p=root/f'src/{name}.py'; shutil.copyfile(p,dest/f'src/{name}.py')
        manifest['inputs'].append({'path':f'src/{name}.py','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'definitions':'whole file'})
    (dest/'response_extraction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    group_source=root/'scripts/pipeline/audit_transfer_distilled_stage_a_inputs.py'
    group_names=['_require','atomic_split_units','grouped_meta_fold']
    group_header='"""Frozen atomic-group validation partitioning."""\nfrom __future__ import annotations\nfrom collections import defaultdict\nfrom collections.abc import Sequence\nfrom hashlib import sha256\nimport numpy as np\n\n'
    group_out=dest/'src/grouped_validation.py'
    group_out.write_text(group_header+segments(group_source,group_names).rstrip()+'\n')
    manifest['grouped_validation']={'source':str(group_source.relative_to(root)),
                                   'source_sha256':hashlib.sha256(group_source.read_bytes()).hexdigest(),
                                   'definitions':group_names,'output_sha256':hashlib.sha256(group_out.read_bytes()).hexdigest()}
    (dest/'response_extraction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cfgroot=root/'data/model_training/v66/results/structure_aware_dstar_ablation_20260828/configs/esm2_650m'
    configs=list(cfgroot.glob('*/fold_*/seed_*.yaml'))
    if len(configs)!=45: raise ValueError('Expected 45 locked Dstar configs')
    cfg=yaml.safe_load(configs[0].read_text())
    for p in configs:
        other=yaml.safe_load(p.read_text())
        assert other['model']==cfg['model'] and other['training']==cfg['training']
    from src.unified_bscd_residual import FEATURE_NAMES
    protocol={'schema':'ligand2tf-v66-1','architecture':'B','protein_encoder':'esm2_650m',
              'model':cfg['model'],'training':cfg['training'],'seeds':[42,20260717,20260718],
              'feature_names':list(FEATURE_NAMES),'beta_grid':[0,0.25,0.5,0.75,1],
              'gate_epochs':30,'gate_learning_rate':0.01,'rdkit_version':'2026.3.3',
              'reference_configs':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in configs}}
    (dest/'configs').mkdir(exist_ok=True)
    (dest/'configs/v66.json').write_text(json.dumps(protocol,indent=2)+'\n')
    selections=root/'data/model_training/v66/results/efit_refit_three_method_replay_20260830/candidate_gate/validation_selections.jsonl'
    shutil.copyfile(selections,dest/'configs/fold_selections.jsonl')
    epochs=[]
    for path in sorted((root/'data/model_training/v66/results/structure_aware_dstar_ablation_20260828/formal/esm2_650m').glob('*/seed_*/fold_*/fold_manifest.json')):
        item=json.loads(path.read_text())
        epochs.append({k:item[k] for k in ('split','fold','seed','selected_epoch','fit_roles')})
    if len(epochs)!=45:raise ValueError('Expected 45 formal Dstar manifests')
    (dest/'configs/dstar_selected_epochs.json').write_text(json.dumps(epochs,indent=2)+'\n')
    print('Extracted response pipeline; verified all 45 Dstar configurations; copied 15 gate selections.')

if __name__ == '__main__':
    main()
