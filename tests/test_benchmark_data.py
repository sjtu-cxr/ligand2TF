"""Integrity checks for the distributed benchmark, without model training."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks'


def test_benchmark_identities_and_checksums():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    candidates = pd.read_csv(ROOT / 'candidates.tsv', sep='\t')
    edges = pd.read_csv(ROOT / 'responses.tsv', sep='\t', keep_default_na=False)
    assert len(candidates) == candidates.sequence_md5.nunique() == 6457
    assert len(edges) == edges.edge_id.nunique() == 874
    assert edges.ligand_key.nunique() == 552
    assert edges.sequence_md5.nunique() == 476
    assert set(edges.sequence_md5) <= set(candidates.sequence_md5)
    for row in candidates.itertuples():
        assert hashlib.md5(row.protein_sequence.encode()).hexdigest() == row.sequence_md5


def test_fixed_split_coverage_and_disjointness():
    edges = pd.read_csv(ROOT / 'responses.tsv', sep='\t', keep_default_na=False)
    identities = edges.set_index('edge_id')[['ligand_key', 'sequence_md5']]
    for split, expected in [('Random-edge', 874), ('TF-50', 874), ('Ligand-Morgan-0.5', 788)]:
        tests = []
        for fold in range(5):
            frames = [pd.read_csv(ROOT / 'splits' / split / f'fold_{fold}' / f'{role}.tsv', sep='\t', keep_default_na=False)
                      for role in ('train', 'val', 'test')]
            whole = pd.concat(frames)
            assert len(whole) == whole.edge_id.nunique() == expected
            assert whole.set_index('edge_id')[identities.columns].equals(identities.loc[whole.edge_id])
            for i in range(3):
                for j in range(i):
                    assert not set(frames[i].split_unit) & set(frames[j].split_unit)
            tests.extend(frames[2].edge_id)
        assert len(tests) == len(set(tests)) == expected
