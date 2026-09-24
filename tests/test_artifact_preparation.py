import numpy as np
import pytest


def test_mmseqs_cache_preserves_identity_coverage(tmp_path):
    from src.artifact_preparation import prepare_protein
    (tmp_path/'candidates.tsv').write_text('sequence_md5\tprotein_sequence\na\tAAAA\nb\tAAAAAA\n')
    (tmp_path/'hits.m8').write_text('a\tb\t50\t3\t0\t0\t1\t3\t1\t3\t0\t10\n')
    prepare_protein(tmp_path/'candidates.tsv',tmp_path/'hits.m8',tmp_path/'out.npz')
    with np.load(tmp_path/'out.npz') as a:
        assert a['scores'].dtype==np.float32
        assert a['scores'][0,1]==.25
        assert a['scores'][1,0]==0
        assert np.diag(a['scores']).tolist()==[1,1]
