import hashlib
import importlib.util
import json
import numpy as np
import pytest


def bundle(tmp_path, *, labels=True):
    candidates = ['a','b','c']
    np.savez(tmp_path/'protein.npz', candidate_hashes=candidates,
             scores=np.array([[1,.5,.2],[.5,1,.3],[.2,.3,1]], dtype=np.float32))
    (tmp_path/'fit.tsv').write_text('ligand_key\tsequence_md5\nCCO\ta\nCC\tb\n')
    np.savez(tmp_path/'dstar.npz', scores=np.array([[[np.nan,.2,.4]]]*3))
    files={name:{'path':file,'sha256':hashlib.sha256((tmp_path/file).read_bytes()).hexdigest()}
           for name,file in [('fit_edges','fit.tsv'),('protein_similarity','protein.npz'),('dstar_scores','dstar.npz')]}
    data={'schema':'ligand2tf-bundle-1','split':'Random-edge','fold':0,'stage':'test',
          'query_ids':['CCO'],'split_units':['CCO'],'candidate_hashes':candidates,
          'relevant_hashes':[['c']] if labels else [[]], 'seeds':[42,20260717,20260718],
          'beta':0.5,'files':files}
    (tmp_path/'bundle.json').write_text(json.dumps(data))
    return tmp_path/'bundle.json'


def test_bundle_module_exists():
    assert importlib.util.find_spec('src.portable_io') is not None


def test_fold_reconstruction_and_label_free_prediction_agree(tmp_path):
    from src.portable_io import load_bundle
    from src.response_pipeline import build_random_framework_inputs
    path=bundle(tmp_path)
    source,meta=load_bundle(path)
    first=build_random_framework_inputs(source,beta=meta['beta'])
    assert first.eligible_mask.tolist()==[[False,True,True]]
    assert first.features.shape==(1,3,32)
    assert not first.features[0,0].any()
    assert first.labels.tolist()==[[False,False,True]]
    meta['relevant_hashes']=[[]];path.write_text(json.dumps(meta))
    source,_=load_bundle(path)
    second=build_random_framework_inputs(source,beta=meta['beta'])
    np.testing.assert_array_equal(first.features,second.features)
    np.testing.assert_array_equal(first.backbone,second.backbone)


def test_checksum_and_fit_overlap_fail_closed(tmp_path):
    from src.portable_io import load_bundle
    path=bundle(tmp_path)
    meta=json.loads(path.read_text());meta['relevant_hashes']=[['a']]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='fitting'):
        load_bundle(path)
    bundle(tmp_path)
    (tmp_path/'fit.tsv').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        load_bundle(path)
