import importlib.util
import numpy as np
import pytest


def test_dstar_workflow_exists():
    assert importlib.util.find_spec('src.dstar_workflow') is not None


def test_feature_pack_validates_final_dimensions(tmp_path):
    from src.dstar_workflow import PackedFeatures
    path=tmp_path/'features.npz'
    np.savez(path,ligand_keys=['CC'],candidate_hashes=['a'],molformer=np.zeros((1,768),np.float32),
             ecfp=np.zeros((1,2048),np.float32),ion=np.zeros((1,10),np.float32),
             is_ion=np.array([False]),esm=np.zeros((1,640),np.float32))
    with pytest.raises(ValueError,match='esm'):
        PackedFeatures(path)


def test_validation_then_refit_and_label_free_scoring(tmp_path):
    import json
    import torch
    from src.dstar_workflow import train_dstar,score_dstar
    torch.set_num_threads(1)
    rng=np.random.default_rng(3);candidates=[f'p{i:03d}' for i in range(140)]
    feature=tmp_path/'features.npz'
    np.savez(feature,ligand_keys=['CC','CCO'],candidate_hashes=candidates,
             molformer=rng.normal(size=(2,768)).astype('float32'),ecfp=np.zeros((2,2048),'float32'),
             ion=np.zeros((2,10),'float32'),is_ion=np.array([False,False]),
             esm=rng.normal(size=(140,1280)).astype('float32'))
    header='ligand_key\tsequence_md5\tsplit_unit\n'
    (tmp_path/'train.tsv').write_text(header+'CC\tp000\tCC\nCCO\tp001\tCCO\n')
    (tmp_path/'val.tsv').write_text(header+'CC\tp002\tCC\nCCO\tp003\tCCO\n')
    result=train_dstar(tmp_path/'train.tsv',tmp_path/'val.tsv',feature,tmp_path/'model',seed=42,smoke=True)
    assert result['selected_epoch']==1 and result['smoke']
    assert (tmp_path/'model/validation_weights.npz').is_file()
    (tmp_path/'fit.tsv').write_text('ligand_key\tsequence_md5\nCC\tp000\nCCO\tp001\nCC\tp002\nCCO\tp003\n')
    (tmp_path/'queries.json').write_text(json.dumps({'query_ids':['CC']}))
    score_dstar(feature,tmp_path/'fit.tsv',tmp_path/'queries.json',tmp_path/'model/refit_weights.npz',tmp_path/'scores.npz')
    with np.load(tmp_path/'scores.npz') as a:
        assert a['scores'].shape==(1,140)
        assert np.isnan(a['scores'][0,[0,2]]).all()
        assert np.isfinite(a['scores'][0,4:]).all()
