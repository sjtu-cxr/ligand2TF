from dataclasses import replace
import importlib.util
import numpy as np
import pytest
from test_portable_io import bundle


def test_runtime_exists():
    assert importlib.util.find_spec('src.release_runtime') is not None


def test_fallback_checkpoint_and_label_free_scoring(tmp_path):
    from src.portable_io import load_bundle
    from src.response_pipeline import build_random_framework_inputs
    from src.release_runtime import score_gate,fit_gate
    from src.unified_bscd_residual import FEATURE_NAMES
    source,meta=load_bundle(bundle(tmp_path))
    inputs=build_random_framework_inputs(source,beta=meta['beta'])
    payload={'split':'Random-edge','fold':0,'beta':.5,'selected_config':None,
             'feature_names':list(FEATURE_NAMES),'training_seeds':[42,20260717,20260718], 'state_dicts':[]}
    scored=score_gate(inputs,payload,beta=.5)
    assert scored.shape==(1,3) and scored[0,0]<-1e8
    with pytest.raises(ValueError,match='validation'):
        fit_gate(inputs,beta=.5,epochs=1)
    with pytest.raises(ValueError,match='identity'):
        score_gate(inputs,dict(payload,fold=1),beta=.5)


def test_checkpoint_roundtrip(tmp_path):
    from src.portable_io import save_gate,load_gate
    from src.unified_bscd_residual import FEATURE_NAMES
    p={'split':'Random-edge','fold':0,'beta':.5,'selected_config':None,
       'feature_names':list(FEATURE_NAMES),'training_seeds':[42,20260717,20260718],'state_dicts':[]}
    save_gate(tmp_path/'gate',p)
    assert load_gate(tmp_path/'gate')['state_dicts']==[]
    with pytest.raises(FileExistsError):save_gate(tmp_path/'gate',p)
