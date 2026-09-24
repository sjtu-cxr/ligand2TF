import importlib.util


def test_response_pipeline_is_independent_module():
    assert importlib.util.find_spec('src.response_pipeline') is not None


def test_final_model_factory_locks_esm2_650m():
    from src.release_protocol import make_dual_encoder, PROTOCOL
    model = make_dual_encoder()
    assert model.architecture == 'B'
    assert model.esm_dim == 1280
    assert PROTOCOL['seeds'] == [42, 20260717, 20260718]


def test_inference_does_not_require_heldout_labels():
    from src.response_pipeline import _label_matrix
    assert not _label_matrix(((),), ('a','b')).any()


def test_unknown_relevant_candidate_is_rejected():
    import pytest
    from src.response_pipeline import _label_matrix
    with pytest.raises(ValueError, match='absent'):
        _label_matrix((('missing',),), ('a','b'))
