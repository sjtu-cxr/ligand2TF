import importlib.util


def test_portable_gate_training_module_is_available():
    assert importlib.util.find_spec('src.gate_training') is not None
