import importlib.util
from pathlib import Path


def test_grouping_is_a_core_module_not_a_research_script_dependency():
    assert importlib.util.find_spec('src.grouped_validation') is not None
    root=Path(__file__).resolve().parents[1]
    for file in (root/'src').glob('*.py'):
        assert 'from scripts.pipeline' not in file.read_text(),file.name


def test_connected_units_never_cross_meta_folds():
    from src.grouped_validation import grouped_meta_fold
    result=grouped_meta_fold(['a;b','b;c','d','e','f'])
    assert result[0]==result[1]
    assert set(result)=={0,1,2}
