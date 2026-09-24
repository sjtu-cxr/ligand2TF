import pytest


def test_fifteen_arbitrary_folds_cannot_claim_complete(tmp_path, monkeypatch):
    from src import release_verification as module
    for fold in range(15):
        path = tmp_path / 'Random-edge' / f'fold_{fold}' / 'test' / 'bundle.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    (tmp_path / 'expected_metrics.tsv').write_text('split\tmodel\tH10\tH50\tMRR\n')
    monkeypatch.setattr(module, 'verify_fold', lambda path: {
        'split': 'Random-edge', 'fold': int(path.parent.parent.name[5:])})
    with pytest.raises(ValueError, match='exact.*15'):
        module.verify_all(tmp_path)
