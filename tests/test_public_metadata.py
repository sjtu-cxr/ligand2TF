"""Public metadata uses stable names rather than research version labels."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_names():
    marker = 'v' + '66'
    paths = ['configs/model.json', 'benchmarks/manifest.json',
             'src/dstar_workflow.py', 'ligand2tf.py']
    for name in paths:
        assert marker not in (ROOT / name).read_text().lower(), name
    candidates = (ROOT / 'benchmarks/candidates.tsv').read_text()
    assert 'internal_' + marker not in candidates.lower()
    protocol = json.loads((ROOT / 'configs/model.json').read_text())
    assert protocol['schema'] == 'ligand2tf-model-1'
