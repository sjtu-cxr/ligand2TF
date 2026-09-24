"""Explicit final-model factory; never rely on historical class defaults."""
import json
import importlib.metadata
from pathlib import Path
from src.unified_dstar_model import UnifiedDStar
from src.unified_bscd_residual import FEATURE_NAMES

PROTOCOL = json.loads((Path(__file__).resolve().parents[1]/'configs/v66.json').read_text())
if PROTOCOL['feature_names'] != list(FEATURE_NAMES):
    raise ValueError('Frozen feature order does not match implementation')
if PROTOCOL['architecture'] != 'B' or PROTOCOL['model']['esm_dim'] != 1280:
    raise ValueError('This release requires Architecture B / ESM2-650M')


def make_dual_encoder():
    return UnifiedDStar(architecture=PROTOCOL['architecture'], **PROTOCOL['model'])


def assert_runtime_compatibility():
    versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','torch','rdkit')}
    if versions['rdkit']!=PROTOCOL['rdkit_version']:
        raise RuntimeError('V66 response replay requires RDKit '+PROTOCOL['rdkit_version']+
                           '; other versions can change Morgan similarities and rankings')
    return versions
