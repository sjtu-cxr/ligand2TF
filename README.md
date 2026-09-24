# ligand2TF

Prioritize ligand-responsive prokaryotic transcription factors by combining
protein- and ligand-side response evidence with molecular representations.

This repository implements the V66 candidate-level gated model used in the
manuscript. It reconstructs fold-local response neighborhoods, masks known
fitting responders, builds the availability-aware backbone and 32 correction
features, and ranks the active protein library. The dual encoder uses
Architecture B with frozen ESM2-650M and MoLFormer features.

The repository remains **private** while the external data/weight archive and
source license are prepared. Cloning it does not download V66 data or weights.

## Install and try

Python 3.10 is required. Install and run a small synthetic workflow:

```bash
python -m pip install .
ligand2tf example --output data/example
ligand2tf train-gate --bundle data/example/validation/bundle.json --output data/example/head --smoke
ligand2tf predict --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/ranks.tsv
```

The example checks execution, not biological prediction quality. Omit --smoke
for formal 30-epoch gate training. **RDKit 2026.3.3 is locked:** older versions
can change Morgan similarities and final rankings. Use the release factory
and configuration, not historical defaults of the low-level model classes.

## Workflows

- prepare-protein: build the fixed-order identity/coverage cache from MMseqs.
- train-dstar: select on validation data, then refit on train + validation.
- score-dstar: generate candidate scores from explicit features and weights.
- bundle: align three seed-score archives with a fitting response graph.
- train-gate: grouped out-of-fold head selection and final head fitting.
- predict / evaluate: rank candidates or compute query-level metrics.
- verify: compare migrated features, scores and ranks with frozen references.

See [usage](docs/USAGE.md) for commands and the distinction between
training-only validation weights and refitted test weights. See
[artifact contracts](docs/ARTIFACTS.md) for schemas, provenance, and
local conversion of existing trusted feature caches.

## Verification scope

Unit tests and synthetic workflows are separate from frozen V66 replay.
The latter reconstructs response evidence and features from the fitting graph
and fixed molecular similarities, applies original gate weights to cached
Dstar scores, and compares all active candidate ranks. It does **not** retrain
all 45 dual encoders or regenerate ESM2/MoLFormer features. Current results
are in [release status](docs/RELEASE_STATUS.md).

```bash
python -m pip install 'pytest==9.1.1'
python -m pytest tests -q
```

## Layout

src/ contains model and workflow code, configs/ the locked model/fold settings,
tests/ executable checks, and tools/ maintainer-only extraction utilities.
Normal package use does not require the original research project.
Extraction manifests record reference definitions and hashes; source_manifest.json
is an archival record of the initial v0.1 extraction.

## Availability

Repository: [sjtu-cxr/ligand2TF](https://github.com/sjtu-cxr/ligand2TF).
No open-source license or public data DOI is asserted at this stage. Dataset,
third-party encoder and checkpoint redistribution terms need review before
public release. Credentials, research logs and manuscript files are excluded.
