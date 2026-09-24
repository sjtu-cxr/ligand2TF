# Training and using ligand2TF

## Installation

Use Python 3.10. From a clean environment and the repository root:

```bash
python -m pip install .
python -m pip install 'pytest==9.1.1'
python -m pytest tests -q
```

For the tested Linux x86_64 CPU environment, install the resolved dependency
lock first, then the package without re-resolving dependencies:

```bash
python -m pip install -r requirements-cpu-lock.txt
python -m pip install --no-deps .
```

The CPU lock was checked in a fresh virtual environment without inherited
site packages. CUDA installation and numerical reproducibility on GPUs are
not certified by that CPU installation check.

RDKit **2026.3.3** is part of the numerical protocol, not an optional version
preference. RDKit 2022.9.5 changed chemical-transfer values in three queries
of Random-edge fold 0 in the migration audit. The runtime rejects an incorrect
RDKit version. Do not rely on defaults of the historical low-level model class;
the release factory loads Architecture B / 1280-dimensional ESM2-650M inputs
from `configs/v66.json`.

## Small runnable example

```bash
ligand2tf example --output data/example
ligand2tf train-gate --bundle data/example/validation/bundle.json --output data/example/head --smoke
ligand2tf predict --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/ranks.tsv
ligand2tf evaluate --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/metrics.tsv
```

This example contains synthetic features and responses. `--smoke` trains for
one epoch and marks its checkpoint as nonformal; omit it for the frozen
30-epoch gate procedure. Example metrics are not scientific results.

## Real fold workflow

The release takes explicit input artifacts rather than searching a research
filesystem. See [artifact contracts](ARTIFACTS.md) for schemas and availability.

1. Prepare fixed normalized `train.tsv`, `val.tsv`, and `test.tsv` and a
   deduplicated candidate library. Never reconstruct splits from case studies
   or select parameters on outer-test responses.
2. Prepare frozen ESM2-650M / MoLFormer embeddings, Morgan fingerprints and ion
   descriptors as `features.npz`. Feature generation is distinct from the
   training of the small dual encoder; the large encoders remain frozen.
3. Run validation selection followed by a fresh refit on train + validation,
   separately for each seed:

```bash
ligand2tf train-dstar --train train.tsv --validation val.tsv --features features.npz --seed 42 --output models/seed_42
```

   Repeat for 20260717 and 20260718. The command never reads test responses.
   It saves validation checkpoints and refit weights. The formal refit resets
   RNGs before model construction and retains their subsequent state, matching
   the frozen reference; a generic refit helper that reseeds after construction
   is not interchangeable. Original selected epochs are recorded for all
   45 model runs in `configs/dstar_selected_epochs.json`.
4. For gate-validation scores, use each seed's **training-only validation
   checkpoint** (`validation_weights.npz`), not refit weights. For outer-test
   scores, use **refit weights** (`refit_weights.npz`).
   Score query JSONs without requiring their held-out labels:

```bash
ligand2tf score-dstar --features features.npz --fit-edges fit.tsv --queries queries.json --weights models/seed_42/refit_weights.npz --output scores_42.npz
```

5. Construct protein similarities from the fixed MMseqs m8 output:

```bash
ligand2tf prepare-protein --candidates candidates.tsv --alignments candidate_vs_candidate.m8 --output protein.npz
```

6. Assemble an explicit stage bundle. The seed-score order is fixed:

```bash
ligand2tf bundle --fit-edges fit.tsv --protein protein.npz --queries queries.json --seed-scores scores_42.npz scores_20260717.npz scores_20260718.npz --split Random-edge --fold 0 --stage test --beta 0.5 --output bundles/test
```

   Use the fold's beta from `configs/fold_selections.jsonl`. This beta is the
   previously selected frozen transfer weight; do not reselect it with refitted
   representations or test labels. For validation-stage bundles, `fit.tsv`
   contains **training responses only**; for test-stage bundles it contains
   **training + validation responses**. The runtime derives exclusions from
   those exact edges and rejects a held-out response that overlaps them.
7. Train the gate only on its validation bundle using grouped three-fold
   out-of-fold selection. Refit the selected head on all validation episodes.
   Apply the retained head to the recomputed outer-test evidence:

```bash
ligand2tf train-gate --bundle bundles/validation/bundle.json --output models/gate
ligand2tf predict --bundle bundles/test/bundle.json --checkpoint models/gate --output predictions.tsv --top-k 10
ligand2tf evaluate --bundle bundles/test/bundle.json --checkpoint models/gate --output metrics.tsv
```

Do not fit the gate on outer-test labels. Its feature scaler/ranks must use
the query-specific active library, after masking fitting responders. Labels
are optional for prediction; they are required for training and evaluation.

## Frozen migration verification

The maintainer exporter requires the original trusted research environment
(which has additional historical dependencies, including LightGBM). It is
not a dependency of installed prediction or training:

```bash
python tools/export_reference_bundles.py --reference-root /path/to/research --output data/verification_v66
ligand2tf verify --bundles data/verification_v66 --output reports/verification.json
```

The exporter runs each fold in a separate process. The verifier compares the
complete 32-feature tensor, TP/TL/D*/B/F scores and active candidate ranks,
published final target ranks, and five-fold query-weighted H@10/H@50/MRR.
Frozen-score replay does not prove that retraining on every hardware/library
combination will reproduce identical weights, nor does it regenerate raw
pretrained-encoder features. These are separately reported verification scopes.
