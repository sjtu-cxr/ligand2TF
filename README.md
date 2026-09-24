# Integrating Response-Network Evidence and Molecular Representations to Prioritize Ligand-Responsive Prokaryotic Transcription Factors

Code accompanying the manuscript of the same title.

The model ranks candidate prokaryotic transcription factors for a query ligand.
It combines sequence similarity to known responders, chemical similarity to
known response ligands, and a dual encoder using fixed protein and ligand
features. A learned correction adjusts the combined ranking according to the
available evidence. Known responders in the fitting data are excluded from the
candidate ranking.

## Installation

Python 3.10 is required. From the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Dependencies, including RDKit 2026.3.3, are specified in
[pyproject.toml](pyproject.toml). A pinned Linux CPU environment is also available
in [requirements-cpu-lock.txt](requirements-cpu-lock.txt); see the
[installation instructions](docs/USAGE.md#installation).

## Example

The following commands generate synthetic inputs, fit the ranking correction,
and write predictions and evaluation results:

```bash
ligand2tf example --output data/example
ligand2tf train-gate --bundle data/example/validation/bundle.json --output data/example/head --smoke
ligand2tf predict --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/ranks.tsv --top-k 10
ligand2tf evaluate --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/metrics.tsv
```

`--smoke` uses one training epoch. The example includes synthetic representation
scores, so it does not train the dual encoder or measure biological performance.

`ranks.tsv` contains the top ten candidates per query, with these columns:

- `query_id`, `candidate_hash`: ligand and candidate sequence identifiers.
- `rank`, `score`: candidate priority and model score. Scores are not response probabilities.
- `protein_transfer_available`, `ligand_transfer_available`: whether each transfer channel has supporting response evidence.

`metrics.tsv` contains per-query evaluation results; aggregate metrics are
printed to the terminal. Commands do not overwrite existing outputs, so use
new output paths when repeating the example.

## Training on biological data

Training requires ligand-response pairs, a candidate sequence library, protein
similarities, and protein/ligand features. The model uses ESM2-650M protein
embeddings, MoLFormer ligand embeddings, Morgan fingerprints and descriptors
for supported ions. Embeddings must be prepared separately.

The [input specification](docs/ARTIFACTS.md) describes file formats and feature
dimensions. Follow the [training guide](docs/USAGE.md#real-fold-workflow) to train
the dual encoder, score candidates, construct response-evidence bundles, and
train the ranking correction. The guide specifies which checkpoints and
response edges to use at validation and test time.

Model settings and benchmark selection records are in [configs/](configs/).
The recorded selections belong to the manuscript benchmarks; parameters for
a new dataset should be selected using its validation data.

For individual command options:

```bash
ligand2tf --help
ligand2tf train-dstar --help
ligand2tf predict --help
```

## Evaluation

The manuscript evaluates held-out response edges, protein sequence components
and ligand chemical components. Hit@10, Hit@50 and mean reciprocal rank are
computed from the highest-ranked known responder for each query.

To run the software tests:

```bash
python -m pip install '.[test]'
python -m pytest tests -q
```

## Availability

The response dataset, evidence-source records, candidate sequences and fixed
train/validation/test partitions are provided in [benchmarks/](benchmarks/).
Its README describes the fields and source attribution. Encoder features must
be prepared separately, as described in the training instructions.

The original software is released under the [MIT License](LICENSE). Third-party
data and encoder assets retain their respective terms.
