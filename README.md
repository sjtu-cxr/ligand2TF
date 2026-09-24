# Response-Network Organization Enables Prioritization of Ligand-Responsive Prokaryotic Transcription Factors

ligand2TF ranks candidate prokaryotic transcription factors for a query ligand
by integrating known ligand-response relationships with molecular representations.
It is designed for candidate prioritization when functional response evidence is
sparse and unevenly distributed across the candidate protein library.

The repository provides the model implementation, training and evaluation
commands, model configurations, and a runnable synthetic example. Pretrained
ligand2TF weights are not distributed; users train the model with their prepared
data and features.

## Method overview

ligand2TF combines three sources of information:

- **Protein-side response transfer:** sequence similarity to proteins with a
  known response to the query ligand.
- **Ligand-side response transfer:** chemical similarity to ligands with a
  known response for the candidate transcription factor.
- **Molecular representations:** a dual encoder trained on fixed protein and
  ligand features to score candidate ligand–protein pairs.

An evidence-aware backbone combines the available signals, and a learned
candidate-level correction refines the ranking. The representation channel
provides a score even when neither transfer channel has a response witness.
Protein features use ESM2-650M; ligand inputs include MoLFormer embeddings,
Morgan fingerprints and descriptors for supported ions.

For each query, the model ranks eligible proteins in the supplied candidate
library. Known responders in the fitting data are excluded from that ranking.
Predictions prioritize candidates for follow-up; they do not establish ligand
binding or a regulatory mechanism.

## Installation

Use **Python 3.10**. From a local clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For the tested Linux CPU environment, a resolved dependency list is available in
[requirements-cpu-lock.txt](requirements-cpu-lock.txt). Installation options are
described in the [usage guide](docs/USAGE.md#installation).

RDKit **2026.3.3** is required for the chemical-similarity calculations. The
package specifies its dependencies in [pyproject.toml](pyproject.toml).

## Quick start

Run a small synthetic example without downloading biological data or pretrained
ligand2TF weights:

```bash
ligand2tf example --output data/example
ligand2tf train-gate --bundle data/example/validation/bundle.json --output data/example/head --smoke
ligand2tf predict --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/ranks.tsv --top-k 10
ligand2tf evaluate --bundle data/example/test/bundle.json --checkpoint data/example/head --output data/example/metrics.tsv
```

The example supplies synthetic response evidence and representation scores.
It trains the ranking correction for one epoch, writes the top ten candidates
per query, and evaluates their ranks. It demonstrates the interface, not
biological performance or end-to-end dual-encoder training.

The prediction file is a tab-separated table containing the query identifier,
candidate sequence identifier, rank, model score, and indicators of whether
protein-side and ligand-side transfer evidence is available. Lower ranks indicate
higher priority; scores are not calibrated response probabilities.

Output destinations must be new. Choose different paths when repeating the
example. Run `ligand2tf --help` or `ligand2tf <command> --help` for command options.

## Training and prediction with biological data

Prepare ligand-response pairs, a candidate sequence library, protein similarities,
and fixed protein/ligand features. The [input specification](docs/ARTIFACTS.md)
defines the required fields and array dimensions. Features must be supplied
separately; the package does not generate ESM2 or MoLFormer embeddings from raw
sequences or molecules.

The [training and prediction guide](docs/USAGE.md#real-fold-workflow) explains how
to train the dual encoder, generate candidate scores, assemble response-evidence
bundles, and fit the ranking correction. It distinguishes validation-stage
models from models refitted on training and validation data, keeping test
responses out of fitting and model selection.

Prediction requires prepared features and locally trained weights; held-out
response labels are needed only for evaluation. Model settings and benchmark
selection records are provided in [configs/](configs/). Benchmark-specific
selections should not be treated as parameters selected for a new dataset.

## Evaluation and reproducibility

The evaluation protocols assess held-out response edges, held-out protein
sequence components, and held-out ligand chemical components. Performance is
measured by the rank of the highest-ranked known responder, using Hit@10,
Hit@50 and mean reciprocal rank.

The [numerical verification report](reports/benchmark_verification.json) records
agreement with the reference implementation across all 15 benchmark folds.
This checks inference from stored representation scores and trained correction
weights; it is not a from-scratch retraining result. Verification details are
available in the [technical validation record](docs/RELEASE_STATUS.md).

To run the software tests:

```bash
python -m pip install '.[test]'
python -m pytest tests -q
```

## Data and model availability

This repository contains source code, configurations, documentation and tests.
It does not distribute pretrained ligand2TF weights, the biological benchmark
datasets, feature caches or third-party encoder assets. The synthetic example
is available directly through the command line; reproducing the manuscript
benchmarks additionally requires the corresponding data, splits and features.

## Citation and license

A manuscript citation will be added when a citable record is available.
No software license has yet been assigned to this repository.
