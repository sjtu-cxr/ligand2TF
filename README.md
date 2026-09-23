# ligand2TF

Source code for prioritizing ligand-responsive prokaryotic transcription
factors by combining response evidence with molecular representations.

## Release status

This is a **private source-release candidate** for the V66 main model. The
core numerical modules and self-contained tests have been extracted from
the research workspace. The full benchmark command-line workflow and its
data/checkpoint distribution are not yet packaged. This repository must not
yet be described as a complete reproduction release or a pretrained predictor.

GitHub repository: [sjtu-cxr/ligand2TF](https://github.com/sjtu-cxr/ligand2TF).
The repository is initially private while the release package is completed.

## Model components

- `src/unified_chemical_transfer.py`: chemical response transfer.
- `src/benchmark_agnostic_transfer.py` and
  `src/unified_transfer_first_routing.py`: legal-witness availability and
  transfer-first routing, with representation fallback.
- `src/unified_dstar_model.py`: dual-encoder architectures.
- `src/unified_dstar_data.py` and `src/unified_dstar_training.py`: feature and
  split contracts, training, checkpoint selection, and full-library evaluation.
- `src/unified_bscd_residual.py`: the 32 candidate-level correction features.
- `src/candidate_gated_top10.py`: bounded gated correction, weighted ranking
  loss, and regularization.
- `src/gate_training.py`: portable gate fitting, full-library scoring,
  constrained configuration selection, grouped out-of-fold predictions,
  three-seed rank aggregation, and query-level metrics.
- `src/fit_candidate_mask.py`: fit-response masking and ranking metrics.
- `src/unified_dstar_ensemble.py`: deterministic candidate ranking and score
  archives; equal scores are ordered by candidate hash.

The historical `src` module namespace is retained to avoid changing numerical
implementation during extraction. Some modules contain reusable historical
helpers; their presence does not identify them as part of the final model.
The audit module under `scripts/pipeline` is an import dependency, not an
end-to-end release entry point.

Gate functions consume fold-local arrays: features of shape
`(n_queries, n_candidates, 32)`, backbone scores and boolean eligibility masks
of shape `(n_queries, n_candidates)`, and boolean labels for training only.
Candidate hashes and query IDs must align with their corresponding axes.
See `tests/test_gate_training.py` for a runnable synthetic fit/score example:

```bash
python -m pytest tests/test_gate_training.py -q
```

The synthetic example checks interfaces and numerical validity, not biological
prediction quality. Prediction does not require labels. Never include held-out
responses when constructing training evidence or selecting configurations.

## Run the core tests

Use Python 3.10 and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

The version list records the environment used during extraction. A fresh
dependency installation on other platforms is not yet verified. Tests use
synthetic inputs; passing them does not establish reproduction of manuscript
metrics. See [release status](docs/RELEASE_STATUS.md) for remaining work.

## Data, weights, and rights

No response dataset, sequence collection, third-party encoder weights, trained
checkpoint, manuscript, cluster log, or authentication file is included.
The source manifest records checksums of the original files. Publication
licensing and data/weight redistribution terms require author review before
public release; no open-source license has been applied at this stage.
