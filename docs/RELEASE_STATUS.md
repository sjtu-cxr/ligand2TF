# Release checklist

## Included in the source candidate

- Core dual-encoder, transfer-routing, correction, masking and evaluation code.
- Seven self-contained test modules copied from the research workspace,
  plus gate-training tests with only the import path adapted.
- Portable gate-training, selection, grouped OOF, seed aggregation, prediction
  and metric functions extracted with unchanged numerical definitions.
- Source-file provenance checksums; the copied audit module's external
  trusted-workspace path is removed, with no numerical formula changes.

## Required before a complete manuscript release

- Extract protein similarity construction and fold-local response witnesses.
- Package final ESM2-650M training configurations and feature preparation.
  The dual-encoder class's historical default ESM dimension is not the final
  model's feature configuration; callers must use the actual encoder dimension.
- Connect the extracted gate primitives to portable evidence construction and
  train-plus-validation refit orchestration.
- Provide portable train/rank/evaluate commands and a runnable example.
- Supply V66 split manifests, candidate ordering, artifact checksums, and
  download or reconstruction instructions after redistribution review.
- Replay frozen score inputs and compare all three benchmarks with the
  manuscript, independently of the original workspace.
- Test a clean environment installation; scan the final staged tree.
- Choose a source license before public release.

No dataset or model training has been rerun merely by assembling this candidate.

## Verification on 2026-09-23

From the independent repository, with PYTHONPATH unset:

    python -m pytest tests -q -ra

Result: 128 passed, 3 skipped. The skips require the real TF-50 and
ligand-component datasets/features, or CUDA. No real-data replay or GPU
training is claimed. The environment was an existing Python 3.10 environment,
not a newly installed virtual environment.

All copied numerical source modules match their recorded source hashes.
The only change to a copied source file removes the external trusted root
from the audit module. Extracted training functions/classes were checked by
AST comparison against their original definitions. A scan of the 28 assembled
files found no GitHub-token/private-key patterns, absolute research-workspace
paths, symlinks, or files over 10 MB; this is not a comprehensive security audit.

The author completed GitHub device authorization as sjtu-cxr. The private
repository sjtu-cxr/ligand2TF has been created for this source candidate.
Git uses the account's ID-based GitHub noreply address for commit attribution.
Authentication files are stored outside this repository and are not included.
