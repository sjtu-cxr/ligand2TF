# V66 implementation release status

## Implemented in v0.2

- Locked Architecture B / ESM2-650M settings, 32-feature order, three seeds,
  45 selected training epochs and 15 frozen gate selections.
- Portable fold-local protein/chemical response evidence, active-candidate
  masking, Dstar calibration, backbone and candidate-gate correction.
- Explicit dual-encoder validation selection and train-plus-validation refit,
  separate validation/refit weight exports, and candidate-library scoring.
- Grouped validation-only gate selection/refitting; ranking and evaluation CLI.
- Safe numeric NPZ input contracts, checksums, package installation and a
  self-contained synthetic example. Normal use needs no research-workspace import.

The initial v0.1 extraction was not a complete executable release. Its broad
research audit module has been replaced with three unchanged grouped-validation
helpers; the original remains in Git history and in the untouched research tree.

## Verification on 2026-09-24

Fresh source test run, with PYTHONPATH unset: **146 passed, 3 skipped**. The
skips are two legacy tests that expect original dataset paths and one CUDA test.
New portable real-data replay is checked separately, not counted as a skipped
legacy test. Synthetic tests exercise training/refitting and scoring, not
biological accuracy or convergence of all formal training runs.

A wheel was installed in a fresh Python 3.10 virtual environment without
inherited packages. From outside the repository, the installed CLI completed
example generation, gate training and prediction. Import locations were checked
inside that environment. Its Linux CPU dependency set is recorded in
requirements-cpu-lock.txt; GPU reproducibility has not been certified.

Source comparison confirmed 20 extracted response-pipeline definitions and
16 gate-training definitions are unchanged, along with source/output hashes.
The portable feature converter successfully loaded the existing trusted caches
and produced a non-pickled pack for 552 ligand keys and 6,457 candidates.
This is cache conversion, not new encoder feature generation.

### Environment correction

The initial RDKit 2022.9.5 requirement did not reproduce frozen chemical
similarities: three queries in Random-edge fold 0 differed. The reference
environment uses RDKit 2026.3.3. Pinning that version restored the exact feature
hash, channel rankings and published target ranks in that fold. Runtime checks
now reject a different RDKit version rather than silently change predictions.
No model formula or comparison tolerance was changed to conceal this mismatch.

### Full frozen V66 replay

All **15 folds / 1,995 query episodes** passed. For TP (S), TL (C), Dstar, B
and F, the maximum score difference was **0.0**, all active-candidate ranks
matched, and every reconstructed 32-feature tensor matched its reference hash.
Final target ranks also matched the previously frozen per-query result table.
All five channels' H@10, H@50 and MRR matched the manuscript metric archive
within absolute tolerance 1e-12. The machine-readable record is
[benchmark verification report](../reports/benchmark_verification.json).

| Split | Queries | Final H@10 | Final H@50 | Final MRR |
|---|---:|---:|---:|---:|
| Random-edge | 758 | 0.451187 | 0.592348 | 0.288098 |
| TF-50 | 706 | 0.131728 | 0.252125 | 0.070140 |
| Ligand-Morgan-0.5 | 531 | 0.269303 | 0.404896 | 0.161544 |

These are inference migration checks using original cached Dstar scores and
gate weights, not newly trained benchmark results. Existing invalid chemical
keys retain the reference implementation's no-structural-witness behavior;
the migration does not silently repair or relabel the dataset.

## Boundaries before public release

- Frozen-score replay does not establish independent biological validation,
  regenerate ESM2/MoLFormer features or retrain all 45 dual encoders.
- Source data, split manifests, feature caches and checkpoints remain local;
  a reader cannot reproduce manuscript metrics from Git alone yet.
- Pretrained weights will not be distributed. Public benchmark data access,
  data redistribution review and a source license remain unresolved; users
  must train their own weights. No download URL or license has been invented.
- The GitHub repository remains private. Authentication files and private
  verification inputs are excluded from version control.
