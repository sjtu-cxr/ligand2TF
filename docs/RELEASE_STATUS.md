# Verification results

## Benchmark prediction checks

Predictions were compared with the manuscript results across all 15 folds
and 1,995 query episodes. Protein-side transfer, ligand-side transfer, the
dual encoder, the backbone and the final ranker had identical scores and
active-candidate ranks. The maximum score difference was 0.0, and all
32-feature tensors matched their reference checksums. H@10, H@50 and MRR
matched within an absolute tolerance of 1e-12.

The full record is in the [benchmark verification report](../reports/benchmark_verification.json).

| Split | Queries | Final H@10 | Final H@50 | Final MRR |
|---|---:|---:|---:|---:|
| Random-edge | 758 | 0.451187 | 0.592348 | 0.288098 |
| TF-50 | 706 | 0.131728 | 0.252125 | 0.070140 |
| Ligand-Morgan-0.5 | 531 | 0.269303 | 0.404896 | 0.161544 |

These checks verify the ranking implementation using previously computed
representation scores and fitted correction heads from the manuscript benchmarks.

## Software and data checks

The test suite covers input validation, response transfer, candidate masking,
training and refitting on synthetic inputs, ranking and evaluation. Benchmark
data tests check sequence identities, file checksums and split consistency.

The package was also installed in an isolated Python 3.10 environment, where
the example, correction training and prediction commands completed successfully.
The tested Linux CPU dependencies are recorded in
[requirements-cpu-lock.txt](../requirements-cpu-lock.txt).

RDKit 2026.3.3 is required for the benchmark chemical similarities; the runtime
checks its version.

## Available materials

The [benchmark directory](../benchmarks/) contains response edges, evidence-source
records, candidate sequences and fixed partitions. Encoder features require
separate preparation, as described in the [input specification](ARTIFACTS.md).

The repository provides the model implementation, benchmark configurations,
and commands for training, prediction and evaluation. The
[training guide](USAGE.md) describes the workflow and required inputs.
