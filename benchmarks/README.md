# Benchmark data

The benchmark contains 874 documented ligand–TF responses, 552 ligand identities,
476 response-positive protein sequences and a library of 6,457 candidate proteins.
Files use tab-separated UTF-8 text. These are the fixed inputs underlying the
manuscript benchmarks, not a new curation or a new split assignment.

## Files

- `responses.tsv`: one row per ligand–sequence response, with identifiers, names,
  molecular identities, regulator family, evidence category and source memberships.
- `response_evidence.tsv`: evidence links for the response edges, with source-record
  identifiers, PMID/DOI where recorded, assay types and support flags. Multiple
  source records may describe the same biological relationship; evidence-link
  counts are not counts of independent experiments.
- `candidates.tsv`: candidate identifiers, exact amino-acid sequences, accessions
  and database memberships, in the original candidate order.
- `splits/<benchmark>/fold_<0–4>/{train,val,test}.tsv`: the fixed training,
  validation and test responses for each fold.
- `manifest.json`: file checksums and source-snapshot checksums.

Source-snapshot checksums use descriptive identifiers rather than local file
paths. Candidate-source labels beginning with `response_resource` identify
records contributed by this study's curated response resource.

`sequence_md5` is the MD5 of the exact protein sequence and joins all tables.
`ligand_key` is the model identifier; explicit non-SMILES ions use an `ion:` key.
Do not replace identifiers with display names or silently standardize the
structures again. The resource includes one unparseable structure, retained in
edge/protein benchmarks with no structural transfer witness.

## Split files

Each split file contains `edge_id`, `ligand_key`, `sequence_md5`, `split_unit`
and `split_role`. Random-edge and TF-50 each cover 874 responses; the
Ligand-Morgan-0.5 benchmark covers 788 structure-eligible responses. Every edge
in a benchmark appears in one outer test fold. The three roles are disjoint
within a fold; component split units are retained exactly.

For example, the training entry point accepts:

```bash
ligand2tf train-dstar --train benchmarks/splits/Random-edge/fold_0/train.tsv --validation benchmarks/splits/Random-edge/fold_0/val.tsv --features features.npz --seed 42 --output models/fold_0_seed_42
```

`features.npz` must be prepared separately using the encoders and input schema
in [the input specification](../docs/ARTIFACTS.md). The benchmark tables alone
are not the complete feature input to training.

## Sources and reuse

Response-source memberships and evidence identifiers are retained in the two
response tables. Candidate database memberships and record identifiers are
retained in `candidates.tsv`. Consult those source records and the manuscript
references when interpreting or reusing the data. Sequence identity defines the
modeling unit; it does not remove uncertainty in the original evidence-to-sequence
assignment or make all records direct-binding measurements.

The repository's MIT license covers the original software and documentation.
It does not replace the terms of third-party database content or encoder assets.
Source attribution must be retained when reusing the benchmark.
