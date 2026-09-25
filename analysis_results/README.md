# Analysis results

These files accompany *Integrating Response-Network Evidence and Molecular Representations to Prioritize Ligand-Responsive Prokaryotic Transcription Factors*.

## Resource and response organization

- `resource_candidate_counts.tsv`: curation counts and candidate-source memberships.
- `figure_s1_source_overlap.pdf`: response-source intersections.
- `global_locality_diagnostics.tsv`: global locality and reference-graph summaries.
- `family_organization_and_recovery.tsv`: family-level locality, response degree and recovery.
- `family_locality_null.tsv`: replicate-level family reference statistics; missing
  values indicate that a replicate had no eligible neighborhood for that statistic.
- `family_associations.tsv`: exploratory associations across the eight jointly
  assessable families.

## Ranking and evaluation

- `benchmark_composition.tsv`: benchmark composition summaries, not the split edge files.
- `retrieval_metrics.tsv`: model and comparator retrieval metrics.
- `random_edge_seeded_baselines.tsv`: Random-edge random-ranking and
  response-degree metrics, including H@5, averaged over 100 seeded rankings.
- `channel_query_metrics.tsv`, `channel_fold_metrics.tsv`: channel-level query and fold results.
- `paired_bootstrap_contrasts.tsv`: paired differences and grouped-bootstrap intervals.
- `random_edge_target_ranks.tsv`: ranks for all 874 held-out response edges.
- `random_edge_route_targets.tsv`, `random_edge_route_recovery.tsv`: target-level
  ranks and summaries by evidence availability.
- `component_support_queries.tsv`, `component_support_recovery.tsv`: query-level
  response support and stratified recovery under component holdout.
- `case_rank_comparison.tsv`: identities, response witnesses, literature references
  and channel ranks for the four main-text examples.

TP and TL denote protein-side and ligand-side transfer; Dstar denotes the
representation ensemble; B and F denote the backbone and final ranker. Ranks
start at one and refer to each query's active candidate library. An unavailable
transfer channel can still have a deterministic fallback rank; that rank is not
response evidence (Supplementary Table S1).

Primary retrieval metrics use the best held-out responder per query. Family and
route analyses use individual held-out response edges; their denominators differ.

## Protein representation comparison

`protein_encoder_metrics.tsv` reports the six frozen protein-representation
configurations within the dual encoder across all three evaluation regimes.
Table S8 presents the Random-edge subset; component-holdout results remain in
these files.
`protein_encoder_fold_metrics.tsv` provides fold-level ensemble metrics;
`protein_encoder_seed_metrics.tsv` provides individual-seed metrics pooled across
folds. Ensemble scores were averaged before ranking, so ensemble metrics are not
the arithmetic mean of seed-level metrics. These results concern the representation
channel, not the final integrated ranker. Encoder identifiers distinguish ProstT5
AA2fold, sequence-only ProSST, and ProSST supplied with ESMFold structure tokens.

## Configurations and provenance

`configurations/` contains fold-specific model-selection records.
`joint_knn_selected_configurations.tsv` and `extratrees_configurations.tsv` contain
comparator settings. `manifest.json` provides SHA-256 checksums for the files in
this directory, with paths relative to `analysis_results/`. Training inputs,
candidate sequences, and evaluation partitions are available in
[`benchmarks/`](../benchmarks/).

## Exploratory component-support diagnostics

`component_support_queries.tsv` and `component_support_recovery.tsv` retain the exploratory support-stratified results. For protein-component holdout, target support is the maximum identity-by-coverage to a fitting-graph responder for the same ligand; for chemical-component holdout, it is the maximum Morgan similarity to a fitting-graph ligand of the target protein. Query support is the maximum across documented test responders. Unanchored queries have no finite target-specific witness and are distinct from low-similarity anchored queries. These strata use test-responder identities and describe recovery retrospectively, not prospective confidence. Intervals use 2,000 grouped-bootstrap resamples with the primary evaluation grouping units.
