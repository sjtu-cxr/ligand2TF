# V66 reproducibility update

This update preserves the frozen numerical implementation and uses the
research workspace only as a read-only source and verification oracle.

1. Lock Architecture B / ESM2-650M, feature order, seeds and the 15 selected
   gate configurations in versioned release configuration files.
2. Extract the active-library response pipeline with unchanged function
   definitions: protein/chemical evidence, masking, calibration, 32 features.
3. Add portable bundle input, checkpoint inference, ranking and metric CLI;
   retain separate fitting labels and held-out evaluation labels.
4. Add gate selection/refitting entry points and documented dual-encoder
   training/refit configuration. Exercise synthetic training independently.
5. Export private verification bundles from existing artifacts; reconstruct
   channels/features in this repository and replay every held-out fold. Compare
   full scores/ranks against the reference implementation and aggregate metrics
   against frozen manuscript results. Report inference replay separately from
   retraining and raw encoder feature generation.
6. Verify isolated imports and installation, document data/weight contracts,
   scan staged files, then commit and update the private GitHub repository.

Do not upload unpublished source data, checkpoints, third-party assets or
credentials automatically. Record hashes and acquisition/reconstruction
requirements; no invented download URL, DOI, or license.

Implementation and numerical checks are recorded in RELEASE_STATUS.md.
The private code update does not complete the separate public artifact and
license release. Full formal retraining remains outside this verification run.
