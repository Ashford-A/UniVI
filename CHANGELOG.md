# Changelog

## 0.5.0

This feature release packages the biological workflows accompanying the published
Genome Research manuscript. It adds a Sphinx / Read the Docs site and public APIs
for analyses previously expressed through manuscript-specific notebook code.

### Added

- Decoder/encoder freeze and unfreeze methods that preserve frozen BatchNorm
  statistics and Dropout behavior across `model.train()` calls.
- `add_classification_head()` without reinitializing the generative reference,
  plus optional LayerNorm heads and serializable class-label mappings.
- `UniVIRefiner` for head warmup and selected-encoder adaptation, with per-head
  missing-label masks, separate learning rates, reference latent preservation,
  paired reconstruction replay, and validation checkpoint selection.
- RNA, ADT, and ATAC preprocessors fit on training cells and reused on queries.
- Reference bundles preserving weights, configuration, transforms, feature order,
  freeze state, head vocabulary, and user-supplied run metadata.
- Loader and stacked-embedding helpers, scNMT success/coverage adapters, and
  explicit feature-perturbation predictions.
- Lazy top-level imports for the new public workflow APIs.
- Figure-linked biological tutorials through Figure 8, biological supplemental
  Figures S1–S7, and Supplemental Notebook S1, with downloadable notebooks.
- Notebook-source provenance audit, automated helper checks, synthetic tutorial
  checks, documentation CI, and release artifact verification.

### Fixed

- Recursive resolution of lazy submodule imports.
- Classification-head `None` defaults now inherit model dropout/BatchNorm values.
- Fractional perturbations of integer-typed inputs are not silently truncated.
- A stale preprocessing bundle is rejected before overwriting model weights.

### Compatibility and scope

The generative training objective and archived manuscript notebooks are retained.
The new refiner is an explicit opt-in workflow with its own documented checkpoint
and normalization rules. A corrected head default can change behavior in code
that relied on `batchnorm=None` being treated as false; use `batchnorm=False`
explicitly when that is intended. Prior checkpoint formats are not rewritten;
the new reference bundle is a separate format.

Tutorials using Seurat v3 HVGs need the `tutorials` extra (or `scikit-misc` in a
conda environment). `joblib>=1.3` is a core dependency for reference bundles.
Several tutorial preprocessing choices deliberately improve separation of
training and held-out cells; consult the provenance audit before exact numerical
replication. Software checks use small synthetic inputs, not reruns of the full
published biological experiments.

### Version policy

0.5.0 identifies a feature release beyond the manuscript's 0.4.7 implementation.
Publication does not require a major-version change. A future 1.0.0 will mark an
explicit stable public API commitment; after that, compatible additions belong
in minor releases and incompatible public API changes in major releases.
