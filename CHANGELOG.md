# Changelog

## 1.0.0

First stable release following publication of the UniVI article in *Genome Research*
([doi:10.1101/gr.281431.125](https://doi.org/10.1101/gr.281431.125)). The public API documented
at <https://univi.readthedocs.io> now follows semantic versioning; transformer encoders and the
multimodal transformer remain experimental. There are no changes to the training objective, model
architecture, or checkpoint and reference-bundle formats, so code written for 0.5.x runs unchanged.

### Added

- `univi.datasets` downloads, verifies (md5/sha256), and caches the datasets used in the tutorials and
  the article: `list_datasets()`, `load()`, `fetch()`, `dataset_info()`, `register_dataset()`, and
  shortcuts such as `pbmc_multiome_10k()`. Mirrors, `file://` URLs, `UNIVI_DATA_DIR`, and
  `UNIVI_DATASET_REGISTRY` support offline clusters and private data.
- A warning when `use_moe_gating=True` is combined with an objective that gives the gating network no
  training signal (`loss_mode="v1"` with a non-fused `v1_recon` and no classification heads).
- Rebuilt documentation: six tutorial notebooks (quickstart, CITE-seq, query mapping, classification
  heads and refinement, generation and perturbation, custom modalities) that open in Colab and are
  executed in CI; a user guide; a figure-by-figure guide to the article's analyses; and an API reference.
- `CITATION.cff`, a contributing guide, and issue templates.

### Changed

- `univi.datasets` is now a package; the scNMT-seq readers live in `univi.datasets.scnmt` and all
  previous import paths still work.
- Package metadata: keywords, classifiers, documentation and paper links, and `docs`/`test` extras.

### Fixed

- `MultiModalDataset` copies label arrays before converting them to tensors, avoiding a PyTorch warning
  for read-only arrays (for example those returned by pandas 3).
- Documentation examples that did not match the API (`MultiModalDataset(X_key=...)`,
  `export_supplemental_table_s1`, `TokenizerConfig` coordinate fields, the hyperparameter-search entry point).

### Removed

- The script-generated tutorial pages under `docs/examples/`, `scripts/build_tutorial_docs.py`,
  `tests/smoke_examples.py`, and `RELEASE_NOTES_v0.5.0.md` (superseded by the notebooks, the notebook
  tests, and this changelog).

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
