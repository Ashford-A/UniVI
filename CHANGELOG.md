# Changelog

## 1.2.1

### Fixed

- Relative-position attention bias (`TransformerConfig(use_relpos_bias=True)` with peak coordinates
  attached): distances are now computed in float32 per chromosome, instead of float64 with per-chromosome
  offsets, and the gradient of the small bias table is computed as a histogram (`torch.bincount`) instead
  of through the scatter of the default indexing backward. In a real-data run of the transformer tutorial
  on a consumer GPU, the model with genomic position had trained about 18 times slower than the same model
  without it. Bins are unchanged within a chromosome (up to float32 rounding of positions, 16 bp at 250 Mb)
  and pairs on different chromosomes still fall in the farthest bin.

### Changed

- Experimental tutorials, after a first run on the real data:
  - *In-silico chromatin perturbation*: the chance rate of motif matches is measured on shuffled peak
    sequences rather than assumed, so short motifs are no longer flagged merely for being short; flagged
    motif sets are no longer used by default; the tiling-screen null is matched on accessibility, width and
    GC; the genome-wide axis shows canonical chromosomes only.
  - *Unified PBMC atlas*: the adversarial cohort models also train on mixed-cohort RNA-only batches (with
    single-cohort batches alone, mixing did not improve at any adversary strength); "who is the expert" now
    measures how far the fused embedding moves when each modality is left out; the latent walk reports how
    well its decoded protein changes agree with the observed ones.
  - *Transformer encoders*: ATAC features default to peaks near the modeled genes (`ATAC_FEATURES =
    "near_genes"`); in a run with the most accessible peaks genome-wide, only about four gene tokens per
    cell had a nearby peak among the peak tokens. A new peak–gene link benchmark scores every gene–peak
    pair within 100 kb by distance, fused-transformer attention and gradient attribution, against
    correlation-based reference links across pseudobulks (with a background of accessibility-matched peaks
    on other chromosomes), overall and within distance bins, with loop-style locus views.
  - *Prediction uncertainty* and *cross-modal QC*: conformal calibration uses a split kept apart from the
    validation cells used for early stopping. With validation cells doubling as calibration cells, the QC
    notebook's realized false discovery rate was 16% at a 10% target.

## 1.2.0

### Added

- *Extended and experimental tutorials* (`docs/tutorials/experimental/`), five Colab-ready notebooks that use the
  public API and `univi.datasets`: in-silico chromatin perturbation of transcription regulators (ATAC → RNA),
  with matched random-peak nulls, length-corrected JASPAR motif peak sets, a single-peak in-silico tiling
  screen at each regulator's locus and genome-wide views of where effects land; per-modality and fused
  transformer encoders, with attention maps, attention entropy, attention rollout and a permutation-calibrated
  cis-attention test against untrained controls; a unified PBMC atlas from CITE-seq, Multiome and TEA-seq,
  validated on a held-out TEA-seq well, with an adversarial cohort head tuned on validation cells,
  mixture-of-experts weights per cell type, modality agreement and latent traversals; calibrated
  (split-conformal) uncertainty for cross-modal predictions; and quality control from cross-modal
  disagreement (mis-paired barcodes and doublets, conformal p-values, false discovery rate control).
  `tests/test_tutorials.py` executes them on synthetic data and `scripts/execute_tutorials.py --experimental`
  on the real data.
- `TokenizerConfig` fields for the options the encoders' tokenizer reads: `use_feature_embedding`,
  `feature_emb_dim`, `feature_emb_mode`, `use_coord_embedding`, `n_chroms`, `coord_emb_dim`, `coord_mode`,
  `coord_mlp_hidden`, `coord_num_frequencies` and `token_proj_dim`. As fields they reach the fused encoder and
  are written by `save_reference` and read by `load_reference`.
- Tokenizer mode `topk_embed` now works in the encoders: top-k channel tokens with a learned feature-ID
  embedding of width `d_model`, and coordinate embeddings when `use_coords=True` (coordinates can be given in
  `feature_info`).
- `TransformerConfig(use_relpos_bias=True)` now reaches the per-modality transformer encoders; when peak
  coordinates are attached (`vec2tok.set_feature_coords`), the binned genomic-distance attention bias is
  applied, with tokens on different chromosomes in the farthest bin.
- `tests/test_transformer_fixes.py` covers the fixes below.
- The synthetic Multiome stand-in used by the tests includes gene and peak coordinates and a panel of
  transcription-factor genes.

### Changed

- The Datasets documentation page now describes every hosted dataset (files, feature counts, what `.X` and
  `.obs` hold, splits, original data) and includes the export recipes previously on *How the hosted datasets
  are built*, which now points there.
- The Genome Research manuscript notebooks (`notebooks/GR_manuscript_reproducibility/`) render inline figures
  at `DISPLAY_DPI` (80) so the executed notebooks stay small; figures saved to disk keep their own resolution.
- `.gitignore` excludes files written by the tutorials (`univi_outputs/`).

### Fixed

- Fused multimodal transformer with `loss_mode="v2"`: the per-modality encoders are trained only by the
  alignment term. While that term was annealed to weight 0 it stayed in the backpropagated loss, so the
  encoders received zero gradients, and the coupled weight decay of `torch.optim.Adam` drove their weights to
  zero before alignment started; they did not recover, and every per-modality embedding became constant. A
  zero-weighted alignment term is now left out of the backpropagated loss, so the optimizer leaves those
  encoders untouched until alignment begins. Other models are unaffected (the term contributed nothing).
- Feature-ID embedding options set on a `TokenizerConfig` were silently dropped by the fused encoder, which
  copies tokenizer configs with `dataclasses.replace`; they are now fields.
- Genomic coordinate embeddings ignored the position: the coordinate MLP normalized a single scalar with
  `LayerNorm`, which maps every value to the same output (and `coord_scale` was multiplied where
  `univi.models.tokenizers` divides). Positions are now encoded with multi-scale sinusoidal features
  (wavelengths 1 kb to 100 Mb) before the MLP.
- `topk_embed` passed configuration validation but was rejected when the encoder was built.
- `use_relpos_bias`, `relpos_num_bins` and `relpos_max_dist` were dropped when a `TransformerConfig` was
  converted for the encoder.
- `aml_mosaic`: the Knorr et al. (2023) CITE-seq data are cited as GEO GSE220473 (the CITE-seq SubSeries of
  SuperSeries GSE220474). The registry description and the documented export recipe now match the hosted
  DAb-seq file, which holds processed protein values and variant-level genotype columns rather than
  `mut_<GENE>` columns.
- `shareseq_mouse_skin`: the documentation and the Supplemental Fig. S3 API notebook now give the split stored
  in the hosted files (21,845 / 3,112 / 6,263 cells, stratified 70/10/20) instead of describing it as 80/10/10.

## 1.1.0

### Fixed

- Replaced the scNMT-seq dataset-catalog placeholder and added the published dataset DOI
  ([10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463)), file links,
  dimensions, and loading guidance across the documentation and S4 notebooks.

- `MultiModalDataset` copies AnnData views once instead of re-slicing them for every item, which made
  training on subsets (for example `adata[train_idx]`) extremely slow or run out of memory.

### Added

- `univi.datasets.export_dataset` packages AnnData objects (raw counts, selected metadata, an optional
  train/validation/test split in `obs["split"]`) with md5 checksums, a manifest, and a record description.
- `scripts/zenodo_upload.py` uploads an exported dataset to a Zenodo draft through the deposit API, verifies
  each file's checksum, and links the record to the article. `scripts/prepare_zenodo_release.py prepare` now
  uses `export_dataset`.
- Seven notebooks under *Paper analyses with the public API* that run the analyses of Figs. 2–7 and
  Supplemental Figs. S3–S4 through `univi.datasets`, the fitted preprocessors, `UniVITrainer`, and
  `UniVIRefiner`, with the settings of the archived notebooks.
- Documentation of how each hosted dataset is exported from the archived notebooks.

### Changed

- Placeholder registry entries use the file names written by `export_dataset`.
- The tutorial test suite also executes the paper-analysis notebooks on synthetic stand-in data.
- `scnmt_gastrulation` downloads three compact `.h5ad` files from Zenodo (RNA counts; CpG/GpC fractions
  with success/coverage layers; the article's split) instead of the 33 GB EBI bundle.

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
