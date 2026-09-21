# S4: coverage-aware trimodal scNMT-seq

**Biological question:** can RNA, DNA methylation and chromatin accessibility jointly describe developmental progression without requiring feature-level correspondence?

Use the parsed mouse-gastrulation bundle to build three paired modalities: RNA, genebody CpG methylation, and genebody GpC accessibility. The adapter is promoted from the actual scNMT notebook, including the metadata-based mapping from RNA IDs to shared sample IDs.

| Representation | Encoder input | Reconstruction target |
| --- | --- | --- |
| RNA | Library-normalized log1p counts, no HVG or Z scaling | Gaussian target in the same expression space |
| CpG | Supplied fractions in `.X` | `meth_successes`, `meth_total_count` |
| GpC | Supplied fractions in `.X` | `acc_successes`, `acc_total_count` |

**Input:** `data/tutorials/scnmt/scnmt_gastrulation.tar.gz`, containing `rna/counts.txt.gz`, `met/feature_level/genebody.tsv.gz`, `acc/feature_level/genebody.tsv.gz`, and `sample_metadata.txt`. [Acquisition and schemas](../guides/data.md).

A feature with zero coverage is unobserved. It must not be treated as a confidently measured unmethylated feature. The beta-binomial likelihood receives successes and coverage through `recon_targets_spec`; the required collator preserves both arrays. Fraction-only Gaussian shortcuts change the statistical model.

Follow the proof-of-concept settings explicitly: no extra joint QC or feature filtering beyond the parsed-bundle selection, random 85/5/10 split, seed 0. The actual loaders use batch size **24**; the notebook's `TrainingConfig(batch_size=16)` does not override a prebuilt loader.
**Notebook reference:** [UniVI_manuscript_GR-Supple_____scNMT-seq_mouse_gastrulation_data.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Supple_____scNMT-seq_mouse_gastrulation_data.ipynb), zero-based cells 4–19, 25–34, 38–56. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-s4.png
:alt: Published UniVI Figure s4, published figure reference
:class: published-figure

Published Figure s4, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
