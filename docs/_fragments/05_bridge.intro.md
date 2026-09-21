# Figure 5 and S5: bridge independent cohorts and refine labels

**Biological question:** can a paired Multiome reference place independent RNA-only and ATAC-only cohorts into a useful shared coordinate system?

This tutorial has two separately saved stages. First, fit the reference and project Ding RNA and Satpathy ATAC by encoder inference. Second, optionally add a cell-type head and refine encoders using harmonized query labels, with the generative decoders frozen and the paired reference replayed during optimization.

| Panels | Stage and output |
| --- | --- |
| 5A–C | Unsupervised reference projection, colored by cohort, coarse label, technology |
| 5D–F | Corresponding views after supervised refinement |
| 5G | Directional query-to-query transfer before and after refinement |
| 5H–J | Reference-cell annotation and marker validation |
| S5 | Predicted label, maximum class probability, expanded marker views |

**Inputs:** paired `rna.h5ad`/`atac.h5ad`, `ding_rna.h5ad`, `satpathy_atac.h5ad` under `data/tutorials/bridge/`. Queries need `celltype_harmonized` and `technology` metadata. ATAC query counts must be quantified in the reference peak universe or share its retained peaks exactly. A same-width matrix from a different peak set or an independently fitted LSI basis is not compatible.

The modern tutorial selects HVGs from reference training counts on a shared measurable gene panel. The visible original notebook also contains a combined reference/query HVG-selection cell that feeds its later preprocessing call. We make this training-only modernization explicit rather than claiming identical reproduction of that exploratory execution path.

The new public API replaces manual head injection and parameter-name loops. Warmup updates only the head; the second stage updates relevant encoders and the head. The replay term uses the original paired generative objective. Decoder freezing fixes the decoder function in latent coordinates; changing encoders can still change a cell's decoded profile.
**Notebook reference:** [UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb), zero-based cells 18, 20–23, 36–44, 63–68, 125–148. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-5.png
:alt: Published UniVI Figure 5, published figure reference
:class: published-figure

Published Figure 5, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s5.png
:alt: Published UniVI Figure s5, published figure reference
:class: published-figure

Published Figure s5, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```
