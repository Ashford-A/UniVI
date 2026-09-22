# Genome Research paper reproduction

Ashford AJ, Enright T, Somers J, Nikolova O, Demir E. *Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework.* Genome Research (2026). [doi:10.1101/gr.281431.125](https://doi.org/10.1101/gr.281431.125)

Every figure in the article was produced by a notebook in [`notebooks/GR_manuscript_reproducibility/`](https://github.com/Ashford-A/UniVI/tree/main/notebooks/GR_manuscript_reproducibility). These notebooks are kept exactly as they were run and are the record of the settings used for each analysis; Supplemental Tables S7 (datasets, splits, preprocessing) and S8 (hyperparameters) summarize them.

## Figures and notebooks

| Figure | Analysis | Notebook | Data |
| --- | --- | --- | --- |
| 1 | Model and evaluation overview | — | — |
| 2 | Paired CITE-seq integration | `Figure__2__CITE_paired` (also covered by the Figure 3 notebook) | [`hao_citeseq_pbmc`](datasets.md) |
| 3, S1 | Cross-modal marker reconstruction, subtype structure | `Figure__3__CITE_paired_biological_latent` | `hao_citeseq_pbmc` |
| 4, S2 | Paired Multiome integration, 3D latent views | `Figure__4__Multiome_paired` | `pbmc_multiome_10k` |
| 5, S5 | Bridging unpaired RNA and ATAC cohorts; supervised refinement | `Figure__5__Multiome_bridge_mapping_and_fine-tuning` | `pbmc_multiome_bridge` |
| 6, S6 | Trimodal TEA-seq, held-out well | `Figure__6__TEA-seq_tri-modal` | `teaseq_pbmc` |
| 7, S7 | AML mosaic integration, mutation heads | `Figure__7__AML_bridge_mapping_and_fine-tuning` | `aml_mosaic` |
| 8 | Benchmark against 14 methods | `Figure__8__benchmarking_against_pytorch_tools`, `..._against_R_tools`, `..._merging_and_plotting_runs` | `pbmc_multiome_10k` |
| 9 | Paired-overlap ablation, computational scaling | `Figure__9__paired_data_ablation_and_computational_scaling_performance` (+ `_compile_plots_from_results_df`) | `pbmc_multiome_10k` |
| 10 | Cell-population ablation, modality gating | `Figure_10__cell_population_ablation_MoE` (+ `_compile_plots_from_results_df`) | `pbmc_multiome_10k` |
| S3 | SHARE-seq mouse skin | `Supple_____mouse_skin_SHARE-seq_integration` | `shareseq_mouse_skin` |
| S4 | scNMT-seq mouse gastrulation (RNA, CpG, GpC) | `Supple_____scNMT-seq_mouse_gastrulation_data` | `scnmt_gastrulation` |
| S8–S11 | Hyperparameter, latent-size, dropout sensitivity; scaling | `Supple_____grid-sweep` (+ `_compile_plots_from_results_df`) | `pbmc_multiome_10k` |
| Supp. Notebook S1 | Raw-count decoding and in-silico peak perturbation | `Supple_____Supplemental_Notebook_S1` | `pbmc_multiome_10k` |

Notebook file names begin with `UniVI_manuscript_GR-`. Figures 8–10 and S8–S11 have `_compile_plots_from_results_df` companions that redraw the panels from saved results tables. The Supplemental Code archive published with the article is a snapshot of this repository at release v0.4.7.

## Rerunning an analysis

1. **Install the release the article used**, ideally in the recorded environment:

   ```bash
   conda env create -f envs/univi_v0.4.7_env.yml
   conda activate univi_v0.4.7
   ```

   or `pip install univi==0.4.7` in a fresh environment.

2. **Get the data** listed for the figure on the [datasets page](datasets.md). Datasets marked as hosted download with `univi.datasets`; the others come from the listed accessions.

3. **Point the notebook at your files.** The first cells of each notebook set paths (for example `DATA_ROOT`, `RNA_PATH`); edit them for your machine.

4. **Run the notebook top to bottom.** Training used an NVIDIA GPU for the main-text figures (Supplemental Table S8 lists the device per analysis). Neural network training is not bit-for-bit reproducible across hardware and library versions, so expect small numerical differences.

The `parameter_files/params_*_GR_fig*.json` files and `scripts/revision_reproduce_all.sh` offer a script-driven route through the same analyses (see [scripts](../user_guide/advanced.md#scripts-and-command-line)).

## Running the analyses with the public API

[Paper analyses with the public API](api/index.md) has one notebook per analysis (Figs. 2–7, S3, S4) that downloads the data and runs it with UniVI's reusable components instead of the archived notebooks' own code. [How the hosted datasets are built](building_datasets.md) documents how the downloadable datasets were exported from the archived notebooks.

## The tutorials and the paper

The [tutorials](../tutorials/index.md) follow the same workflows with the newer public API (fitted preprocessors, `make_loader`, `UniVIRefiner`, reference bundles, dataset downloads) and use the paper's CITE-seq and Multiome hyperparameters. They are written for learning and reuse; for the published numbers, run the archived notebooks with v0.4.7.

## Published figures

Figures are reproduced from the article and its Supplemental Material, © 2026 Ashford et al., published by Cold Spring Harbor Laboratory Press under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

::::{tab-set}
:::{tab-item} Main text
```{image} ../_static/figures/figure-2.png
:class: published-figure
:alt: Figure 2
```
Figure 2. Paired CITE-seq integration.

```{image} ../_static/figures/figure-3.png
:class: published-figure
:alt: Figure 3
```
Figure 3. Cross-modal reconstruction of CITE-seq markers.

```{image} ../_static/figures/figure-4.png
:class: published-figure
:alt: Figure 4
```
Figure 4. Paired 10x Multiome integration.

```{image} ../_static/figures/figure-5.png
:class: published-figure
:alt: Figure 5
```
Figure 5. Bridging unpaired RNA and ATAC cohorts.

```{image} ../_static/figures/figure-6.png
:class: published-figure
:alt: Figure 6
```
Figure 6. Trimodal TEA-seq.

```{image} ../_static/figures/figure-7.png
:class: published-figure
:alt: Figure 7
```
Figure 7. AML mosaic integration.

```{image} ../_static/figures/figure-8.png
:class: published-figure
:alt: Figure 8
```
Figure 8. Benchmarking.
:::
:::{tab-item} Supplemental
```{image} ../_static/figures/figure-s1.png
:class: published-figure
:alt: Supplemental Figure S1
```
Supplemental Figure S1. CITE-seq marker recovery at finer annotation.

```{image} ../_static/figures/figure-s2.png
:class: published-figure
:alt: Supplemental Figure S2
```
Supplemental Figure S2. 3D views of the Multiome latent space.

```{image} ../_static/figures/figure-s3.png
:class: published-figure
:alt: Supplemental Figure S3
```
Supplemental Figure S3. SHARE-seq mouse skin.

```{image} ../_static/figures/figure-s4.png
:class: published-figure
:alt: Supplemental Figure S4
```
Supplemental Figure S4. scNMT-seq mouse gastrulation. Processed RNA, CpG, and GpC inputs: [Zenodo 10.5281/zenodo.22885463](https://doi.org/10.5281/zenodo.22885463); load with `uds.load("scnmt_gastrulation")`. See the [S4 API notebook](api/figS4_scnmt.ipynb).

```{image} ../_static/figures/figure-s5.png
:class: published-figure
:alt: Supplemental Figure S5
```
Supplemental Figure S5. Marker validation of transferred labels.

```{image} ../_static/figures/figure-s6.png
:class: published-figure
:alt: Supplemental Figure S6
```
Supplemental Figure S6. TEA-seq held-out well.

```{image} ../_static/figures/figure-s7.png
:class: published-figure
:alt: Supplemental Figure S7
```
Supplemental Figure S7. AML bridge context and mutation heads.
:::
::::
