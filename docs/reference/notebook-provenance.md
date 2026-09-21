# Notebook provenance and implementation audit

This documentation was derived from the published manuscript, supplemental material, Supplemental Code archive, executed Supplemental Notebook S1, and the analysis notebooks at repository commit **`8353ec8d422841e756b3abe4e9a5c4286c0c81dd`**. Parameter JSON files were not used as the implementation authority.

All **19 mapped notebooks** in the Supplemental Code archive have the same cell types and cell source text as their repository counterparts. The published Supplemental Notebook S1 also has identical cell source to its repository counterpart and contains **113 executed code cells**. This establishes source identity; it does not establish that every notebook's current source was the exact run that generated a published panel.

The {download}`machine-readable provenance manifest <../_downloads/notebook-provenance.json>` records source-material SHA-256 hashes, normalized notebook-source hashes, archive mappings, cell counts, and the comparison results. Cell indices below are **zero-based positions among all notebook cells**, including Markdown. Pinned source links are provided on each tutorial page.

## Figure-to-workflow map

| Published material | Notebook implementation reference | Tutorial |
| --- | --- | --- |
| Figure 1 | Architecture and existing public API | [Framework](../tutorials/01_framework.md) |
| Figure 2 | Figure 2 CITE paired; Figure 3 shared biological workflow | [CITE-seq](../tutorials/02_citeseq.md) |
| Figure 3, S1 | Figure 3 CITE biological latent | [Marker reconstruction](../tutorials/03_cite_reconstruction.md) |
| Figure 4, S2 | Figure 4 Multiome paired | [Multiome](../tutorials/04_multiome.md) |
| Figure 5, S5 | Figure 5 Multiome bridge and fine-tuning | [Reference bridging](../tutorials/05_bridge.md) |
| Figure 6, S6 | Figure 6 TEA-seq trimodal | [TEA-seq](../tutorials/06_teaseq.md) |
| Figure 7, S7 | Figure 7 AML bridge and fine-tuning | [AML](../tutorials/07_aml.md) |
| Figure 8 | Python benchmark runner; R runner; merge/plot notebook | [UniVI metric families](../tutorials/08_evaluation.md) |
| S3 | Mouse skin SHARE-seq integration | [SHARE-seq](../tutorials/s03_shareseq.md) |
| S4 | scNMT-seq mouse gastrulation | [scNMT-seq](../tutorials/s04_scnmt.md) |
| Supplemental Notebook S1 | Executed raw-feature Multiome analysis | [Feature perturbation](../tutorials/s1_perturbation.md) |

Supplemental **Figure** S1 and Supplemental **Notebook** S1 are different sources. Figures 9–10 and S8–S11 concern robustness, ablation, and scaling; they remain in the [original manuscript notebook directory](https://github.com/Ashford-A/UniVI/tree/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility) and are outside this biological tutorial series.

## CITE-seq: Figure 2 is not the same split recipe as Figure 3

Figure 2 cell 19 stratifies by `celltype.l1`, caps the eligible pool at 2,000 per label, and uses seed 0. Figure 3 cell 20 instead uses `celltype.l3`, a pool cap of 1,200, and seed 42. The cap is applied **before** the train/validation/test split; it is not a training-set size of 1,200. Overflow beyond the cap is then added to test (`unused_to_test=True`), so it is retained rather than discarded. The shared Figures 2–3 tutorial follows the Figure 3 workflow and exposes an explicit barcode split map for original-run replication.

The Figure 3 visible preprocessing uses HVG-first RNA library normalization followed by log1p and ADT CLR. These cells do not perform the Z scaling implied by some general descriptions. The tutorial preserves that behavior. Figure 3 cells 30 and 35 specify latent dimension 30, beta 1.5, alignment gamma 5, encoder/decoder dropout 0.1/0, 3,000 maximum epochs, learning rate 1e-3, weight decay 1e-4, and patience 100. The Figure 2 training recipe has a different patience setting (50).

## Multiome: split cap, retained LSI component, and count preservation

Figure 4 cell 18 caps the per-label pool at 5,000 before an 80/10/10 split. This is different from the supplemental table's train/validation cap description. Cell 22 fits 100 LSI components and keeps component 0. RNA preprocessing is not Z-scored in the visible workflow. Cells 30 and 35 set beta 1.25, gamma 4.35, encoder/decoder dropout 0.1/0.05, and 5,000 maximum epochs with patience 300.

The tutorial follows the visible model recipe and retained-LSI convention. It preserves raw counts in a separate layer rather than mutating that record while preparing a Gaussian input. S2's additional embedding views are represented by explicitly labeled modern 3D PCA views; they are not claimed to reproduce the published 3D projection coordinates.

## Bridge mapping: head architecture and preprocessing access

Figure 5 cell 18 uses a 90/10 reference split. Cell 21 selects a shared HVG set using combined reference/query information, and that set feeds the subsequent preprocessing. The tutorial deliberately fits learned preprocessing on the reference training cells only, after defining the common measurable feature universe. This changes the original feature-selection access and should not be described as exact numerical replication.

The visible model/training settings use beta 1, gamma 5, encoder/decoder dropout 0.25/0.05, and 5,000 maximum epochs. The cell 128 refinement helper's default is 1,000 epochs with 300 head-only warmup epochs; another description lists a longer run. The visible head configuration near cell 131 leaves architecture fields at package defaults, whereas the supplement describes a `[64,64,32]` LayerNorm head. The tutorial makes that latter architecture explicit and labels it as a choice.

The public refiner preserves the task—head warmup, frozen decoders, selected encoder adaptation, and paired generative replay—with explicit observed-label normalization and checkpoint rules. It is not a line-for-line transcription of the bespoke notebook optimizer loop.

## TEA-seq: holdout interpretation and transform corrections

The published design trains on wells 3, 4, and 6 and holds out well 5. These are capture/library partitions within the run; this is not a new-patient test. The visible configuration in cell 29 uses gamma **1.45**, rather than 1.35 in a supplemental description, with beta 1.15 and RNA/ADT/ATAC reconstruction weights 1/2/1.45.

Several original preprocessing details complicate strict inductive interpretation. RNA scaling is followed by computing a mean/std from the already scaled matrix for later reuse; ADT scaling is performed separately by well, including the holdout; and the HVG workflow applies Seurat v3 selection to transformed expression in the visible preparation. Reference-well transforms are also fit before the within-reference split.

The tutorial corrects these operations: raw-count HVG selection and all learned scales/TF-IDF/SVD use only optimizer-training cells, with the same fitted objects applied to validation and well 5. The resulting workflow is easier to reuse for a genuinely unseen batch, but its numbers should not be substituted for the publication's original run.

## AML: model recipe, label semantics, and split unit

Figure 7 cell 39 uses latent dimension 30, beta 1.15, gamma 1.75, RNA encoder widths `[1024,512,256,128,64]`, ADT widths `[128,64,32]`, no encoder/decoder BatchNorm, and ADT reconstruction weight 3. The visible training recipe uses learning rate 1e-4 and weight decay 1e-5.

The mutation-head implementation near cell 76 uses per-gene `[64,32]` LayerNorm heads with dropout 0.1. The cell 88 call uses head LR 1e-4, encoder LR 1e-5, latent-preservation weight 5, warmup 10, maximum epochs 1,000, patience 50, and weight decay 1e-6. The tutorial follows these values through the new public refiner and retains missing labels as unknown.

The paired CITE bridge uses grouped samples; mutation-head evaluation uses cell-level splits. Neither the latter nor pooled all-cell probability landscapes establish unseen-patient performance. The notebook's LSC17 implementation is the mean of available panel genes, with Z scoring for visualization, rather than the weighted clinical prognostic assay. ADT-only LSC17 values derive from imputed RNA.

## SHARE-seq: actual split and TF-IDF formula

The visible cell 34 split is 80/10/10 with seed 0; a supplemental methods description instead says 75/10/15. Cell 47 uses beta 1.25 and gamma 6.35. The tutorial follows the code's split defaults and permits explicit barcode maps.

The ATAC transform uses `IDF = log1p(n / (1 + df))`, then `log1p(TF * IDF * 1e4)` and L2 normalization. It fits 101 SVD components and drops component 0. This is not interchangeable with the Figure 4 sklearn TF-IDF recipe. The tutorial retains the mouse blacklist/QC requirement and learns frequency filtering and transformations on training cells.

## scNMT-seq: loader batch size and precision weights

Cell 18's training configuration says batch size **16**, but the explicitly constructed loaders in cell 16 use batch size **24**. The supplied loaders determine runtime batching. The tutorial therefore uses 24, with beta 1.35, gamma 6.25, and latent dimension 20 from the visible model configuration.

The preprocessing retains log-normalized RNA without HVG selection or Z scoring. CpG and GpC targets are success counts with coverage for beta-binomial losses. The adapter preserves the original `require_qc=False`, `filter_features=False`, minimum-coverage-1 analysis choice instead of silently changing the cohort.

The source has 969/57/114 train/validation/test cells out of 1,140. S4's all-cell display includes training cells and uses the same checkpoint. Cell 49 requests `router_x_precision`, but learned gating is disabled in the model configuration. The plotted nonuniform values can arise from posterior precision; the tutorial calls them **normalized posterior-precision contributions**, not learned router weights.

## Figure 8 and Supplemental Notebook S1

The Figure 8 Python runner's visible UniVI recipe uses beta 1.35, gamma 3.75, latent dimension 30, and 200 maximum epochs. The new tutorial demonstrates its ten metric families over three seeds. It does not rerun the R/Python competitors or claim to reconstruct the complete published cross-validation summary.

Supplemental Notebook S1 cell 17 configures a raw-feature NB RNA / Poisson ATAC model with latent dimension 40, beta 0.85, and gamma 1.75. Its promoter-link helper has one set of defaults, but the invocation near cell 63 uses a 10 kb upstream / 2 kb downstream window. The link table supplied to the new tutorial must record the actual window and genome build.

The original notebook includes exploratory large count additions, including a GNLY-related example near cell 95. The portable tutorial starts with peak ablation and matched null sets, with an explicit distinction between model sensitivity and causal regulation. It averages decoded changes over cells; it does not silently equate that with decoding a pseudobulk input.
