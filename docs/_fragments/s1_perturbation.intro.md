# Supplemental Notebook S1: raw-feature models and peak sensitivity

**Biological question:** how does a trained cross-modal model respond when selected accessibility inputs change?

This is **Supplemental Notebook S1**, distinct from the CITE-seq **Supplemental Figure S1**. It extends the manuscript's biological analysis with raw-feature RNA/ATAC models, peak–gene annotations, and exploratory in-silico perturbations. The separately supplied executed notebook and the GitHub notebook have identical cell source text.

The visible configuration uses an NB RNA decoder and a Poisson ATAC decoder over original features, with latent dimension 40. It therefore differs fundamentally from Figure 4's Gaussian LSI decoder. A promoter ablation has a defined input meaning only when the model input actually contains that promoter's peaks.

**Inputs:** raw paired counts under `data/tutorials/raw/`, `cell_type`, ATAC coordinates `chrom`, `chromStart`, `chromEnd`, and an explicit `peak_gene_links.tsv` with `peak_name` and `gene_name`. Generate links using a genome-build-matched GTF and the source notebook's annotation functions. The source's promoter-link call uses 10 kb upstream / 2 kb downstream; record the interval rule rather than assuming the helper's default 2 kb / 0.5 kb.

The example trains the raw-feature model, selects GNLY-linked peaks, ablates them in held-out cells, decodes RNA before and after, and compares the target-gene response with null peak sets matched on chromosome, width and training accessibility frequency. It averages decoded per-cell changes within annotated groups. This differs from decoding a synthetic pseudobulk input, another exploratory option in the original notebook.
**Notebook reference:** [UniVI_manuscript_GR-Supple_____Supplemental_Notebook_S1.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Supple_____Supplemental_Notebook_S1.ipynb), zero-based cells 9–22, 54–78, 90–114. Configuration choices are read from notebook code, not parameter JSON files.
