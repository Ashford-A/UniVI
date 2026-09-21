# UniVI: biological analysis tutorials

**Integrate modalities. Decode biological signals. Extend a reference.**

UniVI learns related latent representations from heterogeneous single-cell assays. This documentation turns the Genome Research analyses into a practical series: paired RNA–protein and RNA–ATAC integration, cross-modal marker reconstruction, independent-cohort bridging, trimodal TEA-seq, mutation-aware AML refinement, mouse skin SHARE-seq, and coverage-aware scNMT-seq.

Start with the [small runnable quickstart](tutorials/00_quickstart.md), or choose a [figure-linked tutorial](tutorials/index.md). Each tutorial includes source-notebook provenance, input requirements, a downloadable notebook and script, expected output files, and guidance on biological interpretation.

```{note}
These tutorials require **UniVI 0.5.0 or later**. Decoder-freezing methods, staged refinement, saved reference transforms, and the new workflow helpers are part of the Python package in 0.5.0. The historical 0.4.7 release does not provide them. See [Installation](installation.md) for package and source-checkout options.
```

| Your analysis | Start here |
| --- | --- |
| Paired RNA + protein | [CITE-seq integration](tutorials/02_citeseq.md) → [marker reconstruction](tutorials/03_cite_reconstruction.md) |
| Paired RNA + accessibility | [Multiome](tutorials/04_multiome.md) or [SHARE-seq skin](tutorials/s03_shareseq.md) |
| Independent RNA-only and ATAC-only cohorts | [Reference bridging and refinement](tutorials/05_bridge.md) |
| Three measured modalities | [TEA-seq](tutorials/06_teaseq.md) or [scNMT-seq](tutorials/s04_scnmt.md) |
| RNA, protein, and genotype across AML cohorts | [AML mosaic analysis](tutorials/07_aml.md) |
| Evaluate a new representation | [Figure 8 metric families](tutorials/08_evaluation.md) |
| Explore peak-associated model responses | [Raw-feature perturbation analysis](tutorials/s1_perturbation.md) |

The tutorials use the analysis notebooks as their implementation references. A [provenance audit](reference/notebook-provenance.md) records differences among visible notebook code, the manuscript, and supplemental tables. Published figure images are included as labeled reference examples; the original biological training runs have not been repeated to produce new performance claims.

```{toctree}
:maxdepth: 2
:caption: Start

installation
tutorials/00_quickstart
tutorials/index
tutorials/01_framework
guides/data
```

```{toctree}
:maxdepth: 1
:caption: Biological tutorials

tutorials/02_citeseq
tutorials/03_cite_reconstruction
tutorials/04_multiome
tutorials/05_bridge
tutorials/06_teaseq
tutorials/07_aml
tutorials/08_evaluation
tutorials/s03_shareseq
tutorials/s04_scnmt
tutorials/s1_perturbation
```

```{toctree}
:maxdepth: 1
:caption: Analysis guides

guides/representations
guides/preprocessing
guides/refinement
guides/evaluation
guides/denoising-generation
guides/reproducibility
guides/troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Reference and maintenance

reference/api
reference/notebook-provenance
reference/validation
reference/changes
reference/citation
guides/publishing
```
