# Figure-to-tutorial map

The main biological series covers Figures 2–7 and supplemental Figures S1–S7. Figure 1 explains the framework. Figure 8 has an evaluation tutorial because its panels compare methods rather than introduce a new biological dataset. Supplemental Notebook S1 adds a distinct raw-feature perturbation workflow.

| Figure | Biological task | Source notebook | Tutorial |
| --- | --- | --- | --- |
| 1 | Model and study-design concepts | Manuscript Methods / package | [Framework](01_framework.md) |
| 2 | Paired CITE-seq integration | Figure 2; shared workflow in Figure 3 | [RNA–protein](02_citeseq.md) |
| 3 + S1 | Marker reconstruction and subtype biology | Figure 3 | [Cross-modal prediction](03_cite_reconstruction.md) |
| 4 + S2 | Paired Multiome and additional latent views | Figure 4 | [RNA–ATAC](04_multiome.md) |
| 5 + S5 | Independent-cohort bridging, optional cell-type refinement | Figure 5 | [Bridge mapping](05_bridge.md) |
| 6 + S6 | Trimodal TEA-seq and well holdout | Figure 6 | [Trimodal PBMCs](06_teaseq.md) |
| 7 + S7 | AML mosaic, mutation heads, stemness and patient context | Figure 7 | [AML biology](07_aml.md) |
| 8 | Biological-structure and correspondence metrics | Figure 8 Python / R / merge notebooks | [Evaluation](08_evaluation.md) |
| S3 | Nonhematopoietic tissue and differentiation | SHARE-seq mouse skin | [Mouse skin](s03_shareseq.md) |
| S4 | RNA + methylation + accessibility | scNMT-seq mouse gastrulation | [scNMT-seq](s04_scnmt.md) |
| Supplemental Notebook S1 | Raw-feature decoding and peak perturbation | Supplied executed S1 notebook | [Peak sensitivity](s1_perturbation.md) |

Figures 9–10 and S8–S11 cover overlap, population ablations, hyperparameter sensitivity, and computational scaling. They remain linked in the complete [source map](../reference/notebook-provenance.md), but are outside the requested biological tutorial series.

Each tutorial can be read as a web page or downloaded as a notebook. The only dependency between biological scripts is that the Figure 3 reconstruction script consumes the reference and test data saved by the Figure 2 script. Other scripts train their own references.

## What “reproduce” means here

There are three useful levels:

1. **Recreate the biological workflow:** follow the tutorial's public APIs and data contracts.
2. **Repeat the archived analysis:** use the original notebook, exact barcode maps, feature panels, preprocessing objects, checkpoint, and package revision.
3. **Reassemble the exact published panels:** use the saved outputs and plotting state from that run.

This documentation provides the first level, traces it to the second, and includes published figures as reference images. It does not claim that retraining under modified or incomplete run state will reproduce every published value or UMAP coordinate.
