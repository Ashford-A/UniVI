## Two evaluations with different evidential meaning

S4A–E use the held-out 114-cell test partition. S4F–J use all 1,140 paired cells with the same trained checkpoint. The latter includes training data and is labeled transductive/all-cell analysis; its stronger structure is not a separate out-of-sample validation. Published RNA–CpG FOSCTTM is 0.192 on the held-out subset and 0.029 on all cells.

Inspect modality, developmental stage, embryo of origin and atlas-transferred lineage labels together. An embedding ordered by stage but dominated by embryo requires a different interpretation from lineage structure replicated across embryos.

The modality-weight heatmap uses **normalized posterior-precision contributions**. In the visible source configuration, learned gating is disabled; the notebook's `router_x_precision` request can still return nonuniform precision-based contributions. These weights summarize encoder uncertainty across latent dimensions. They are not causal importance scores or proof that a learned router discovered biology. See [representations and weights](../guides/representations.md).

Optional RNA→CpG evaluation masks uncovered target entries and reports predicted fractions. Weighting errors by coverage would answer a different question and should be named explicitly.
