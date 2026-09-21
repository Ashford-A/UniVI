## Evaluate refinement without circularity

Hold out labeled query cells before training. The script evaluates query-to-query transfer on these held-out cells and writes their barcodes. Marker enrichment in the unlabeled Multiome reference is useful biological corroboration, but expression markers and the model inputs are related evidence.

Do not compute FOSCTTM between Ding and Satpathy: those cells are not matched biological pairs. Compare immune identity, donor/platform composition, neighborhood structure, and held-out labels instead. The published refinement particularly improved Satpathy→Ding transfer; it did not uniformly improve every direction by the same amount.

The example uses the `[64,64,32]` LayerNorm head described in the supplement. The visible original Figure 5 head configuration leaves architecture fields at package defaults. This is an explicitly documented tutorial choice. Both the base and refined references remain available for comparison. See [refinement mechanics](../guides/refinement.md).
