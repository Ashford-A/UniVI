## Interpret the weaker reconstruction regime

The publication reports 3,139 test pairs across 22 populations, FOSCTTM 0.0546, and relatively modest mean per-feature cross-reconstruction correlations (0.184 for RNA→LSI and 0.132 for ATAC→RNA). Good lineage alignment does not imply precise cell-level recovery of every feature.

Inspect per-class support and errors among neighboring hair-lineage states. Small anchor populations and continuous differentiation can lower macro-F1 without making the entire manifold uninterpretable. The tutorial writes cell counts and per-class metrics to support that distinction.

The supplement contains inconsistent split prose: its methods mention 75/10/15, while its table and the visible notebook use 80/10/10. The tutorial follows the code and accepts an archived barcode map to remove ambiguity. It does not infer lineage directionality or developmental causality from UMAP proximity alone.
