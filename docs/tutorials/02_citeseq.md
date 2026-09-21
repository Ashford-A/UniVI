# Figure 2: paired RNA–protein integration

**Biological question:** do RNA and surface-protein measurements recover compatible immune identities for the same PBMCs?

Train on the Hao CITE-seq reference, encode held-out cells separately from RNA and ADT, and evaluate correspondence and cell-type transfer. Each cell contributes two points to the stacked embedding. Its fused representation, by contrast, would contribute one point.

| Published panels | Tutorial output | Meaning |
| --- | --- | --- |
| 2A–B | `test-stacked.png` and `.h5ad` | Modality and immune-lineage views of the same coordinates |
| 2C–D | Directional confusion matrices | Which immune identities transfer across assays |
| 2E–F | Per-class F1 plots and CSVs | Performance on rare populations as well as abundant ones |
| Paired correspondence | `rna-adt-metrics.json` | FOSCTTM and Recall@10 from true barcode pairs |

**Inputs:** `data/tutorials/cite/rna.h5ad` and `adt.h5ad`; raw counts in `layers["counts"]`; RNA annotations `celltype.l1`, `celltype.l2`, `celltype.l3`. Cells must genuinely be paired. Feature IDs may differ across modalities. [Data preparation](../guides/data.md).

The Figure 3 notebook supplies the shared CITE workflow here because it also contains the biological reconstructions and S1. It stratifies by `celltype.l3`, uses seed 42, and caps the per-label pool at 1,200 **before** allocating 80/10/10 fractions. This is different from capping the training partition at 1,200. If you have the original barcode split map, provide `splits.tsv`; that is the stronger reproduction record.

RNA uses training-selected HVGs, HVG-first library normalization, and log1p; ADT uses per-cell CLR. The visible preprocessing cells do not apply the Z scaling suggested by some prose descriptions. The tutorial preserves those code semantics and keeps raw counts intact.
**Notebook reference:** [UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb), zero-based cells 15–22, 28–38, 43–61. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-2.png
:alt: Published UniVI Figure 2, supplied manuscript reference
:class: published-figure

Published Figure 2, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/02_citeseq.ipynb>` · {download}`Python script <../examples/02_citeseq.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from univi.preprocessing import RNAPreprocessor, ADTPreprocessor, split_by_label
root,out=paths('cite')
paired=read_paired(root,['rna','adt'])
rna=paired['rna']
```

## Split before fitting any learned transform
A provided splits.tsv takes precedence. max_per_label caps the pool BEFORE
its 80/10/10 division, just as Figure 3 cell 15 does. Overflow is added to test.

```python
splits=split_map(paired,root,lambda:split_by_label(rna.obs['celltype.l3'],
    max_per_label=1200,seed=42))
preprocessors={'rna':RNAPreprocessor(n_hvg=2000,scale=False,normalize_on_selected=True),
               'adt':ADTPreprocessor(scale=False)}
parts=preprocess_parts(paired,splits,preprocessors)
```

## Train the paired generative reference
No cell-type labels enter the VAE loss. They are used for split balancing and evaluation.

```python
model,loaders,metadata=train_reference('cite',parts,out,preprocessors=preprocessors)
test=parts['test']
for mod,a in test.items():a.write_h5ad(out/f'{mod}-test.h5ad')
```

## Make the Figure 2 views and bidirectional label-transfer reports
FOSCTTM uses at most 20,000 paired test cells; label transfer uses the full test set.

```python
joint=draw_embedding(model,[('Hao','rna',test['rna']),('Hao','adt',test['adt'])],
    out,'test-stacked',['modality','celltype.l2'])
report=paired_report(model,test['rna'],test['adt'],'rna','adt',out,label='celltype.l2',k=15)
print(report)
```

## Read the results

The manuscript reports a held-out set of 112,359 pairs, FOSCTTM 0.0209 on a 20,000-pair subsample, and RNA→ADT / ADT→RNA accuracies of 0.958 / 0.962. These are published reference values, not acceptance thresholds for a newly trained model. Feature panels, archived splits, checkpoint selection, and software versions affect the result.

Read confusion matrices alongside cell counts and per-class F1. A strong overall accuracy can coexist with weaker minority-state performance. Neighbor transfer within paired test data is an alignment diagnostic; it does not by itself measure transfer to a different donor or platform.

The script saves its model, objective switches, transforms, feature order, and split barcodes. Continue directly to [Figure 3 and S1](03_cite_reconstruction.md) using those files.
