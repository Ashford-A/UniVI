# Figure 3 and S1: cross-modal reconstruction and marker biology

**Biological question:** can one assay recover the lineage-associated features measured by the other?

Use the paired CITE-seq model from [Figure 2](02_citeseq.md). RNA→ADT prediction uses RNA as the only encoder input; ADT→RNA prediction uses ADT alone. The observed target assay is reserved for comparison. Each observed/predicted pair uses the same target-cell UMAP coordinates and shared color limits.

| Panel family | Biological markers | Output |
| --- | --- | --- |
| 3A–B | Cell-type-aggregated RNA / ADT marker means | Paired group heatmaps and mean tables |
| 3C | Coarse immune identities | Shared stacked embedding from tutorial 2 |
| 3D–K | CD79A/CD19, LYZ/CD14, NKG7/CD56, TRAC/CD3 | Observed/predicted marker overlays |
| S1 | Expanded RNA and protein marker sets; `celltype.l2` | Additional overlays and subtype mean tables |

**Inputs:** the saved reference, held-out AnnData files, and stacked UMAP from tutorial 2. No second model fit is needed. Predictions are in the target decoder's training space: log-normalized RNA and CLR ADT in this notebook-derived recipe. They are not raw UMI counts.

CD56 and CD3 antibody features must retain their actual assay IDs. `NCAM1` and `CD3E` describe corresponding genes; they should not silently replace antibody feature names such as `CD56_1` and `CD3_1`. Missing requested markers are written to a report rather than replaced by a different feature.
**Notebook reference:** [UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb), zero-based cells 65–77, 80–101. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-3.png
:alt: Published UniVI Figure 3, supplied manuscript reference
:class: published-figure

Published Figure 3, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s1.png
:alt: Published UniVI Figure s1, supplied manuscript reference
:class: published-figure

Published Figure s1, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/03_cite_reconstruction.ipynb>` · {download}`Python script <../examples/03_cite_reconstruction.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from univi.workflows import load_reference
root,out=paths('cite')
model,preprocessors,metadata=load_reference(out/'reference',device=DEVICE)
rna=sc.read_h5ad(out/'rna-test.h5ad');adt=sc.read_h5ad(out/'adt-test.h5ad')
joint=sc.read_h5ad(out/'test-stacked.h5ad')
# Use each target modality's rows of ONE stacked UMAP, never fit a prediction UMAP.
rna_xy=joint.obsm['X_umap'][joint.obs.modality.eq('rna')]
adt_xy=joint.obsm['X_umap'][joint.obs.modality.eq('adt')]
```

## RNA from ADT: main and expanded biological marker panels

```python
rna_markers=['CD79A','MS4A1','LYZ','FCGR3A','NKG7','GNLY','TRAC','CD3D']
rna_hat=cross_panels(model,adt,rna,'adt','rna',rna_xy,rna_markers,
    'celltype.l1',out,'fig03-rna')
```

## Protein from RNA: keep the actual antibody feature identifiers
NCAM1/CD3E are gene symbols; Hao antibody IDs include CD56-1/CD3-1 or underscore variants.

```python
adt_markers=['CD19','CD20','CD14','CD16','CD56-1','CD56_1','CD3-1','CD3_1','CD4','CD8a']
adt_hat=cross_panels(model,rna,adt,'rna','adt',adt_xy,adt_markers,
    'celltype.l1',out,'fig03-adt')
```

## Supplemental S1: summarize finer cell types using the same predictions

```python
for a,pred,markers,name in [(rna,rna_hat,rna_markers,'rna'),(adt,adt_hat,adt_markers,'adt')]:
    ids=[m for m in markers if m in a.var_names]
    ix=a.var_names.get_indexer(ids)
    for label,values in [('observed',dense(a.X)[:,ix]),('predicted',pred[:,ix])]:
        pd.DataFrame(values,index=a.obs_names,columns=ids).groupby(a.obs['celltype.l2'].astype(str)).mean().to_csv(out/f's1-{name}-{label}-means.csv')
```

## Interpret smooth reconstructions carefully

Successful cross-prediction should preserve lineage contrasts and plausible within-lineage structure. Smoothing alone is not evidence of improvement: a decoder can create clean-looking marker maps while missing cell-level variation. Review the per-feature Pearson correlations and MSE tables, including weakly predicted features.

The group heatmaps use the observed group means to set one standardization for both observed and predicted values. The original notebook also explores separately standardized views; these can emphasize pattern concordance but hide amplitude differences. The tutorial's choice is explicit and reusable.

These RNA-derived protein predictions are model outputs. Do not treat them as additional independent protein measurements in hypothesis tests. For denoising and model-based synthetic sampling, see [additional analyses](../guides/denoising-generation.md).
