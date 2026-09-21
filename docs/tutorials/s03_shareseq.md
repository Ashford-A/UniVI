# S3: SHARE-seq mouse skin and differentiation structure

**Biological question:** does paired alignment extend beyond blood, across developmental lineages and a sparse accessibility assay?

This workflow adapts the paired RNA–ATAC analysis to late-anagen mouse skin: epidermal/hair-follicle states, mesenchymal, vascular, neural-crest-derived and immune populations. It uses the assay-specific QC, gene-family exclusions, log-TF-IDF and first-component removal recorded in the SHARE-seq notebook.

**Inputs:** `data/tutorials/share/rna.h5ad` and `atac.h5ad`, raw counts, and `cell_type`; ENCODE mm10 blacklist peaks must already be removed. Record that upstream operation in `atac.uns["blacklist_removed"]`. Do not merely set the flag to bypass filtering.

RNA cells pass total-count, detected-gene, and mitochondrial-fraction limits. ATAC cells pass count limits; paired barcodes are retained together. Author-labeled `Mix` cells are excluded for correspondence with this analysis, not because every transitional state is necessarily an artifact.

Fit 5,000 eligible RNA HVGs on training counts and exclude the notebook's mitochondrial/ribosomal/Rik/pseudogene families. Retain ATAC features accessible in 0.25–80% of training cells, fit 101 components, drop component 0, and reuse all transformations unchanged. This recipe uses the notebook's `log1p(n/(1+df))` IDF followed by `log1p(TF*IDF*1e4)` and L2 normalization.

S3A–B become the stacked modality/cell-type views; S3C–D become the per-feature RNA→LSI and ATAC→RNA correlation/MSE distributions.
**Notebook reference:** [UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Supple_____mouse_skin_SHARE-seq_integration.ipynb), zero-based cells QC and blacklist cells, 32–49, 54–63. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-s3.png
:alt: Published UniVI Figure s3, published figure reference
:class: published-figure

Published Figure s3, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/s03_shareseq.ipynb>` · {download}`Python script <../examples/s03_shareseq.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from univi.preprocessing import RNAPreprocessor,ATACPreprocessor,split_by_label
root,out=paths('share')
data=read_paired(root,['rna','atac']);rna,atac=data['rna'],data['atac']
if not atac.uns.get('blacklist_removed',False):
    raise ValueError('Remove ENCODE mm10 blacklist peaks before this tutorial; see preparation guide.')
```

## Dataset-specific QC before the paired split

```python
rna.var['mt']=rna.var_names.str.upper().str.startswith('MT-')
rna.X=rna.layers['counts'].copy()
sc.pp.calculate_qc_metrics(rna,qc_vars=['mt'],percent_top=None,log1p=False,inplace=True)
atac_total=np.asarray(atac.layers['counts'].sum(axis=1)).ravel()
keep=(rna.obs.total_counts.between(300,30000)&(rna.obs.n_genes_by_counts>=200)&
      (rna.obs.pct_counts_mt<=5)&(atac_total>=250)&(atac_total<=12500)&rna.obs.cell_type.ne('Mix'))
data={m:a[keep].copy() for m,a in data.items()}
splits=split_map(data,root,lambda:split_by_label(data['rna'].obs.cell_type,seed=0))
exclude=[g for g in data['rna'].var_names if g.startswith(('mt-','Mt-','MT-','Rps','Rpl','Mrps','Mrpl')) or g.endswith('Rik') or '-ps' in g]
prep={'rna':RNAPreprocessor(5000,scale=True,normalize_on_selected=False,exclude_genes=exclude,min_cells=10),
      'atac':ATACPreprocessor(8 if SMOKE else 101,method='signac',drop_first=True,
          min_fraction=.0025,max_fraction=.8,random_state=42)}
parts=preprocess_parts(data,splits,prep)
model,loaders,meta=train_reference('share',parts,out,preprocessors=prep)
test=parts['test']
```

## Test-set geometry and rare-population performance

```python
joint=draw_embedding(model,[('skin',m,a) for m,a in test.items()],out,'test-stacked',['modality','cell_type'])
paired_report(model,test['rna'],test['atac'],'rna','atac',out,label='cell_type',k=15)
test['rna'].obs.cell_type.value_counts().to_csv(out/'test-celltype-support.csv')
```

## Cross-reconstruction is evaluated in normalized RNA and LSI spaces

```python
for source,target in [('rna','atac'),('atac','rna')]:
    coords=joint.obsm['X_umap'][joint.obs.modality.eq(target)]
    cross_panels(model,test[source],test[target],source,target,coords,
        list(test[target].var_names[:4]),'cell_type',out,f's3-{source}-to-{target}')
    table=pd.read_csv(out/f's3-{source}-to-{target}-features.csv')
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    axs[0].hist(table.pearson,bins=40);axs[0].set_xlabel('Per-feature Pearson r')
    axs[1].hist(table.mse,bins=40);axs[1].set_xlabel('Per-feature MSE')
    fig.savefig(out/f's3-{source}-to-{target}-distributions.png',dpi=150,bbox_inches='tight');plt.close(fig)
```

## Interpret the weaker reconstruction regime

The publication reports 3,139 test pairs across 22 populations, FOSCTTM 0.0546, and relatively modest mean per-feature cross-reconstruction correlations (0.184 for RNA→LSI and 0.132 for ATAC→RNA). Good lineage alignment does not imply precise cell-level recovery of every feature.

Inspect per-class support and errors among neighboring hair-lineage states. Small anchor populations and continuous differentiation can lower macro-F1 without making the entire manifold uninterpretable. The tutorial writes cell counts and per-class metrics to support that distinction.

The supplement contains inconsistent split prose: its methods mention 75/10/15, while its table and the visible notebook use 80/10/10. The tutorial follows the code and accepts an archived barcode map to remove ambiguity. It does not infer lineage directionality or developmental causality from UMAP proximity alone.
