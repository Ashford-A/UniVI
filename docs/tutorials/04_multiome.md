# Figure 4 and S2: paired RNA–ATAC integration

**Biological question:** does sparse chromatin accessibility align with expression at cell and lineage resolution?

Fit UniVI to paired 10x Multiome PBMCs. RNA enters as training-selected, log-normalized HVGs. ATAC enters as a training-fitted TF-IDF/LSI representation. Modality-specific posterior means support paired retrieval and cross-modal label transfer; their stacked embedding supports visual inspection.

| Panels | Analysis |
| --- | --- |
| 4A–B | Stacked latent geometry by immune identity and modality |
| 4C,E,G,I | ATAC→RNA reconstruction of CD79A, LYZ, NKG7, TRAC |
| 4D,F,H,J | RNA→ATAC reconstruction in LSI coordinates |
| 4K–L | Directional label-transfer confusion, `k=3` |
| S2 | An additional 3D view of the same high-dimensional latent data |

**Inputs:** `data/tutorials/multiome/rna.h5ad`, `atac.h5ad`, raw counts, and `obs["cell_type"]`. Distinguish this annotated dataset from the separate reference used in Figure 5.

The visible notebook call fits **100 LSI components and retains component 0**. The tutorial follows that call; it does not substitute the README's common 101-then-drop-first recipe. RNA is not Z scaled in the visible transformation function. See the [provenance audit](../reference/notebook-provenance.md) before interpreting differences from the supplement's general description.
**Notebook reference:** [UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb), zero-based cells 15–22, 27–38, 51–68, 3D visualization cells. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-4.png
:alt: Published UniVI Figure 4, published figure reference
:class: published-figure

Published Figure 4, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s2.png
:alt: Published UniVI Figure s2, published figure reference
:class: published-figure

Published Figure s2, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/04_multiome.ipynb>` · {download}`Python script <../examples/04_multiome.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from univi.preprocessing import RNAPreprocessor, ATACPreprocessor, split_by_label
root,out=paths('multiome')
data=read_paired(root,['rna','atac'])
splits=split_map(data,root,lambda:split_by_label(data['rna'].obs['cell_type'],max_per_label=5000,seed=0))
# Notebook cell 22 fits 100 LSI dimensions and keeps the first; it does not fit 101 and drop one.
prep={'rna':RNAPreprocessor(2000,scale=False),
      'atac':ATACPreprocessor(8 if SMOKE else 100,drop_first=False,random_state=42)}
parts=preprocess_parts(data,splits,prep)
model,loaders,meta=train_reference('multiome',parts,out,preprocessors=prep)
test=parts['test']
```

## Paired geometry and cross-modal label transfer

```python
joint=draw_embedding(model,[('Multiome','rna',test['rna']),('Multiome','atac',test['atac'])],
    out,'test-stacked',['cell_type','modality'])
paired_report(model,test['rna'],test['atac'],'rna','atac',out,label='cell_type',k=3)
```

## Decode in both directions and compare in the correct output space

```python
xy=joint.obsm['X_umap']
n=test['rna'].n_obs
cross_panels(model,test['atac'],test['rna'],'atac','rna',xy[:n],
    ['CD79A','LYZ','NKG7','TRAC'],'cell_type',out,'fig04-rna')
cross_panels(model,test['rna'],test['atac'],'rna','atac',xy[n:],
    list(test['atac'].var_names[:4]),'cell_type',out,'fig04-lsi')
```

## S2: a second view of the same high-dimensional latent data
This modern tutorial uses a labeled 3D PCA view; it does not claim it is the archived S2 projection.

```python
from sklearn.decomposition import PCA
xyz=PCA(n_components=3,random_state=0).fit_transform(joint.obsm['X_univi'])
fig=plt.figure(figsize=(10,4))
for i,azimuth in enumerate([35,125],1):
    ax=fig.add_subplot(1,2,i,projection='3d')
    for mod,color in [('rna','#3b8fa5'),('atac','#d97a40')]:
        mask=joint.obs.modality.eq(mod).to_numpy()
        ax.scatter(*xyz[mask].T,c=color,s=3,label=mod)
    ax.view_init(elev=22,azim=azimuth);ax.set(title='Latent PCA, same points',xlabel='PC1',ylabel='PC2',zlabel='PC3');ax.legend()
fig.savefig(out/'s2-latent-pca.png',dpi=170,bbox_inches='tight');plt.close(fig)
```

## What the ATAC decoder predicts

This model's ATAC output has one column per LSI component. A change in `LSI_3` is a change in a learned accessibility coordinate; it is not accessibility at a specific peak or a promoter. The decoder has no peak-level output in this recipe.

The paper's reference test set contains 3,137 paired cells and reports FOSCTTM 0.0479, with label-transfer accuracies near 0.96 in both directions. The tutorial checks the same biological relationships, but altered preprocessing or splits need not yield those values.

The 3D tutorial panel is explicitly a PCA projection of latent coordinates with two camera views. It offers another diagnostic view; paired-retrieval metrics in the full latent space provide the quantitative evidence. For peak-level biological sensitivity analyses, use the [raw-feature S1 notebook tutorial](s1_perturbation.md).
