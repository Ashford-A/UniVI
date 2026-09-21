# Figure 6 and S6: trimodal TEA-seq with a held-out well

**Biological question:** do RNA, surface protein, and chromatin accessibility describe concordant immune compartments when all three are measured?

Train using wells 3, 4, and 6 and evaluate well 5. The three assays enter one model through separate encoders and likelihood-appropriate decoders. Evaluate all three paired relationships rather than selecting only the best pair.

| Panels | Tutorial analysis |
| --- | --- |
| 6A | RNA–ADT, RNA–ATAC, and ADT–ATAC FOSCTTM |
| 6B–C | Cross-modality neighbor fractions, entropy, and distance distributions |
| 6D–F | Stacked UMAP, Leiden communities, modality composition per community |
| 6G | Cytotoxic-lymphocyte RNA/protein markers and representative LSI coordinates |
| S6 | Native-space cluster labels and B-cell, T-cell, myeloid/DC markers |

**Inputs:** paired `rna.h5ad`, `adt.h5ad`, and `atac.h5ad` under `data/tutorials/tea/`, with `obs["well"]` normalized to strings `3`, `4`, `5`, `6`. Provide a common ATAC tile universe; the source notebook constructs this using SnapATAC2. When `n_fragments` is supplied, the tutorial applies the notebook's threshold of 1,500.

A well holdout tests a capture/library partition within a run. It does not establish robustness to a new person or laboratory. The tutorial fits HVGs, scales, TF-IDF and LSI using training cells alone, then applies them unchanged to validation and well 5. This is stricter than several visible original preprocessing cells, including per-well ADT standardization.

The recipe uses notebook `gamma=1.45` and RNA/ADT/ATAC reconstruction weights 1.0/2.0/1.45. The supplemental table lists a different gamma. All these settings remain visible in the shared example helper.
**Notebook reference:** [UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb), zero-based cells 10–20, 29–37, 53–80, 101–121, 126–133. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-6.png
:alt: Published UniVI Figure 6, published figure reference
:class: published-figure

Published Figure 6, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s6.png
:alt: Published UniVI Figure s6, published figure reference
:class: published-figure

Published Figure s6, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/06_teaseq.ipynb>` · {download}`Python script <../examples/06_teaseq.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from univi.preprocessing import RNAPreprocessor,ADTPreprocessor,ATACPreprocessor
from univi.evaluation import compute_modality_entropy
root,out=paths('tea')
data=read_paired(root,['rna','adt','atac'])
if 'n_fragments' in data['atac'].obs:
    keep=data['atac'].obs.n_fragments>=1500
    data={m:a[keep].copy() for m,a in data.items()}
well=data['rna'].obs['well'].astype(str).to_numpy()
ref=np.flatnonzero(np.isin(well,['3','4','6']));heldout=np.flatnonzero(well=='5')
if len(heldout)==0:raise ValueError('No well 5 cells: check the well metadata mapping.')
local=random_split(len(ref),train=.8,val=.1,seed=42)
splits={'train':ref[local['train']],'val':ref[local['val']],'test':heldout}
# Modern strict holdout: all fitted transforms use only optimization-training cells.
prep={'rna':RNAPreprocessor(2000,scale=True,normalize_on_selected=False),
      'adt':ADTPreprocessor(scale=True),
      'atac':ATACPreprocessor(8 if SMOKE else 100,method='tea',drop_first=False)}
parts=preprocess_parts(data,splits,prep)
model,loaders,meta=train_reference('tea',parts,out,preprocessors=prep)
test=parts['test']
```

## Pairwise alignment on held-out well 5

```python
for a,b in combinations(test,2):paired_report(model,test[a],test[b],a,b,out,k=15)
joint=draw_embedding(model,[('well5',m,a) for m,a in test.items()],out,'well5-stacked',['modality'])
```

## Cluster composition and same/cross-modality neighborhood distances

```python
sc.tl.leiden(joint,resolution=.7,key_added='leiden',random_state=42,flavor='igraph',n_iterations=2)
composition=pd.crosstab(joint.obs.leiden,joint.obs.modality,normalize='index')
composition.to_csv(out/'cluster-modality-composition.csv')
composition.plot.bar(stacked=True,figsize=(8,4));plt.ylabel('Fraction of stacked points')
plt.savefig(out/'cluster-composition.png',dpi=150,bbox_inches='tight');plt.close('all')
z=joint.obsm['X_univi'];mods=joint.obs.modality.to_numpy();k=min(30,len(z)-1)
d,idx=NearestNeighbors(n_neighbors=k+1).fit(z).kneighbors(z)
d,idx=d[:,1:],idx[:,1:];same=mods[:,None]==mods[idx]
fig,ax=plt.subplots();ax.hist(d[same],bins=40,alpha=.5,density=True,label='same modality')
ax.hist(d[~same],bins=40,alpha=.5,density=True,label='different modality');ax.legend();ax.set_xlabel('Latent Euclidean distance')
fig.savefig(out/'neighbor-distances.png',dpi=150,bbox_inches='tight');plt.close(fig)
(out/'trimodal-mixing.json').write_text(json.dumps({'mixing':float((~same).mean()),
    'entropy':compute_modality_entropy(z,mods,k=k)},indent=2))
```

## Main and supplemental lineage-marker views in one joint coordinate frame
Unimodal Leiden labels are computed in native model input spaces for S6A.

```python
panels={'rna':['GZMH','GNLY','CCL5','MS4A1','BANK1','PAX5','CD74','HLA-DRA','IGKC','IGLC1','IGHM','IGHA1','IGHA2','LEF1','BCL11B','RUNX3','LYZ','TNFAIP3','ITGAX','ITGAD'],
        'adt':['CD56','CD16','KLRG1','CD19','CD21','CD24','IgD','IgM','CD38','CD3','CD4','CD45RA','CD45RO','CD27','CD127','CD279','CD14','CD11b','CD11c','HLA-DR','CD141','CD172a','CD192','CD304'],
        'atac':list(test['atac'].var_names[:4])}
for mod,a in test.items():
    mask=joint.obs.modality.eq(mod).to_numpy();a.obsm['X_umap']=joint.obsm['X_umap'][mask]
    # Native-space cluster annotations, then display at already computed latent coordinates.
    sc.pp.neighbors(a,use_rep='X',n_neighbors=min(15,a.n_obs-1))
    sc.tl.leiden(a,key_added='native_leiden',random_state=42,flavor='igraph',n_iterations=2)
    markers=[m for m in panels[mod] if m in a.var_names]
    sc.pl.umap(a,color=['native_leiden']+markers,show=False)
    plt.savefig(out/f's6-{mod}-markers.png',dpi=150,bbox_inches='tight');plt.close('all')
sc.pl.umap(joint,color=['modality','leiden'],show=False)
plt.savefig(out/'fig06-joint-clusters.png',dpi=150,bbox_inches='tight');plt.close('all')
joint.write_h5ad(out/'well5-clusters.h5ad')
# Optional RNA -> ADT/LSI predictions, kept separate from the measured marker panels.
for mod in ['adt','atac']:
    cross_panels(model,test['rna'],test[mod],'rna',mod,test[mod].obsm['X_umap'],
        panels[mod][:4],None,out,f'well5-rna-to-{mod}')
```

## Compare biology and modality composition together

A balanced cluster composition is useful only if immune identities remain meaningful. Inspect corresponding RNA and ADT marker families in the same coordinates. The expected random different-modality neighbor fraction is approximately 2/3 for three equally represented modalities; entropy is a separate diagnostic.

Do not force the tutorial to produce exactly the paper's 14 Leiden clusters. Resolution, graph construction, seed, and the input checkpoint determine granularity. Record these choices and ask whether lineage-level structure persists.

Each well 5 cell contributes three stacked points; they are correlated views of one cell, not three independent samples. S6's native-space clustering is an orthogonal diagnostic overlay, and is not a source of supervised training labels. LSI overlays describe accessibility geometry, not named regulatory peaks.
