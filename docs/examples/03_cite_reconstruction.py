# %% [markdown]
# # Figure 3 and S1: cross-modal marker reconstruction
# Run 02_citeseq.py first. All predictions below are for the held-out paired cells.
# %%
from _common import *
from univi.workflows import load_reference
root,out=paths('cite')
model,preprocessors,metadata=load_reference(out/'reference',device=DEVICE)
rna=sc.read_h5ad(out/'rna-test.h5ad');adt=sc.read_h5ad(out/'adt-test.h5ad')
joint=sc.read_h5ad(out/'test-stacked.h5ad')
# Use each target modality's rows of ONE stacked UMAP, never fit a prediction UMAP.
rna_xy=joint.obsm['X_umap'][joint.obs.modality.eq('rna')]
adt_xy=joint.obsm['X_umap'][joint.obs.modality.eq('adt')]
# %% [markdown]
# ## RNA from ADT: main and expanded biological marker panels
# %%
rna_markers=['CD79A','MS4A1','LYZ','FCGR3A','NKG7','GNLY','TRAC','CD3D']
rna_hat=cross_panels(model,adt,rna,'adt','rna',rna_xy,rna_markers,
    'celltype.l1',out,'fig03-rna')
# %% [markdown]
# ## Protein from RNA: keep the actual antibody feature identifiers
# NCAM1/CD3E are gene symbols; Hao antibody IDs include CD56-1/CD3-1 or underscore variants.
# %%
adt_markers=['CD19','CD20','CD14','CD16','CD56-1','CD56_1','CD3-1','CD3_1','CD4','CD8a']
adt_hat=cross_panels(model,rna,adt,'rna','adt',adt_xy,adt_markers,
    'celltype.l1',out,'fig03-adt')
# %% [markdown]
# ## Supplemental S1: summarize finer cell types using the same predictions
# %%
for a,pred,markers,name in [(rna,rna_hat,rna_markers,'rna'),(adt,adt_hat,adt_markers,'adt')]:
    ids=[m for m in markers if m in a.var_names]
    ix=a.var_names.get_indexer(ids)
    for label,values in [('observed',dense(a.X)[:,ix]),('predicted',pred[:,ix])]:
        pd.DataFrame(values,index=a.obs_names,columns=ids).groupby(a.obs['celltype.l2'].astype(str)).mean().to_csv(out/f's1-{name}-{label}-means.csv')
