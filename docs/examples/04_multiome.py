# %% [markdown]
# # Figure 4 and S2: paired RNA–ATAC integration
# RNA reconstructions are in normalized gene space. ATAC reconstructions are LSI coordinates.
# %%
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
# %% [markdown]
# ## Paired geometry and cross-modal label transfer
# %%
joint=draw_embedding(model,[('Multiome','rna',test['rna']),('Multiome','atac',test['atac'])],
    out,'test-stacked',['cell_type','modality'])
paired_report(model,test['rna'],test['atac'],'rna','atac',out,label='cell_type',k=3)
# %% [markdown]
# ## Decode in both directions and compare in the correct output space
# %%
xy=joint.obsm['X_umap']
n=test['rna'].n_obs
cross_panels(model,test['atac'],test['rna'],'atac','rna',xy[:n],
    ['CD79A','LYZ','NKG7','TRAC'],'cell_type',out,'fig04-rna')
cross_panels(model,test['rna'],test['atac'],'rna','atac',xy[n:],
    list(test['atac'].var_names[:4]),'cell_type',out,'fig04-lsi')
# %% [markdown]
# ## S2: a second view of the same high-dimensional latent data
# This modern tutorial uses a labeled 3D PCA view; it does not claim it is the archived S2 projection.
# %%
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
