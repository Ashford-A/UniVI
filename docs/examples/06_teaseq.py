# %% [markdown]
# # Figure 6 and S6: trimodal integration with well 5 held out
# Supply raw RNA/ADT counts and a common ATAC tile matrix with obs['well'].
# Wells 3, 4, 6 are within-run training partitions, not independent patients.
# %%
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
# %% [markdown]
# ## Pairwise alignment on held-out well 5
# %%
for a,b in combinations(test,2):paired_report(model,test[a],test[b],a,b,out,k=15)
joint=draw_embedding(model,[('well5',m,a) for m,a in test.items()],out,'well5-stacked',['modality'])
# %% [markdown]
# ## Cluster composition and same/cross-modality neighborhood distances
# %%
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
# %% [markdown]
# ## Main and supplemental lineage-marker views in one joint coordinate frame
# Unimodal Leiden labels are computed in native model input spaces for S6A.
# %%
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
