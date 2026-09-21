# %% [markdown]
# # S3: SHARE-seq mouse skin and continuous differentiation
# Input cells require obs['cell_type']; ATAC peaks must have the mm10 blacklist
# removed upstream (record atac.uns['blacklist_removed']=True after filtering).
# %%
from _common import *
from univi.preprocessing import RNAPreprocessor,ATACPreprocessor,split_by_label
root,out=paths('share')
data=read_paired(root,['rna','atac']);rna,atac=data['rna'],data['atac']
if not atac.uns.get('blacklist_removed',False):
    raise ValueError('Remove ENCODE mm10 blacklist peaks before this tutorial; see preparation guide.')
# %% [markdown]
# ## Dataset-specific QC before the paired split
# %%
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
# %% [markdown]
# ## Test-set geometry and rare-population performance
# %%
joint=draw_embedding(model,[('skin',m,a) for m,a in test.items()],out,'test-stacked',['modality','cell_type'])
paired_report(model,test['rna'],test['atac'],'rna','atac',out,label='cell_type',k=15)
test['rna'].obs.cell_type.value_counts().to_csv(out/'test-celltype-support.csv')
# %% [markdown]
# ## Cross-reconstruction is evaluated in normalized RNA and LSI spaces
# %%
for source,target in [('rna','atac'),('atac','rna')]:
    coords=joint.obsm['X_umap'][joint.obs.modality.eq(target)]
    cross_panels(model,test[source],test[target],source,target,coords,
        list(test[target].var_names[:4]),'cell_type',out,f's3-{source}-to-{target}')
    table=pd.read_csv(out/f's3-{source}-to-{target}-features.csv')
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    axs[0].hist(table.pearson,bins=40);axs[0].set_xlabel('Per-feature Pearson r')
    axs[1].hist(table.mse,bins=40);axs[1].set_xlabel('Per-feature MSE')
    fig.savefig(out/f's3-{source}-to-{target}-distributions.png',dpi=150,bbox_inches='tight');plt.close(fig)
