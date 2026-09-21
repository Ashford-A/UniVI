# %% [markdown]
# # Figure 7 and S7: AML mosaic, genotype grounding, and stemness
# Required inputs: paired CITE RNA/ADT; van Galen RNA; DAb-seq ADT, all with raw
# counts in layers['counts']. Canonicalize the shared antibody panel first.
# Query obs columns mut_GENE are 0/1 for observed calls, NaN or -1 for missing.
# %%
from _common import *
from copy import deepcopy
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.neighbors import KNeighborsRegressor
from univi import ClassHeadConfig
from univi.preprocessing import RNAPreprocessor,ADTPreprocessor
from univi.evaluation import add_lsc17_scores,LSC17_GENES
from univi.refinement import RefinementConfig,UniVIRefiner,predict_heads_adata
root,out=paths('aml')
data=read_paired(root,['rna','adt'])
vg=read_counts(root/'van_galen_rna.h5ad');dab=read_counts(root/'dab_adt.h5ad')
genes=data['rna'].var_names.intersection(vg.var_names,sort=False)
proteins=data['adt'].var_names.intersection(dab.var_names,sort=False)
if len(proteins)<2:raise ValueError('Canonicalize antibody names before computing the shared panel.')
data['rna']=data['rna'][:,genes].copy();vg=vg[:,genes].copy()
data['adt']=data['adt'][:,proteins].copy();dab=dab[:,proteins].copy()
# Enforce an explicit patient/sample split for the paired bridge.
if not (root/'splits.tsv').exists():raise FileNotFoundError('AML requires a patient-group bridge splits.tsv (see data guide).')
splits=split_map(data,root,lambda:None)
for a,b in [('train','val'),('train','test'),('val','test')]:
    assert set(data['rna'].obs.iloc[splits[a]].sample_id).isdisjoint(data['rna'].obs.iloc[splits[b]].sample_id)
prep={'rna':RNAPreprocessor(None,scale=True,normalize_on_selected=False),
      'adt':ADTPreprocessor(scale=True,clip=10)}
parts=preprocess_parts(data,splits,prep)
base,loaders,meta=train_reference('aml',parts,out,preprocessors=prep,seed=1)
queries={'vanGalen':('rna',prep['rna'].transform(vg)),'DAb':('adt',prep['adt'].transform(dab))}
bridge=parts['test']
# %% [markdown]
# ## Genotype labels and cell-level evaluation partitions
# Do not convert absence of a mutation call to wild type. These modern splits
# are transparent random cell splits; reuse archived maps for paper replication.
# %%
GENES=['NPM1','DNMT3A','FLT3','TP53','NRAS','TET2','IDH2']
qsplit={name:random_split(a.n_obs,seed=1) for name,(mod,a) in queries.items()}
targets={}
for name,(mod,a) in queries.items():
    targets[name]={}
    for gene in GENES:
        col=f'mut_{gene}'
        y=pd.to_numeric(a.obs[col],errors='raise').fillna(-1).to_numpy() if col in a.obs else np.full(a.n_obs,-1)
        if not np.isin(y,[-1,0,1]).all():raise ValueError(f'{name}/{gene}: labels must be -1/0/1.')
        targets[name][gene]=y.astype(int)
        a.obs[col]=np.where(y>=0,y,np.nan)
blocks=[('CITE',m,a) for m,a in bridge.items()]+[(name,mod,a) for name,(mod,a) in queries.items()]
joint=draw_embedding(base,blocks,out,'before-refinement',['block','cell_state','mut_NPM1','sample_id'])
# %% [markdown]
# ## Genotype neighborhood transfer without a mutation-trained encoder
# Only observed labels in source training cells define neighbor probabilities.
# %%
for source,target,k in [('vanGalen','DAb',60),('DAb','vanGalen',50)]:
    sm,sa=queries[source];tm,ta=queries[target]
    train=qsplit[source]['train'];y=targets[source]['NPM1'];known=train[y[train]>=0]
    if len(known):
        reg=KNeighborsRegressor(n_neighbors=min(k,len(known)),weights='distance')
        reg.fit(latent(base,sa[known],sm),y[known])
        ta.obs[f'NPM1_knn_from_{source}']=reg.predict(latent(base,ta,tm))
        pd.Series(ta.obs[f'NPM1_knn_from_{source}'],index=ta.obs_names).to_csv(out/f'NPM1-{source}-to-{target}.csv')
        # Plot target probabilities on the SAME baseline coordinates as observed calls.
        column=f'NPM1_knn_from_{source}'
        joint.obs[column]=np.nan
        target_rows=joint.obs['cohort'].eq(target)
        joint.obs.loc[target_rows,column]=joint.obs.loc[target_rows,'cell_id'].map(ta.obs[column]).to_numpy()
        sc.pl.umap(joint,color=column,vmin=0,vmax=1,show=False)
        plt.savefig(out/f'NPM1-{source}-to-{target}.png',dpi=160,bbox_inches='tight');plt.close('all')
joint.write_h5ad(out/'before-refinement-with-knn.h5ad')
# %% [markdown]
# ## Attach one binary head per supported gene
# Positive weights use only observed training labels. Genes without two training
# classes are excluded explicitly; they are never presented as learned predictors.
# %%
ft=deepcopy(base);active=[];coverage=[]
for gene in GENES:
    train_y=np.concatenate([targets[name][gene][qsplit[name]['train']] for name in queries])
    known=train_y[train_y>=0];positive=int((known==1).sum());negative=int((known==0).sum())
    coverage.append({'gene':gene,'train_labeled':len(known),'train_positive':positive,'train_negative':negative})
    if min(positive,negative)==0:continue
    active.append(gene)
    pos_weight=float(np.clip(negative/positive,1,100))
    ft.add_classification_head(ClassHeadConfig(gene,2,head_type='binary',hidden_dims=[64,32],
        dropout=.1,batchnorm=False,layernorm=True,pos_weight=pos_weight),label_names=['WT','MUT'])
pd.DataFrame(coverage).to_csv(out/'training-label-coverage.csv',index=False)
train_loaders=[];val_loaders=[]
for name,(mod,a) in queries.items():
    for split,container in [('train',train_loaders),('val',val_loaders)]:
        idx=qsplit[name][split]
        container.append(make_loader({mod:a[idx]},labels={g:targets[name][g][idx] for g in active},
            batch_size=16 if SMOKE else 128,shuffle=split=='train'))
refiner=UniVIRefiner(ft,train_loaders,val_loaders,device=DEVICE,
    config=RefinementConfig(max_epochs=2 if SMOKE else 1000,warmup_epochs=1 if SMOKE else 10,
        lr_head=1e-4,lr_encoder=1e-5,latent_weight=5.,weight_decay_head=1e-6,
        weight_decay_encoder=1e-6,patience=50))
fit=refiner.fit()
# %% [markdown]
# ## Held-out mutation AUC/AP with explicit denominators
# A cell split supports within-cohort evaluation, not unseen-patient generalization.
# %%
rows=[]
for name,(mod,a) in queries.items():
    p=predict_heads_adata(ft,a,mod,device=DEVICE)
    idx=qsplit[name]['test']
    for gene in active:
        a.obs[f'p_{gene}']=p[gene].ravel()
        y=targets[name][gene];known=idx[y[idx]>=0];yy=y[known];ss=p[gene].ravel()[known]
        valid=len(known)>=10 and len(np.unique(yy))==2
        rows.append({'cohort':name,'gene':gene,'n_labeled':len(known),'n_mut':int((yy==1).sum()),
            'prevalence':float(yy.mean()) if len(yy) else np.nan,
            'auc':roc_auc_score(yy,ss) if valid else np.nan,
            'average_precision':average_precision_score(yy,ss) if valid else np.nan,
            'status':'evaluated' if valid else 'insufficient labels or one class'})
performance=pd.DataFrame(rows);performance.to_csv(out/'mutation-test-performance.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4));sns.barplot(data=performance,x='gene',y='auc',hue='cohort',ax=ax,errorbar=None)
ax.axhline(.5,c='gray',ls='--');ax.set_ylim(0,1);fig.savefig(out/'fig07-mutation-auc.png',dpi=160,bbox_inches='tight');plt.close(fig)
# %% [markdown]
# ## LSC17 expression signature and all-cell predicted landscapes (S7)
# This is the mean-expression signature used in the notebook, not the clinical
# weighted prognostic assay. DAb values below derive from imputed RNA.
# %%
for name,mod,a in blocks:
    p=predict_heads_adata(ft,a,mod,device=DEVICE)
    for gene in active:a.obs[f'p_{gene}']=p[gene].ravel()
    if mod=='rna':
        if set(LSC17_GENES)&set(a.var_names):add_lsc17_scores(a,layer='log1p')
        a.obs['LSC17_source']='observed RNA'
    else:
        pred=cross_modal_predict(ft,a,'adt','rna',device=DEVICE)
        # Undo the TRAIN RNA scaler to put decoded values back in log-normalized units.
        log_pred=prep['rna'].scaler_.inverse_transform(pred)
        proxy=ad.AnnData(log_pred,obs=a.obs.copy(),var=pd.DataFrame(index=prep['rna'].features_))
        if set(LSC17_GENES)&set(proxy.var_names):
            add_lsc17_scores(proxy)
            for col in ['LSC17_score','LSC17_z','LSC17_n_genes']:a.obs[col]=proxy.obs[col].to_numpy()
        a.obs['LSC17_source']='ADT-imputed RNA'
refined=draw_embedding(ft,blocks,out,'after-refinement',
    ['block','cell_state','mut_NPM1','LSC17_z','sample_id','disease_status']+[f'p_{g}' for g in active])
meta['refinement_split_type']='cell-level; not patient holdout'
meta['refinement_splits']={name:{key:queries[name][1].obs_names[idx].tolist() for key,idx in split.items()} for name,split in qsplit.items()}
meta['active_genes']=active;meta['refinement_best_epoch']=fit['best_epoch']
save_reference(out/'refined-reference',ft,preprocessors=prep,metadata=meta)
