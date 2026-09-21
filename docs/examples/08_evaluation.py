# %% [markdown]
# # Figure 8: evaluate biological structure and paired correspondence
# This runs the UniVI arm and exports all ten panel metric families. It does not
# rerun competing tools; the archived Python/R runners remain the source for them.
# %%
from _common import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from univi.evaluation import encode_fused_adata_pair
from univi.preprocessing import RNAPreprocessor,ATACPreprocessor,split_by_label
root,out=paths('benchmark')
data=read_paired(root,['rna','atac'])
rows=[]
# %% [markdown]
# ## Refit preprocessing independently for each split and seed
# Use archived barcode maps for exact fold correspondence with published baselines.
# %%
for seed in ([0] if SMOKE else [0,1,2]):
    run=out/f'seed-{seed}';run.mkdir(exist_ok=True)
    splits=split_by_label(data['rna'].obs.cell_type,seed=seed)
    prep={'rna':RNAPreprocessor(2000,scale=True),
          'atac':ATACPreprocessor(8 if SMOKE else 101,drop_first=True,random_state=seed)}
    parts=preprocess_parts(data,splits,prep)
    model,loaders,meta=train_reference('benchmark',parts,run,preprocessors=prep,seed=seed)
    train,test=parts['train'],parts['test']
    ztr=encode_fused_adata_pair(model,train,device=DEVICE,write_to_adatas=False)['Z_fused']
    zte=encode_fused_adata_pair(model,test,device=DEVICE,write_to_adatas=False)['Z_fused']
    ytr=train['rna'].obs.cell_type.astype(str).to_numpy();yte=test['rna'].obs.cell_type.astype(str).to_numpy()
    _,acc,_=label_transfer_knn(ztr,ytr,zte,yte,k=15)
    groups=KMeans(n_clusters=len(np.unique(yte)),random_state=seed,n_init=10).fit_predict(zte)
    row={'method':'UniVI','seed':seed,'fused_knn_accuracy':acc,
        'fused_ari':adjusted_rand_score(yte,groups),'fused_nmi':normalized_mutual_info_score(yte,groups),
        'fused_silhouette':silhouette_score(zte,yte,sample_size=min(5000,len(yte)),random_state=seed),
        'fit_seconds':meta['fit_seconds'],'scope':'inductive','uses_supervised_labels':False}
    paired=paired_report(model,test['rna'],test['atac'],'rna','atac',run,label='cell_type',k=15)
    row.update(foscttm=paired['rna_to_atac']['foscttm'],recall10=paired['rna_to_atac']['recall10'],
        cross_knn_accuracy=paired['transfer_a_b']['accuracy'],cross_knn_macro_f1=paired['transfer_a_b']['macro_f1'],
        stacked_mixing=paired['stacked_mixing'])
    rows.append(row)
results=pd.DataFrame(rows);results.to_csv(out/'univi-evaluation.csv',index=False)
# %% [markdown]
# ## Compare only compatible metrics
# An absent cross-modality embedding is NA, never a fabricated zero. Three seeds
# demonstrate the workflow; they do not reproduce the manuscript's full CV sweep.
# %%
metrics=['fused_knn_accuracy','fused_ari','fused_nmi','fused_silhouette','fit_seconds',
         'foscttm','recall10','cross_knn_accuracy','cross_knn_macro_f1','stacked_mixing']
fig,axs=plt.subplots(2,5,figsize=(17,7))
for ax,metric in zip(axs.ravel(),metrics):
    ax.scatter(np.arange(len(results)),results[metric]);ax.set(title=metric,xlabel='Seed')
fig.tight_layout();fig.savefig(out/'fig08-univi-metric-families.png',dpi=150);plt.close(fig)
