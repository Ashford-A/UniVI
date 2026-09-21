# %% [markdown]
# # Supplemental Notebook S1: raw-feature decoding and in-silico peak sensitivity
# This is distinct from Supplemental Figure S1 (CITE-seq markers).
# The model uses raw RNA counts and ATAC peak counts, not LSI features.
# %%
from _common import *
from univi.preprocessing import split_by_label
from univi.perturbation import predict_feature_perturbation
root,out=paths('raw')
data=read_paired(root,['rna','atac'])
for a in data.values():a.X=a.layers['counts'].copy()
splits=split_map(data,root,lambda:split_by_label(data['rna'].obs.cell_type,seed=0))
parts={key:{m:a[idx].copy() for m,a in data.items()} for key,idx in splits.items()}
# %% [markdown]
# ## Build the count-space model shown in S1's configuration cell 17
# Public NB/Poisson decoders make the returned cross-predictions count-like means.
# %%
rna_hidden=[16,8] if SMOKE else [1024,512,256,128]
atac_hidden=[16,8] if SMOKE else [2048,1024,512,256,128]
cfg=UniVIConfig(latent_dim=4 if SMOKE else 40,beta=.85,gamma=1.75,
    encoder_dropout=.075,decoder_dropout=.035,encoder_batchnorm=False,decoder_batchnorm=False,
    kl_anneal_start=0,kl_anneal_end=100,align_anneal_start=30,align_anneal_end=130,
    modalities=[ModalityConfig('rna',data['rna'].n_vars,rna_hidden,rna_hidden[::-1],likelihood='nb'),
                ModalityConfig('atac',data['atac'].n_vars,atac_hidden,atac_hidden[::-1],likelihood='poisson')])
model=UniVIMultiModalVAE(cfg,loss_mode='v1',v1_recon='avg',normalize_v1_terms=True).to(DEVICE)
tcfg=TrainingConfig(n_epochs=int(os.environ.get('UNIVI_EPOCHS',5000)),lr=1e-3,weight_decay=1e-4,
    batch_size=16 if SMOKE else 128,device=DEVICE,early_stopping=True,patience=200,
    best_epoch_warmup=0 if SMOKE else 10,grad_clip=5.)
trainer=UniVITrainer(model,make_loader(parts['train'],batch_size=tcfg.batch_size,shuffle=True),
    make_loader(parts['val'],batch_size=tcfg.batch_size),tcfg,device=DEVICE)
trainer.fit()
save_reference(out/'reference',model,metadata={'features':{m:a.var_names.tolist() for m,a in data.items()},
    'split':{k:parts[k]['rna'].obs_names.tolist() for k in parts},'input_space':'raw counts'})
# %% [markdown]
# ## Select a gene-associated peak set from an explicit, versioned link table
# The table is for interpretive perturbations; it was not an alignment requirement.
# Use a matching genome build and consistent genomic coordinate convention.
# %%
links=pd.read_csv(root/'peak_gene_links.tsv',sep='\t')
target_genes=['GNLY']
features=links.loc[links.gene_name.isin(target_genes),'peak_name'].drop_duplicates().tolist()
features=[p for p in features if p in data['atac'].var_names]
if not features:raise ValueError('No linked target peaks: inspect the annotation/link table.')
query=parts['test']['atac']
result=predict_feature_perturbation(model,query,source_modality='atac',target_modality='rna',
    features=features,mode='off',device=DEVICE)
# %% [markdown]
# ## Summarize per-cell changes by annotated population
# Mean of per-cell decoded changes is deliberately distinguished from decoding a
# pseudobulk profile: nonlinear networks generally make those different estimands.
# %%
delta=pd.DataFrame(result['delta'],index=query.obs_names,columns=data['rna'].var_names)
means=delta.groupby(query.obs.cell_type.astype(str)).mean()
means.to_csv(out/'peak-ablation-rna-delta-by-celltype.csv')
genes_show=delta.mean().abs().nlargest(min(20,delta.shape[1])).index
fig,ax=plt.subplots(figsize=(10,max(3,len(means)*.3)))
sns.heatmap(means[genes_show],center=0,cmap='vlag',ax=ax)
ax.set_title('Model-predicted RNA change after peak ablation')
fig.savefig(out/'s1-peak-ablation.png',dpi=160,bbox_inches='tight');plt.close(fig)
# %% [markdown]
# ## Matched null peak sets, with matching based on TRAINING accessibility
# Match chromosome, width bin, and accessibility-frequency bin. Overlapping
# null candidates are allowed between draws, but each draw contains unique peaks.
# %%
v=data['atac'].var
required={'chrom','chromStart','chromEnd'}
if not required.issubset(v):raise ValueError('ATAC coordinates are required for matched nulls.')
train_counts=parts['train']['atac'].X
freq=np.asarray((train_counts>0).mean(axis=0)).ravel()
width=(v.chromEnd-v.chromStart).to_numpy()
width_bin=np.floor(np.log2(np.maximum(width,1))).astype(int)
freq_bin=np.minimum((freq*10).astype(int),9)
key=np.array([f'{c}:{w}:{f}' for c,w,f in zip(v.chrom.astype(str),width_bin,freq_bin)])
target_idx=v.index.get_indexer(features);rng=np.random.default_rng(0)
nulls=[];n_draws=2 if SMOKE else 100
for draw in range(n_draws):
    chosen=[]
    for ix in target_idx:
        candidates=np.flatnonzero(key==key[ix])
        candidates=np.setdiff1d(candidates,np.r_[target_idx,chosen])
        if not len(candidates):raise ValueError('No matched null candidates; prespecify coarser bins or a wider reference peak universe.')
        chosen.append(int(rng.choice(candidates)))
    null=predict_feature_perturbation(model,query,source_modality='atac',target_modality='rna',
        features=v.index[chosen].tolist(),mode='off',device=DEVICE)
    gene_idx=data['rna'].var_names.get_indexer(target_genes)
    gene_idx=gene_idx[gene_idx>=0]
    if not len(gene_idx):raise ValueError('No target genes in the RNA decoder feature list.')
    nulls.append(float(null['delta'][:,gene_idx].mean()))
observed=float(delta.loc[:,[g for g in target_genes if g in delta]].to_numpy().mean())
# A two-sided empirical tail around zero, with the finite-sample +1 correction.
p=(1+int((np.abs(nulls)>=abs(observed)).sum()))/(1+len(nulls))
(out/'matched-null.json').write_text(json.dumps({'observed_mean_delta':observed,
    'null_mean_deltas':nulls,'empirical_tail_fraction':p,'n_draws':len(nulls)},indent=2))
