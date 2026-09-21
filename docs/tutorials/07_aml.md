# Figure 7 and S7: AML mosaic integration and mutation-aware refinement

**Biological question:** can paired RNA–protein anchors connect independent AML cohorts and expose genotype-associated and differentiation-related structure?

Train a bridge on paired Knorr AML CITE-seq. Project van Galen RNA and DAb-seq proteins with their respective encoders. Genotypes are optional supervised targets and validation annotations; they are not falsely presented as a third measurement available in every cell.

| Panel family | Analysis |
| --- | --- |
| 7A,D | Cohort and van Galen cell-state structure before refinement |
| 7B–F | Observed NPM1 calls and directional neighborhood-probability transfer |
| 7G–K | Refined geometry, observed NPM1 status, and LSC17 expression signature |
| 7L | Held-out mutation AUC/AP, with per-gene label denominators |
| S7 | Patient/disease context and predicted driver landscapes over all cells |

**Inputs:** paired `rna.h5ad`/`adt.h5ad`, `van_galen_rna.h5ad`, `dab_adt.h5ad`, and a patient/sample-level bridge `splits.tsv` under `data/tutorials/aml/`. Reference `sample_id` must identify the actual grouping unit. Query `mut_GENE` columns contain `0`, `1`, or missing; cell-state and sample fields remain metadata. Canonicalize antibodies and document raw versus already processed DAb values before applying a reference transform.

The bridge uses all shared genes rather than silently replacing them with 2,000 HVGs. Per-gene binary heads use `[64,32]` hidden layers, LayerNorm, dropout 0.1, training-only class weights, 10 warmup epochs, head LR 1e-4, encoder LR 1e-5, and latent-preservation weight 5.0, following the visible refinement call. Missing mutation calls never enter a gene's supervised loss.
**Notebook reference:** [UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb), zero-based cells 17–39, 55–63, 75–88, 107–128. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-7.png
:alt: Published UniVI Figure 7, supplied manuscript reference
:class: published-figure

Published Figure 7, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s7.png
:alt: Published UniVI Figure s7, supplied manuscript reference
:class: published-figure

Published Figure s7, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/07_aml.ipynb>` · {download}`Python script <../examples/07_aml.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
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
```

## Genotype labels and cell-level evaluation partitions
Do not convert absence of a mutation call to wild type. These modern splits
are transparent random cell splits; reuse archived maps for paper replication.

```python
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
```

## Genotype neighborhood transfer without a mutation-trained encoder
Only observed labels in source training cells define neighbor probabilities.

```python
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
```

## Attach one binary head per supported gene
Positive weights use only observed training labels. Genes without two training
classes are excluded explicitly; they are never presented as learned predictors.

```python
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
```

## Held-out mutation AUC/AP with explicit denominators
A cell split supports within-cohort evaluation, not unseen-patient generalization.

```python
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
```

## LSC17 expression signature and all-cell predicted landscapes (S7)
This is the mean-expression signature used in the notebook, not the clinical
weighted prognostic assay. DAb values below derive from imputed RNA.

```python
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
```

## Separate measurement, prediction, and supervision

The saved table reports the number of labeled test cells, the positive count, prevalence, AUC, and average precision for each cohort/gene. AUC is undefined for a one-class test set; sparse genes are explicitly marked unevaluable. Average precision should be read relative to prevalence. The published dense DAb targets perform better than several sparse van Galen targets; annotation quality and label availability matter.

A missing call is not wild type. In targeted transcript-derived genotyping, even an apparent negative can mean failure to detect a mutant transcript. Patient-level labels attached to all cells also have a different meaning from measured cell-level genotype. Record which type of evidence each column represents.

The example's mutation-head split is at cell level, matching the scope discussed in the supplement. Its performance is not a new-patient estimate. A stronger future study would group mutation-head splits by patient and evaluate unseen patients with adequate positives/negatives; do not silently replace a failed patient split with a cell split.

The LSC17 analysis here is the arithmetic mean of available panel genes in log-normalized expression, with a within-cohort Z score. It is not the clinically weighted LSC17 prognostic assay. ADT-only cells receive an explicitly labeled **imputed-RNA signature**. Their signature and mutation-head probabilities are model outputs, not independent molecular measurements.

After mutation supervision, better genotype separation is expected from the objective. The most informative checks are held-out labels, preserved unsupervised biology, and comparison with the frozen-projection baseline. [Refinement guide](../guides/refinement.md).
