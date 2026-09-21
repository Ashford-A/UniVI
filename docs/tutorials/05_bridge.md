# Figure 5 and S5: bridge independent cohorts and refine labels

**Biological question:** can a paired Multiome reference place independent RNA-only and ATAC-only cohorts into a useful shared coordinate system?

This tutorial has two separately saved stages. First, fit the reference and project Ding RNA and Satpathy ATAC by encoder inference. Second, optionally add a cell-type head and refine encoders using harmonized query labels, with the generative decoders frozen and the paired reference replayed during optimization.

| Panels | Stage and output |
| --- | --- |
| 5A–C | Unsupervised reference projection, colored by cohort, coarse label, technology |
| 5D–F | Corresponding views after supervised refinement |
| 5G | Directional query-to-query transfer before and after refinement |
| 5H–J | Reference-cell annotation and marker validation |
| S5 | Predicted label, maximum class probability, expanded marker views |

**Inputs:** paired `rna.h5ad`/`atac.h5ad`, `ding_rna.h5ad`, `satpathy_atac.h5ad` under `data/tutorials/bridge/`. Queries need `celltype_harmonized` and `technology` metadata. ATAC query counts must be quantified in the reference peak universe or share its retained peaks exactly. A same-width matrix from a different peak set or an independently fitted LSI basis is not compatible.

The modern tutorial selects HVGs from reference training counts on a shared measurable gene panel. The visible original notebook also contains a combined reference/query HVG-selection cell that feeds its later preprocessing call. We make this training-only modernization explicit rather than claiming identical reproduction of that exploratory execution path.

The new public API replaces manual head injection and parameter-name loops. Warmup updates only the head; the second stage updates relevant encoders and the head. The replay term uses the original paired generative objective. Decoder freezing fixes the decoder function in latent coordinates; changing encoders can still change a cell's decoded profile.
**Notebook reference:** [UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb), zero-based cells 18, 20–23, 36–44, 63–68, 125–148. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-5.png
:alt: Published UniVI Figure 5, supplied manuscript reference
:class: published-figure

Published Figure 5, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

```{figure} ../_static/figures/figure-s5.png
:alt: Published UniVI Figure s5, supplied manuscript reference
:class: published-figure

Published Figure s5, reproduced from the supplied manuscript or supplement as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/05_bridge.ipynb>` · {download}`Python script <../examples/05_bridge.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from copy import deepcopy
from univi import ClassHeadConfig
from univi.preprocessing import RNAPreprocessor, ATACPreprocessor, split_by_label
from univi.refinement import UniVIRefiner, RefinementConfig, predict_heads_adata
root,out=paths('bridge')
data=read_paired(root,['rna','atac'])
ding=read_counts(root/'ding_rna.h5ad')
satpathy=read_counts(root/'satpathy_atac.h5ad')
# Shared gene IDENTIFIERS can define the measurable panel without fitting query values.
shared=data['rna'].var_names.intersection(ding.var_names,sort=False)
data['rna']=data['rna'][:,shared].copy();ding=ding[:,shared].copy()
splits=split_map(data,root,lambda:random_split(data['rna'].n_obs,train=.9,val=.1))
prep={'rna':RNAPreprocessor(2000,scale=True),
      'atac':ATACPreprocessor(8 if SMOKE else 100,drop_first=False)}
parts=preprocess_parts(data,splits,prep)
base,loaders,meta=train_reference('bridge',parts,out,preprocessors=prep)
queries={'Ding':('rna',prep['rna'].transform(ding)),
         'Satpathy':('atac',prep['atac'].transform(satpathy))}
reference={mod:prep[mod].transform(a) for mod,a in data.items()}
blocks=[('Multiome',mod,a) for mod,a in reference.items()]+[(cohort,mod,a) for cohort,(mod,a) in queries.items()]
```

## Parameter-frozen projection
Forward inference alone is sufficient for mapping. No query optimizer is run here.

```python
before=draw_embedding(base,blocks,out,'before-refinement',['block','celltype_harmonized','technology'])
```

## Build a training-only label vocabulary and separate cohort loaders
Unknown labels are masked (-1). Hold out labels BEFORE fitting a head.

```python
exclude={'Unknown','Unassigned','General PBMC','Megakaryocyte'}
query_splits={cohort:split_by_label(a.obs.celltype_harmonized.fillna('Unknown'),seed=0)
              for cohort,(mod,a) in queries.items()}
classes=sorted({str(label) for cohort,(mod,a) in queries.items()
    for label in a.obs.iloc[query_splits[cohort]['train']].celltype_harmonized.dropna()
    if str(label) not in exclude})
label_to_id={label:i for i,label in enumerate(classes)}
ft=deepcopy(base)
ft.add_classification_head(ClassHeadConfig('celltype',len(classes),hidden_dims=[64,64,32],
    dropout=.1,batchnorm=False,layernorm=True),label_names=classes)
ft.freeze_decoders()
train_loaders=[];val_loaders=[]
for cohort,(mod,a) in queries.items():
    codes=a.obs.celltype_harmonized.map(label_to_id).fillna(-1).to_numpy(dtype=int)
    for key,target in [('train',train_loaders),('val',val_loaders)]:
        idx=query_splits[cohort][key]
        target.append(make_loader({mod:a[idx].copy()},labels={'celltype':codes[idx]},
            batch_size=16 if SMOKE else 256,shuffle=key=='train',drop_last=key=='train' and len(idx)%256==1))
```

## Head warmup, then small-learning-rate encoder updates with paired replay
The generative replay constrains the reference while decoder parameters remain frozen.
This explicit LayerNorm head follows the supplemental architecture description;
the visible Figure 5 source cell instead leaves its head architecture at defaults.

```python
refiner=UniVIRefiner(ft,train_loaders,val_loaders,device=DEVICE,replay_loader=loaders['train'],
    config=RefinementConfig(max_epochs=2 if SMOKE else 1000,warmup_epochs=1 if SMOKE else 300,
        lr_head=3e-4,lr_encoder=1e-5,weight_decay_head=1e-4,weight_decay_encoder=0.,
        replay_weight=2.,latent_weight=0.))
fit=refiner.fit()
meta['refinement_splits']={cohort:{key:queries[cohort][1].obs_names[idx].tolist()
    for key,idx in split.items()} for cohort,split in query_splits.items()}
meta['refinement_best_epoch']=fit['best_epoch']
save_reference(out/'refined-reference',ft,preprocessors=prep,metadata=meta)
pd.DataFrame(fit['history']).to_csv(out/'refinement-history.csv',index=False)
after=draw_embedding(ft,blocks,out,'after-refinement',['block','celltype_harmonized','technology'])
```

## Compare held-out cross-cohort transfer before and after refinement
Independent cohorts are not paired; FOSCTTM has no meaning between them.

```python
for stage,m in [('before',base),('after',ft)]:
    for source,target in [('Ding','Satpathy'),('Satpathy','Ding')]:
        sm,sa=queries[source];tm,ta=queries[target]
        sa=sa[query_splits[source]['test']].copy();ta=ta[query_splits[target]['test']].copy()
        sa=sa[sa.obs.celltype_harmonized.isin(classes)].copy();ta=ta[ta.obs.celltype_harmonized.isin(classes)].copy()
        transfer_report(latent(m,sa,sm),sa.obs.celltype_harmonized.astype(str),
            latent(m,ta,tm),ta.obs.celltype_harmonized.astype(str),out,f'{stage}-{source}-to-{target}')
```

## Predict labels and validate markers in the unlabeled reference (S5)
Confidence here is maximum softmax probability, not a calibrated certainty.

```python
rna=reference['rna']
prob=predict_heads_adata(ft,rna,'rna',device=DEVICE)['celltype']
rna.obs['predicted_celltype']=pd.Categorical(np.asarray(classes)[prob.argmax(1)])
rna.obs['max_probability']=prob.max(1)
rna.obsm['X_univi']=latent(ft,rna,'rna')
sc.pp.neighbors(rna,use_rep='X_univi');sc.tl.umap(rna,random_state=42)
markers=[g for g in ['MS4A1','CD79A','FCGR3A','LST1','IL7R','TRAC','CD8A','NKG7','S100A8','LYZ','CLEC10A'] if g in rna.var_names]
sc.pl.umap(rna,color=['predicted_celltype','max_probability']+markers,layer='log1p',show=False)
plt.savefig(out/'s5-labels-confidence-markers.png',dpi=150,bbox_inches='tight');plt.close('all')
if markers:
    sc.pl.dotplot(rna,markers,groupby='predicted_celltype',layer='log1p',show=False)
    plt.savefig(out/'fig05-marker-validation.png',dpi=150,bbox_inches='tight');plt.close('all')
rna.write_h5ad(out/'reference-predicted-labels.h5ad')
```

## Evaluate refinement without circularity

Hold out labeled query cells before training. The script evaluates query-to-query transfer on these held-out cells and writes their barcodes. Marker enrichment in the unlabeled Multiome reference is useful biological corroboration, but expression markers and the model inputs are related evidence.

Do not compute FOSCTTM between Ding and Satpathy: those cells are not matched biological pairs. Compare immune identity, donor/platform composition, neighborhood structure, and held-out labels instead. The published refinement particularly improved Satpathy→Ding transfer; it did not uniformly improve every direction by the same amount.

The example uses the `[64,64,32]` LayerNorm head described in the supplement. The visible original Figure 5 head configuration leaves architecture fields at package defaults. This is an explicitly documented tutorial choice. Both the base and refined references remain available for comparison. See [refinement mechanics](../guides/refinement.md).
