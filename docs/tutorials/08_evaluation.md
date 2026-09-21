# Figure 8: evaluate integration and biological structure

**Question:** is the representation useful across complementary biological and cell-correspondence diagnostics?

Figure 8 compares integration methods rather than introducing a new biological dataset. This tutorial runs the UniVI evaluation arm and exports the metric families underlying all ten panels. The archived Python and R benchmark notebooks retain the full competitor-specific setup and cross-validation orchestration.

| Panels | Metric family | Desired direction |
| --- | --- | --- |
| A | Fused-space held-out label transfer | Higher |
| B–C | Fused-space k-means ARI / NMI | Higher |
| D | Ground-truth-label silhouette | Context dependent; higher separation |
| E | Model-fit wall time | Lower at comparable performance |
| F–G | Paired FOSCTTM / Recall@10 | Lower / higher |
| H–I | Cross-modality label transfer accuracy / macro-F1 | Higher |
| J | Stacked modality mixing | Compare with representation and composition |

**Inputs:** the annotated Multiome counts under `data/tutorials/benchmark/`, with `cell_type`. Refit all learned transforms per split. The three example seeds demonstrate evaluation mechanics; they do not constitute the published complete cross-validation sweep.

The source runner sets a 30-dimensional latent, `beta=1.35`, `gamma=3.75`, and 200 epochs. This tutorial keeps those code-derived defaults while exposing the preprocessing and evaluation boundaries. Only methods with comparable train/test scope, input measurements, and label access should appear on a common ranking.
**Notebook reference:** [UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb), zero-based cells 18–29, evaluation runner and cross-validation cells. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-8.png
:alt: Published UniVI Figure 8, published figure reference
:class: published-figure

Published Figure 8, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/08_evaluation.ipynb>` · {download}`Python script <../examples/08_evaluation.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from univi.evaluation import encode_fused_adata_pair
from univi.preprocessing import RNAPreprocessor,ATACPreprocessor,split_by_label
root,out=paths('benchmark')
data=read_paired(root,['rna','atac'])
rows=[]
```

## Refit preprocessing independently for each split and seed
Use archived barcode maps for exact fold correspondence with published baselines.

```python
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
```

## Compare only compatible metrics
An absent cross-modality embedding is NA, never a fabricated zero. Three seeds
demonstrate the workflow; they do not reproduce the manuscript's full CV sweep.

```python
metrics=['fused_knn_accuracy','fused_ari','fused_nmi','fused_silhouette','fit_seconds',
         'foscttm','recall10','cross_knn_accuracy','cross_knn_macro_f1','stacked_mixing']
fig,axs=plt.subplots(2,5,figsize=(17,7))
for ax,metric in zip(axs.ravel(),metrics):
    ax.scatter(np.arange(len(results)),results[metric]);ax.set(title=metric,xlabel='Seed')
fig.tight_layout();fig.savefig(out/'fig08-univi-metric-families.png',dpi=150);plt.close(fig)
```

## Avoid misleading comparisons

A method may produce only a single fused representation. Its modality-specific FOSCTTM or cross-latent retrieval is then unavailable; record NA. Duplicating its fused coordinates and calling them RNA and ATAC would create artificially perfect pairing.

The script preserves directional paired-transfer metrics in its JSON outputs. Its summary CSV selects RNA→ATAC explicitly. For a published comparison, apply the same bidirectional reduction rule to every method and retain the direction-specific values.

Fit-time values exclude data acquisition and preprocessing. CUDA is synchronized for timing here; another backend or pipeline boundary needs an equally explicit timing convention. A different machine's wall time is not directly comparable to the published benchmark table.

For the complete historical comparator analysis, follow the three Figure 8 notebooks in the [notebook map](../reference/notebook-provenance.md), including the merge/plot notebook. Figures 9–10 and S8–S11 remain linked as robustness and scaling material, outside this biological tutorial series.
