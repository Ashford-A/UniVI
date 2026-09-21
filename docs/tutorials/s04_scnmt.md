# S4: coverage-aware trimodal scNMT-seq

**Biological question:** can RNA, DNA methylation and chromatin accessibility jointly describe developmental progression without requiring feature-level correspondence?

Use the parsed mouse-gastrulation bundle to build three paired modalities: RNA, genebody CpG methylation, and genebody GpC accessibility. The adapter is promoted from the actual scNMT notebook, including the metadata-based mapping from RNA IDs to shared sample IDs.

| Representation | Encoder input | Reconstruction target |
| --- | --- | --- |
| RNA | Library-normalized log1p counts, no HVG or Z scaling | Gaussian target in the same expression space |
| CpG | Supplied fractions in `.X` | `meth_successes`, `meth_total_count` |
| GpC | Supplied fractions in `.X` | `acc_successes`, `acc_total_count` |

**Input:** `data/tutorials/scnmt/scnmt_gastrulation.tar.gz`, containing `rna/counts.txt.gz`, `met/feature_level/genebody.tsv.gz`, `acc/feature_level/genebody.tsv.gz`, and `sample_metadata.txt`. [Acquisition and schemas](../guides/data.md).

A feature with zero coverage is unobserved. It must not be treated as a confidently measured unmethylated feature. The beta-binomial likelihood receives successes and coverage through `recon_targets_spec`; the required collator preserves both arrays. Fraction-only Gaussian shortcuts change the statistical model.

Follow the proof-of-concept settings explicitly: no extra joint QC or feature filtering beyond the parsed-bundle selection, random 85/5/10 split, seed 0. The actual loaders use batch size **24**; the notebook's `TrainingConfig(batch_size=16)` does not override a prebuilt loader.
**Notebook reference:** [UniVI_manuscript_GR-Supple_____scNMT-seq_mouse_gastrulation_data.ipynb](https://github.com/Ashford-A/UniVI/blob/8353ec8d422841e756b3abe4e9a5c4286c0c81dd/notebooks/GR_manuscript_reproducibility/UniVI_manuscript_GR-Supple_____scNMT-seq_mouse_gastrulation_data.ipynb), zero-based cells 4–19, 25–34, 38–56. Configuration choices are read from notebook code, not parameter JSON files.

```{figure} ../_static/figures/figure-s4.png
:alt: Published UniVI Figure s4, published figure reference
:class: published-figure

Published Figure s4, reproduced from the published article or Supplemental Material as a visual reference. The tutorial does not claim to regenerate these exact numbers or coordinates.
```

{download}`Download notebook <../examples/s04_scnmt.ipynb>` · {download}`Python script <../examples/s04_scnmt.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
from _common import *
from univi.datasets import load_scnmt_gastrulation_genebody_triplet,build_univi_inputs_from_scnmt_triplet
from univi.evaluation import encode_moe_gates_from_tensors
root,out=paths('scnmt')
raw=load_scnmt_gastrulation_genebody_triplet(root/'scnmt_gastrulation.tar.gz',
    require_qc=False,filter_features=False,min_cov_cpg=1,min_cov_gpc=1,verbose=False)
u=build_univi_inputs_from_scnmt_triplet(raw['rna'],raw['cpg'],raw['gpc'])
data=align_paired_obs_names(u['adata_dict']);spec=u['recon_targets_spec']
```

## Coverage-aware targets: an unobserved feature is not an observed zero
Encoder inputs are fractions. Decoder targets are the original successes/trials.

```python
for mod in ['cpg','gpc']:
    success=dense(data[mod].layers[spec[mod]['successes_layer']])
    total=dense(data[mod].layers[spec[mod]['total_count_layer']])
    assert np.isfinite(success).all() and np.isfinite(total).all()
    assert (success>=0).all() and (total>=success).all()
    assert np.allclose(success,np.round(success)) and np.allclose(total,np.round(total))
splits=split_map(data,root,lambda:random_split(data['rna'].n_obs,train=.85,val=.05,seed=0))
parts={key:{mod:a[idx].copy() for mod,a in data.items()} for key,idx in splits.items()}
model,loaders,meta=train_reference('scnmt',parts,out,recon_targets_spec=spec,seed=0)
```

## Evaluate held-out cells and all cells as separately labeled analyses
The same checkpoint is used for both. The all-cell result contains training cells.

```python
for scope,cohort in [('inductive-test',parts['test']),('all-cells-transductive',data)]:
    scope_out=out/scope;scope_out.mkdir(exist_ok=True)
    joint=draw_embedding(model,[(scope,m,a) for m,a in cohort.items()],scope_out,'stacked',
        ['modality','stage','embryo','lineage10x'])
    paired_report(model,cohort['rna'],cohort['cpg'],'rna','cpg',scope_out)
    weights=encode_moe_gates_from_tensors(model,{m:a.X for m,a in cohort.items()},
        device=DEVICE,modality_order=['rna','cpg','gpc'],kind='effective_precision',return_logits=False)
    table=pd.DataFrame(weights['weights'],index=cohort['rna'].obs_names,columns=weights['modality_order'])
    table.to_csv(scope_out/'precision-contributions-per-cell.csv')
    means=table.groupby(cohort['rna'].obs.lineage10x.astype(str)).mean()
    means.to_csv(scope_out/'precision-contributions-by-lineage.csv')
    fig,ax=plt.subplots(figsize=(7,max(3,len(means)*.25)))
    sns.heatmap(means,vmin=0,vmax=1,cmap='viridis',annot=True,fmt='.2f',ax=ax)
    ax.set_title('Normalized posterior-precision contributions')
    fig.savefig(scope_out/'s4-modality-contributions.png',dpi=150,bbox_inches='tight');plt.close(fig)
```

## Optional RNA -> CpG prediction, scored only where CpG was observed
Beta-binomial outputs are predicted fractions here, not methylated read counts.

```python
test=parts['test']
fractions=cross_modal_predict(model,test['rna'],'rna','cpg',device=DEVICE)
coverage=dense(test['cpg'].layers['meth_total_count'])
truth=dense(test['cpg'].X);observed=coverage>0
(out/'cpg-prediction.json').write_text(json.dumps({'n_observed_entries':int(observed.sum()),
    'observed_entry_mse':float(np.mean((fractions[observed]-truth[observed])**2))},indent=2))
np.save(out/'cpg-predicted-fractions.npy',fractions)
```

## Two evaluations with different evidential meaning

S4A–E use the held-out 114-cell test partition. S4F–J use all 1,140 paired cells with the same trained checkpoint. The latter includes training data and is labeled transductive/all-cell analysis; its stronger structure is not a separate out-of-sample validation. Published RNA–CpG FOSCTTM is 0.192 on the held-out subset and 0.029 on all cells.

Inspect modality, developmental stage, embryo of origin and atlas-transferred lineage labels together. An embedding ordered by stage but dominated by embryo requires a different interpretation from lineage structure replicated across embryos.

The modality-weight heatmap uses **normalized posterior-precision contributions**. In the visible source configuration, learned gating is disabled; the notebook's `router_x_precision` request can still return nonuniform precision-based contributions. These weights summarize encoder uncertainty across latent dimensions. They are not causal importance scores or proof that a learned router discovered biology. See [representations and weights](../guides/representations.md).

Optional RNA→CpG evaluation masks uncovered target entries and reports predicted fractions. Weighting errors by coverage would answer a different question and should be named explicitly.
