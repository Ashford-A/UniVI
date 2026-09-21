# %% [markdown]
# # S4: RNA, CpG methylation, and GpC accessibility in mouse gastrulation
# Use the parsed EBI bundle. Its assay-specific identifiers are mapped through
# sample_metadata.txt, rather than intersecting raw RNA IDs with methylation IDs.
# %%
from _common import *
from univi.datasets import load_scnmt_gastrulation_genebody_triplet,build_univi_inputs_from_scnmt_triplet
from univi.evaluation import encode_moe_gates_from_tensors
root,out=paths('scnmt')
raw=load_scnmt_gastrulation_genebody_triplet(root/'scnmt_gastrulation.tar.gz',
    require_qc=False,filter_features=False,min_cov_cpg=1,min_cov_gpc=1,verbose=False)
u=build_univi_inputs_from_scnmt_triplet(raw['rna'],raw['cpg'],raw['gpc'])
data=align_paired_obs_names(u['adata_dict']);spec=u['recon_targets_spec']
# %% [markdown]
# ## Coverage-aware targets: an unobserved feature is not an observed zero
# Encoder inputs are fractions. Decoder targets are the original successes/trials.
# %%
for mod in ['cpg','gpc']:
    success=dense(data[mod].layers[spec[mod]['successes_layer']])
    total=dense(data[mod].layers[spec[mod]['total_count_layer']])
    assert np.isfinite(success).all() and np.isfinite(total).all()
    assert (success>=0).all() and (total>=success).all()
    assert np.allclose(success,np.round(success)) and np.allclose(total,np.round(total))
splits=split_map(data,root,lambda:random_split(data['rna'].n_obs,train=.85,val=.05,seed=0))
parts={key:{mod:a[idx].copy() for mod,a in data.items()} for key,idx in splits.items()}
model,loaders,meta=train_reference('scnmt',parts,out,recon_targets_spec=spec,seed=0)
# %% [markdown]
# ## Evaluate held-out cells and all cells as separately labeled analyses
# The same checkpoint is used for both. The all-cell result contains training cells.
# %%
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
# %% [markdown]
# ## Optional RNA -> CpG prediction, scored only where CpG was observed
# Beta-binomial outputs are predicted fractions here, not methylated read counts.
# %%
test=parts['test']
fractions=cross_modal_predict(model,test['rna'],'rna','cpg',device=DEVICE)
coverage=dense(test['cpg'].layers['meth_total_count'])
truth=dense(test['cpg'].X);observed=coverage>0
(out/'cpg-prediction.json').write_text(json.dumps({'n_observed_entries':int(observed.sum()),
    'observed_entry_mse':float(np.mean((fractions[observed]-truth[observed])**2))},indent=2))
np.save(out/'cpg-predicted-fractions.npy',fractions)
