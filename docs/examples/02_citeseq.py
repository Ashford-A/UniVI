# %% [markdown]
# # Figures 2–3 foundation: paired CITE-seq
# Use the Figure 3 notebook as the shared CITE reference (its label vocabulary,
# split helper and patience differ from Figure 2's standalone notebook).
# %%
from _common import *
from univi.preprocessing import RNAPreprocessor, ADTPreprocessor, split_by_label
root,out=paths('cite')
paired=read_paired(root,['rna','adt'])
rna=paired['rna']
# %% [markdown]
# ## Split before fitting any learned transform
# A provided splits.tsv takes precedence. max_per_label caps the pool BEFORE
# its 80/10/10 division, just as Figure 3 cell 15 does. Overflow is added to test.
# %%
splits=split_map(paired,root,lambda:split_by_label(rna.obs['celltype.l3'],
    max_per_label=1200,seed=42))
preprocessors={'rna':RNAPreprocessor(n_hvg=2000,scale=False,normalize_on_selected=True),
               'adt':ADTPreprocessor(scale=False)}
parts=preprocess_parts(paired,splits,preprocessors)
# %% [markdown]
# ## Train the paired generative reference
# No cell-type labels enter the VAE loss. They are used for split balancing and evaluation.
# %%
model,loaders,metadata=train_reference('cite',parts,out,preprocessors=preprocessors)
test=parts['test']
for mod,a in test.items():a.write_h5ad(out/f'{mod}-test.h5ad')
# %% [markdown]
# ## Make the Figure 2 views and bidirectional label-transfer reports
# FOSCTTM uses at most 20,000 paired test cells; label transfer uses the full test set.
# %%
joint=draw_embedding(model,[('Hao','rna',test['rna']),('Hao','adt',test['adt'])],
    out,'test-stacked',['modality','celltype.l2'])
report=paired_report(model,test['rna'],test['adt'],'rna','adt',out,label='celltype.l2',k=15)
print(report)
