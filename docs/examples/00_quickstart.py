# %% [markdown]
# # A small, runnable UniVI quickstart
# Synthetic counts verify installation and the API; they do not establish biological performance.
# %%
import numpy as np
import pandas as pd
import anndata as ad
import torch
from univi import UniVIConfig,ModalityConfig,TrainingConfig,UniVIMultiModalVAE
from univi.preprocessing import RNAPreprocessor,ADTPreprocessor
from univi.trainer import UniVITrainer
from univi.workflows import make_loader
from univi.evaluation import encode_adata,cross_modal_predict
from univi.utils.seed import set_seed
set_seed(0)
rng=np.random.default_rng(0)
obs=pd.DataFrame(index=[f'cell-{i}' for i in range(80)])
rna=ad.AnnData(rng.poisson(3,(80,30)).astype('float32'),obs=obs.copy())
adt=ad.AnnData(rng.poisson(5,(80,6)).astype('float32'),obs=obs.copy())
for a in [rna,adt]:a.layers['counts']=a.X.copy()
# %% [markdown]
# ## Fit transforms on training cells; apply unchanged to held-out cells
# %%
rna_prep=RNAPreprocessor(n_hvg=None,scale=True).fit(rna[:60])
adt_prep=ADTPreprocessor(scale=True).fit(adt[:60])
train={'rna':rna_prep.transform(rna[:60]),'adt':adt_prep.transform(adt[:60])}
valid={'rna':rna_prep.transform(rna[60:70]),'adt':adt_prep.transform(adt[60:70])}
test={'rna':rna_prep.transform(rna[70:]),'adt':adt_prep.transform(adt[70:])}
# %% [markdown]
# ## Train with paired self/cross reconstruction and posterior alignment
# %%
cfg=UniVIConfig(latent_dim=4,encoder_batchnorm=False,modalities=[
    ModalityConfig('rna',30,[16],[16],likelihood='gaussian'),
    ModalityConfig('adt',6,[8],[8],likelihood='gaussian')])
model=UniVIMultiModalVAE(cfg,loss_mode='v1',v1_recon='avg',normalize_v1_terms=True)
trainer=UniVITrainer(model,make_loader(train,batch_size=20,shuffle=True),
    make_loader(valid,batch_size=10),TrainingConfig(n_epochs=3,device='cpu'))
trainer.fit()
# %% [markdown]
# ## Encode RNA alone and predict protein from the same cells
# %%
z=encode_adata(model,test['rna'],'rna',latent='modality_mean')
pred=cross_modal_predict(model,test['rna'],'rna','adt')
assert z.shape==(10,4) and pred.shape==(10,6)
print('Held-out RNA latent:',z.shape,'RNA-to-ADT prediction:',pred.shape)
