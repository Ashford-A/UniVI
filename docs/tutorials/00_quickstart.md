# Quickstart: train, encode, and predict

Run a small paired RNA–protein analysis on a CPU. This example creates synthetic counts, holds out cells, fits normalization on the training partition, trains UniVI, and predicts protein from RNA alone. It is a complete installation check without a download.

**What you learn:** the public configuration classes, the required collator through `make_loader`, modality-specific encoding, and cross-modal prediction. The shapes at the end should be `(10, 4)` for RNA latent coordinates and `(10, 6)` for predicted proteins. Random synthetic counts have no intended biological signal.

Install the [tutorial source version](../installation.md) before running this example. Subsequent tutorials expect real, prepared data with the [documented input contracts](../guides/data.md).

{download}`Download notebook <../examples/00_quickstart.ipynb>` · {download}`Python script <../examples/00_quickstart.py>` · {download}`Shared helper <../examples/_common.py>` · {download}`All tutorials <../_downloads/tutorials.zip>`

```python
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
```

## Fit transforms on training cells; apply unchanged to held-out cells

```python
rna_prep=RNAPreprocessor(n_hvg=None,scale=True).fit(rna[:60])
adt_prep=ADTPreprocessor(scale=True).fit(adt[:60])
train={'rna':rna_prep.transform(rna[:60]),'adt':adt_prep.transform(adt[:60])}
valid={'rna':rna_prep.transform(rna[60:70]),'adt':adt_prep.transform(adt[60:70])}
test={'rna':rna_prep.transform(rna[70:]),'adt':adt_prep.transform(adt[70:])}
```

## Train with paired self/cross reconstruction and posterior alignment

```python
cfg=UniVIConfig(latent_dim=4,encoder_batchnorm=False,modalities=[
    ModalityConfig('rna',30,[16],[16],likelihood='gaussian'),
    ModalityConfig('adt',6,[8],[8],likelihood='gaussian')])
model=UniVIMultiModalVAE(cfg,loss_mode='v1',v1_recon='avg',normalize_v1_terms=True)
trainer=UniVITrainer(model,make_loader(train,batch_size=20,shuffle=True),
    make_loader(valid,batch_size=10),TrainingConfig(n_epochs=3,device='cpu'))
trainer.fit()
```

## Encode RNA alone and predict protein from the same cells

```python
z=encode_adata(model,test['rna'],'rna',latent='modality_mean')
pred=cross_modal_predict(model,test['rna'],'rna','adt')
assert z.shape==(10,4) and pred.shape==(10,6)
print('Held-out RNA latent:',z.shape,'RNA-to-ADT prediction:',pred.shape)
```

## Continue with real data

Start with [paired CITE-seq](02_citeseq.md) for a fully paired assay, [reference bridging](05_bridge.md) for independent unimodal cohorts, or [scNMT-seq](s04_scnmt.md) for coverage-aware methylation data. Use the same sequence: split, fit preprocessing on training cells, train, infer on held-out data, then evaluate.
