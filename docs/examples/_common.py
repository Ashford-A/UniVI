"""Shared plotting and run bookkeeping; all model operations use public UniVI APIs.

Run examples from the repository root. UNIVI_DATA defaults to data/tutorials;
UNIVI_OUTPUT defaults to results/tutorials. UNIVI_EPOCHS may shorten development
runs. UNIVI_SMOKE=1 uses tiny networks for software checks, never paper metrics.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from univi import ModalityConfig, UniVIConfig, TrainingConfig, UniVIMultiModalVAE
from univi.data import align_paired_obs_names
from univi.trainer import UniVITrainer
from univi.utils.seed import set_seed
from univi.workflows import make_loader, save_reference, stack_embeddings
from univi.evaluation import (encode_adata, cross_modal_predict, compute_foscttm,
    compute_match_recall_at_k, compute_modality_mixing, label_transfer_knn,
    reconstruction_metrics)
from sklearn.metrics import classification_report

DATA = Path(os.environ.get('UNIVI_DATA', 'data/tutorials'))
OUTPUT = Path(os.environ.get('UNIVI_OUTPUT', 'results/tutorials'))
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SMOKE = os.environ.get('UNIVI_SMOKE') == '1'
SEED = 42
set_seed(SEED)
sc.settings.verbosity = 0

# Read directly from configuration/training cells in the figure notebooks.
# Full cell provenance and differences from supplemental tables are documented.
RECIPES = {
 'cite': dict(beta=1.5, gamma=5.0, enc_drop=.1, dec_drop=0., schedules=(0,0,0,0), epochs=3000, batch=256, lr=1e-3, wd=1e-4, patience=100),
 'multiome': dict(beta=1.25, gamma=4.35, enc_drop=.1, dec_drop=.05, schedules=(50,85,75,110), epochs=5000, batch=128, lr=1e-3, wd=1e-4, patience=300),
 'bridge': dict(beta=1., gamma=5., enc_drop=.25, dec_drop=.05, schedules=(0,25,15,40), epochs=5000, batch=256, lr=1e-3, wd=1e-4, patience=150),
 'tea': dict(beta=1.15, gamma=1.45, enc_drop=.1, dec_drop=0., schedules=(0,50,25,75), epochs=3000, batch=256, lr=1e-4, wd=1e-4, patience=200),
 'aml': dict(beta=1.15, gamma=1.75, enc_drop=.1, dec_drop=.05, schedules=(0,0,0,0), epochs=3000, batch=256, lr=1e-4, wd=1e-5, patience=150),
 'share': dict(beta=1.25, gamma=6.35, enc_drop=.1, dec_drop=0., schedules=(50,85,75,110), epochs=5000, batch=128, lr=1e-3, wd=1e-4, patience=200),
 'scnmt': dict(beta=1.35, gamma=6.25, enc_drop=.01, dec_drop=.005, schedules=(30,60,40,70), epochs=5000, batch=24, lr=1e-3, wd=1e-4, patience=300),
 'benchmark': dict(beta=1.35, gamma=3.75, enc_drop=.1, dec_drop=.05, schedules=(25,75,45,95), epochs=200, batch=256, lr=1e-3, wd=1e-4, patience=50),
}


def paths(name):
    out = OUTPUT / name
    out.mkdir(parents=True, exist_ok=True)
    return DATA / name, out


def read_counts(path):
    a = sc.read_h5ad(path)
    if 'counts' not in a.layers:
        raise ValueError(f'{path}: add raw counts to layers["counts"] during data preparation.')
    return a


def read_paired(root, modalities):
    data = {mod: read_counts(root / f'{mod}.h5ad') for mod in modalities}
    before = {mod: a.n_obs for mod, a in data.items()}
    data = align_paired_obs_names(data)
    print('Paired cells:', before, '->', next(iter(data.values())).n_obs)
    return data


def split_map(data, root, fallback):
    """Prefer explicit barcode maps; never silently ignore mismatched barcodes."""
    first = next(iter(data.values()))
    file = root / 'splits.tsv'
    if not file.exists():
        return fallback()
    table = pd.read_csv(file, sep='\t', index_col=0)
    if not table.index.is_unique or set(table.index) != set(first.obs_names):
        raise ValueError('splits.tsv must contain each paired barcode exactly once.')
    split = table.reindex(first.obs_names)['split']
    if not split.isin(['train', 'val', 'test']).all():
        raise ValueError('Split values must be train, val, or test.')
    return {key: np.flatnonzero(split.to_numpy() == key) for key in ['train','val','test']}


def random_split(n, train=.8, val=.1, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(train*n), int((train+val)*n)
    return {'train':idx[:a], 'val':idx[a:b], 'test':idx[b:]}


def preprocess_parts(data, splits, preprocessors):
    parts = {key:{mod:a[idx].copy() for mod,a in data.items()} for key,idx in splits.items()}
    for mod, transform in preprocessors.items():
        transform.fit(parts['train'][mod])
        for key in parts:
            if parts[key][mod].n_obs:
                parts[key][mod] = transform.transform(parts[key][mod])
    return parts


def train_reference(recipe, parts, out, *, preprocessors=None, recon_targets_spec=None, seed=42):
    set_seed(seed)
    settings = RECIPES[recipe]
    widths = {'rna':[512,256,128], 'adt':[128,64], 'atac':[128,64]}
    if recipe == 'aml': widths.update(rna=[1024,512,256,128,64], adt=[128,64,32])
    if recipe == 'share': widths.update(rna=[1024,512,256,128], atac=[256,128,64])
    if recipe == 'scnmt': widths.update(rna=[512,256,128,64], cpg=[256,128,64], gpc=[256,128,64])
    if recipe == 'benchmark': widths['atac'] = [256,128,64]
    modalities = []
    for mod, a in parts['train'].items():
        hidden = [16,8] if SMOKE else widths[mod]
        weight = 2. if recipe == 'tea' and mod == 'adt' else (1.45 if recipe == 'tea' and mod == 'atac' else (3. if recipe == 'aml' and mod == 'adt' else 1.))
        modalities.append(ModalityConfig(mod, a.n_vars, hidden, hidden[::-1],
            likelihood='beta_binomial' if mod in ['cpg','gpc'] else 'gaussian', recon_weight=weight))
    ks,ke,als,ale = settings['schedules']
    cfg = UniVIConfig(latent_dim=4 if SMOKE else (20 if recipe=='scnmt' else 30),
        modalities=modalities, beta=settings['beta'], gamma=settings['gamma'],
        encoder_dropout=settings['enc_drop'], decoder_dropout=settings['dec_drop'],
        encoder_batchnorm=recipe!='aml', decoder_batchnorm=False,
        kl_anneal_start=ks, kl_anneal_end=ke, align_anneal_start=als, align_anneal_end=ale)
    batch = 16 if SMOKE else settings['batch']
    loaders = {key:make_loader(values, batch_size=batch, shuffle=key=='train',
        drop_last=(key=='train' and len(next(iter(values.values())))%batch==1),
        recon_targets_spec=recon_targets_spec, seed=seed) for key,values in parts.items()
        if len(next(iter(values.values()))) > 0}
    tcfg = TrainingConfig(n_epochs=int(os.environ.get('UNIVI_EPOCHS', settings['epochs'])),
        batch_size=batch, lr=settings['lr'], weight_decay=settings['wd'], device=DEVICE,
        early_stopping=True, patience=settings['patience'], grad_clip=5., log_every=50,
        seed=seed, best_epoch_warmup=0 if SMOKE else (30 if recipe=='scnmt' else 0))
    model = UniVIMultiModalVAE(cfg, loss_mode='v1', v1_recon='avg', normalize_v1_terms=True).to(DEVICE)
    trainer = UniVITrainer(model, loaders['train'], loaders['val'], tcfg, device=DEVICE)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    history=trainer.fit()
    if DEVICE == 'cuda': torch.cuda.synchronize()
    fit_seconds=time.perf_counter()-started
    metadata={'recipe':recipe, 'seed':seed, 'best_epoch':trainer.best_epoch,
              'fit_seconds':fit_seconds, 'smoke_test':SMOKE,
              'splits':{key:next(iter(values.values())).obs_names.tolist() for key,values in parts.items()},
              'features':{mod:a.var_names.tolist() for mod,a in parts['train'].items()}}
    save_reference(out/'reference', model, preprocessors=preprocessors, metadata=metadata)
    pd.DataFrame(history).to_csv(out/'training.csv',index=False)
    return model, loaders, metadata


def latent(model, a, mod):
    return encode_adata(model, a, mod, device=DEVICE, latent='modality_mean')


def draw_embedding(model, blocks, out, name, colors):
    joint=stack_embeddings(model, blocks, device=DEVICE)
    sc.pp.neighbors(joint, use_rep='X_univi', n_neighbors=min(30,joint.n_obs-1), random_state=42)
    sc.tl.umap(joint, random_state=42)
    sc.pl.umap(joint, color=[c for c in colors if c in joint.obs], show=False, frameon=False)
    plt.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight');plt.close('all')
    # Empty values are kept as missing; separate modality rows remain distinct.
    joint.write_h5ad(out/f'{name}.h5ad')
    return joint


def transfer_report(zs, ys, zt, yt, out, name, k=15):
    pred,acc,cm,order,f1 = label_transfer_knn(zs,np.asarray(ys),zt,np.asarray(yt),
        k=k,return_label_order=True,return_f1=True)
    pd.DataFrame(classification_report(yt,pred,output_dict=True,zero_division=0)).T.to_csv(out/f'{name}-classes.csv')
    normal=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(max(6,len(order)*.28),max(5,len(order)*.24)))
    sns.heatmap(normal,xticklabels=order,yticklabels=order,cmap='Blues',vmin=0,vmax=1,ax=ax)
    ax.set(xlabel='Predicted label',ylabel='Observed label',title=name)
    fig.savefig(out/f'{name}-confusion.png',dpi=150,bbox_inches='tight');plt.close(fig)
    report=pd.DataFrame(classification_report(yt,pred,output_dict=True,zero_division=0)).T
    perclass=report.loc[[str(label) for label in order],'f1-score']
    fig,ax=plt.subplots(figsize=(7,max(3,len(order)*.2)))
    perclass.plot.barh(ax=ax);ax.set(xlim=(0,1),xlabel='Per-class F1',title=name)
    fig.savefig(out/f'{name}-f1.png',dpi=150,bbox_inches='tight');plt.close(fig)
    return {'accuracy':acc,**f1}


def paired_report(model, a, b, mod_a, mod_b, out, *, label=None, k=15, limit=20000):
    if not a.obs_names.equals(b.obs_names):raise ValueError('Paired metrics require identical ordered barcodes.')
    za,zb=latent(model,a,mod_a),latent(model,b,mod_b)
    idx=np.random.default_rng(0).choice(a.n_obs,min(limit,a.n_obs),replace=False)
    za_s,zb_s=za[idx],zb[idx]
    report={}
    for source,target,x,y in [(mod_a,mod_b,za_s,zb_s),(mod_b,mod_a,zb_s,za_s)]:
        f,sem=compute_foscttm(x,y,return_sem=True)
        report[f'{source}_to_{target}']={'foscttm':f,'sem':sem,
            'recall10':compute_match_recall_at_k(x,y,k=min(10,len(x)))}
    if label:
        report['transfer_a_b']=transfer_report(za,a.obs[label].astype(str),zb,b.obs[label].astype(str),out,f'{mod_a}-to-{mod_b}',k)
        report['transfer_b_a']=transfer_report(zb,b.obs[label].astype(str),za,a.obs[label].astype(str),out,f'{mod_b}-to-{mod_a}',k)
    report['n_pairs']=int(len(idx))
    report['stacked_mixing']=compute_modality_mixing(np.vstack([za,zb]),np.array([mod_a]*len(za)+[mod_b]*len(zb)),k=min(30,2*len(za)-1))
    (out/f'{mod_a}-{mod_b}-metrics.json').write_text(json.dumps(report,indent=2))
    return report


def dense(x):return x.toarray() if sp.issparse(x) else np.asarray(x)


def cross_panels(model, source, target, src_mod, tgt_mod, coords, markers, group, out, name):
    if not source.obs_names.equals(target.obs_names):raise ValueError('Observed-vs-predicted comparisons require paired cells.')
    pred=cross_modal_predict(model,source,src_mod,tgt_mod,device=DEVICE)
    if pred.shape!=target.shape:raise ValueError('Target feature space differs from decoder output.')
    target.layers[f'predicted_from_{src_mod}']=pred
    truth=dense(target.X)
    stats=reconstruction_metrics(truth,pred)
    # Explicit per-feature summaries have stable table columns across library versions.
    from univi.evaluation import mse_per_feature,pearson_corr_per_feature
    pd.DataFrame({'feature':target.var_names,'mse':mse_per_feature(truth,pred),
        'pearson':pearson_corr_per_feature(truth,pred)}).to_csv(out/f'{name}-features.csv',index=False)
    valid=[m for m in markers if m in target.var_names]
    (out/f'{name}-missing-markers.json').write_text(json.dumps([m for m in markers if m not in target.var_names]))
    for marker in valid:
        j=target.var_names.get_loc(marker);both=np.concatenate([truth[:,j],pred[:,j]])
        vmin,vmax=np.quantile(both,[.01,.99])
        fig,axs=plt.subplots(1,2,figsize=(8,3.5),sharex=True,sharey=True)
        for ax,values,title in zip(axs,[truth[:,j],pred[:,j]],['Observed',f'Predicted from {src_mod}']):
            h=ax.scatter(coords[:,0],coords[:,1],c=values,s=3,cmap='viridis',vmin=vmin,vmax=vmax,rasterized=True)
            ax.set(title=f'{marker}: {title}',xlabel='UMAP 1',ylabel='UMAP 2');fig.colorbar(h,ax=ax)
        fig.savefig(out/f'{name}-{marker.replace("/","-")}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    if group and valid:
        ix=target.var_names.get_indexer(valid)
        obs=pd.DataFrame(truth[:,ix],columns=valid,index=target.obs_names).groupby(target.obs[group].astype(str)).mean()
        hat=pd.DataFrame(pred[:,ix],columns=valid,index=target.obs_names).groupby(target.obs[group].astype(str)).mean().reindex(obs.index)
        # Use the observed group means to define one scale for both heatmaps.
        mean,sd=obs.mean(),obs.std(ddof=0).replace(0,1)
        fig,axs=plt.subplots(1,2,figsize=(12,max(4,len(obs)*.25)))
        for ax,values,title in zip(axs,[obs,hat],['Observed','Predicted']):
            sns.heatmap((values-mean)/sd,cmap='vlag',center=0,vmin=-2,vmax=2,ax=ax);ax.set_title(title)
        fig.savefig(out/f'{name}-group-means.png',dpi=150,bbox_inches='tight');plt.close(fig)
        obs.to_csv(out/f'{name}-observed-means.csv');hat.to_csv(out/f'{name}-predicted-means.csv')
    target.write_h5ad(out/f'{name}-predictions.h5ad')
    return pred
