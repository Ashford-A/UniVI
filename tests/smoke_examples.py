"""Execute every tutorial with small synthetic inputs; not a biological validation.

Run manually: python tests/smoke_examples.py
The synthetic inputs and outputs are temporary and excluded from release artifacts.
"""
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
base=Path(tempfile.mkdtemp(prefix='univi-tutorial-smoke-'))
data_root=base/'data';output_root=base/'output'
rng=np.random.default_rng(4)
markers=['CD79A','MS4A1','LYZ','FCGR3A','NKG7','GNLY','TRAC','CD3D','GZMH','CCL5','BANK1','PAX5','CD74','HLA-DRA','IGKC','IGLC1','IGHM','IGHA1','IGHA2','LEF1','BCL11B','RUNX3','TNFAIP3','ITGAX','ITGAD','IL7R','LST1','CD8A','S100A8','CLEC10A','DNMT3B','ZBTB46','NYNRIN','ARHGAP22','LAPTM4B','MMRN1','DPYSL3','FAM30A','CDK6','CPXM1','SOCS2','SMIM24','EMP1','BEX3','CD34','AKR1C3','ADGRG1']
genes=markers+[f'GENE{i}' for i in range(300-len(markers))]
proteins=['CD19','CD20','CD14','CD16','CD56_1','CD3_1','CD4','CD8a','CD56','KLRG1','CD21','CD24','IgD','IgM','CD38','CD3','CD45RA','CD45RO','CD27','CD127','CD279','CD11b','CD11c','HLA-DR','CD141','CD172a','CD192','CD304']
peaks=[f'chr1:{i*1000}-{i*1000+500}' for i in range(512)]

def make(n=240,mod='rna',prefix='cell'):
    labels=np.array(['B','T','NK','Mono']*(n//4)+['B']*(n%4))
    obs=pd.DataFrame(index=[f'{prefix}{i}' for i in range(n)])
    for name in ['celltype.l1','celltype.l2','celltype.l3','cell_type','cell_state','celltype_harmonized']:obs[name]=labels
    obs['technology']='synthetic';obs['sample_id']=[f'patient{i//20}' for i in range(n)]
    obs['well']=[str([3,4,5,6][i%4]) for i in range(n)];obs['disease_status']='AML'
    for g in ['NPM1','DNMT3A','FLT3','TP53','NRAS','TET2','IDH2']:
        y=rng.integers(0,2,n).astype(float);y[rng.random(n)<.12]=np.nan;obs[f'mut_{g}']=y
    names=genes if mod=='rna' else proteins if mod=='adt' else peaks
    x=rng.poisson(3,(n,len(names))).astype('float32')
    if mod=='atac':x*=rng.random(x.shape)<.4
    v=pd.DataFrame(index=names)
    if mod=='atac':
        v['chrom']='chr1';v['chromStart']=np.arange(len(names))*1000;v['chromEnd']=v.chromStart+500
        obs['n_fragments']=2000
    a=ad.AnnData(sp.csr_matrix(x),obs=obs,var=v);a.layers['counts']=a.X.copy()
    if mod=='atac':a.uns['blacklist_removed']=True
    return a

for name,mods in [('cite',['rna','adt']),('multiome',['rna','atac']),('bridge',['rna','atac']),('tea',['rna','adt','atac']),('aml',['rna','adt']),('share',['rna','atac']),('benchmark',['rna','atac']),('raw',['rna','atac'])]:
    path=data_root/name;path.mkdir(parents=True)
    for mod in mods:make(mod=mod).write_h5ad(path/f'{mod}.h5ad')
    if name=='bridge':
        make(mod='rna',prefix='ding').write_h5ad(path/'ding_rna.h5ad')
        make(mod='atac',prefix='sat').write_h5ad(path/'satpathy_atac.h5ad')
    if name=='aml':
        make(mod='rna',prefix='vg').write_h5ad(path/'van_galen_rna.h5ad')
        make(mod='adt',prefix='dab').write_h5ad(path/'dab_adt.h5ad')
        pd.DataFrame({'cell_id':[f'cell{i}' for i in range(240)],'split':['train']*160+['val']*40+['test']*40}).set_index('cell_id').to_csv(path/'splits.tsv',sep='\t')
    if name=='raw':pd.DataFrame({'peak_name':peaks[:2],'gene_name':['GNLY']*2}).to_csv(path/'peak_gene_links.tsv',sep='\t',index=False)

path=data_root/'scnmt';path.mkdir()
n=120
samples=[f's{i}' for i in range(n)];rna_ids=[f'r{i}' for i in range(n)]
meta=pd.DataFrame({'sample':samples,'id_rna':rna_ids,'id_met':samples,'id_acc':samples,
    'stage':['E4.5','E5.5','E6.5','E7.5']*30,'embryo':[f'emb{i%6}' for i in range(n)],'lineage10x':['endo','meso','ecto']*40})
counts=pd.DataFrame(rng.poisson(4,(25,n)),columns=rna_ids);counts.insert(0,'ens_id',[f'g{i}' for i in range(25)])
lines=[]
for i,s in enumerate(samples):
    for j in range(16):
        cov=int(rng.integers(1,15));success=int(rng.integers(0,cov+1))
        lines.append(f'{s}\tg{j}\tgenebody\t{success}\t{cov}\t{100*success/cov}\n')
with tarfile.open(path/'scnmt_gastrulation.tar.gz','w:gz') as tar:
    for name,content in [('sample_metadata.txt',meta.to_csv(sep='\t',index=False).encode()),
        ('rna/counts.txt.gz',gzip.compress(counts.to_csv(sep='\t',index=False).encode())),
        ('met/feature_level/genebody.tsv.gz',gzip.compress(''.join(lines).encode())),
        ('acc/feature_level/genebody.tsv.gz',gzip.compress(''.join(lines[::-1]).encode()))]:
        info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))

env={**os.environ,'UNIVI_DATA':str(data_root),'UNIVI_OUTPUT':str(output_root),'UNIVI_SMOKE':'1','UNIVI_EPOCHS':'2',
     'MPLBACKEND':'Agg','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1',
     'PYTHONPATH':str(ROOT)}
examples=['00_quickstart.py','02_citeseq.py','03_cite_reconstruction.py','04_multiome.py','05_bridge.py','06_teaseq.py','07_aml.py','08_evaluation.py','s03_shareseq.py','s04_scnmt.py','s1_perturbation.py']
print('Synthetic test workdir:',base,flush=True)
for example in examples:
    log=base/(example+'.log')
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,str(ROOT/'docs/examples'/example)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print('FAIL',example,flush=True);print(log.read_text()[-6500:]);sys.exit(1)
    print('PASS',example,flush=True)
print('All tutorials executed on synthetic data.',flush=True)
