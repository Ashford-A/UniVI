"""Generate readable tutorial pages/notebooks and AST-based API reference.

No UniVI import, model training, or dataset download occurs in a docs build.
Run after edits: python scripts/build_tutorial_docs.py
CI: python scripts/build_tutorial_docs.py --check
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
EXAMPLES=DOCS/'examples'
TUTORIALS=['00_quickstart','02_citeseq','03_cite_reconstruction','04_multiome','05_bridge',
           '06_teaseq','07_aml','08_evaluation','s03_shareseq','s04_scnmt','s1_perturbation']
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
changed=[]

def write(path,text):
    old=path.read_text(encoding="utf-8") if path.exists() else ''
    if old!=text:
        changed.append(str(path.relative_to(ROOT)))
        if not args.check:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text, encoding="utf-8")

for name in TUTORIALS:
    source=(EXAMPLES/f'{name}.py').read_text(encoding="utf-8")
    segments=re.split(r'^# %%(?: \[(markdown)\])?\s*\n',source,flags=re.M)
    cells=[];rendered=[]
    for i in range(1,len(segments),2):
        mode=segments[i];content=segments[i+1].strip()
        if not content:continue
        if mode=='markdown':
            content='\n'.join(re.sub(r'^# ?', '',line) for line in content.splitlines())
            cell=nbformat.v4.new_markdown_cell(content)
            # First markdown cell repeats the page title; keep it only in the notebook.
            if cells:rendered.append(content)
        else:
            cell=nbformat.v4.new_code_cell(content)
            rendered.append('```python\n'+content+'\n```')
        cell['id']=hashlib.sha256((name+str(i)+content).encode()).hexdigest()[:12]
        cells.append(cell)
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
        'univi_tutorial':{'execution_status':'Unexecuted portable tutorial; synthetic script checks reported separately.'}})
    write(EXAMPLES/f'{name}.ipynb',nbformat.writes(nb)+'\n')
    fragments=DOCS/'_fragments'
    intro=(fragments/f'{name}.intro.md').read_text(encoding="utf-8")
    after=(fragments/f'{name}.after.md').read_text(encoding="utf-8")
    downloads=f'\n{{download}}`Download notebook <../examples/{name}.ipynb>` · {{download}}`Python script <../examples/{name}.py>` · {{download}}`Shared helper <../examples/_common.py>` · {{download}}`All tutorials <../_downloads/tutorials.zip>`\n'
    write(DOCS/'tutorials'/f'{name}.md',intro+downloads+'\n'+ '\n\n'.join(rendered)+'\n\n'+after)

# Lightweight, source-derived API documentation; deliberately avoids heavyweight
# autodoc imports of Torch/SCANPY on Read the Docs.
selected={
 'config.py':['ModalityConfig','UniVIConfig','TrainingConfig','ClassHeadConfig'],
 'models/univi.py':['UniVIMultiModalVAE'],
 'preprocessing.py':['RNAPreprocessor','ADTPreprocessor','ATACPreprocessor','split_by_label'],
 'refinement.py':['RefinementConfig','UniVIRefiner','predict_heads_adata'],
 'workflows.py':['make_loader','stack_embeddings','save_reference','load_reference'],
 'datasets.py':['load_scnmt_gastrulation_genebody_triplet','build_univi_inputs_from_scnmt_triplet'],
 'perturbation.py':['predict_feature_perturbation'],
 'evaluation.py':['encode_adata','encode_fused_adata_pair','cross_modal_predict','denoise_adata',
   'denoise_from_multimodal','evaluate_alignment','compute_foscttm','compute_match_recall_at_k',
   'label_transfer_knn','reconstruction_metrics','encode_moe_gates_from_tensors','add_lsc17_scores',
   'fit_label_latent_gaussians','sample_latent_by_label','generate_from_latent'],
}
methods={'__init__','fit','transform','fit_transform','evaluate','add_classification_head','freeze_decoders',
 'unfreeze_decoders','freeze_encoders','unfreeze_encoders','predict_heads','encode_fused','train'}
api=['# API reference\n','Generated from the package source using Python AST, without importing the model during documentation builds. The new preprocessing, refinement, workflow, and perturbation modules and component-freezing methods require UniVI 0.5.0 or later. Public helpers are available from their modules and through lazy top-level imports.\n']
for file,names in selected.items():
    tree=ast.parse((ROOT/'univi'/file).read_text(encoding="utf-8"))
    api.append(f'## `univi.{file[:-3].replace("/", ".")}`\n')
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.ClassDef)) or node.name not in names:continue
        api.append(f'### `{node.name}`\n')
        if isinstance(node,ast.FunctionDef):api.append('```python\n'+node.name+'('+ast.unparse(node.args)+')\n```\n')
        else:
            fields=[ast.unparse(n) for n in node.body if isinstance(n,ast.AnnAssign)]
            if fields:api.append('```python\n'+'\n'.join(fields)+'\n```\n')
        doc=ast.get_docstring(node)
        if doc:api.append('```text\n'+doc+'\n```\n')
        if isinstance(node,ast.ClassDef):
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name in methods:
                    api.append('#### `'+node.name+'.'+method.name+'`\n\n```python\n'+method.name+'('+ast.unparse(method.args)+')\n```\n')
                    text=ast.get_docstring(method)
                    if text:api.append('```text\n'+text+'\n```\n')
write(DOCS/'reference/api.md','\n'.join(api))
if args.check:
    if changed:raise SystemExit('Generated docs are stale:\n'+'\n'.join(changed))
else:
    # The ZIP preserves the helper alongside notebooks for immediate local use.
    with zipfile.ZipFile(DOCS/'_downloads/tutorials.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(EXAMPLES.iterdir()):
            if file.suffix in {'.py','.ipynb','.md'}:z.write(file,arcname=file.name)
print('Generated files are current.' if args.check else f'Updated {len(changed)} generated files.')
