"""Prepare the existing conda-forge recipe from a verified UniVI source archive.

Default: fetch the *published* version from PyPI and verify its SHA-256.
--sdist FILE: prepare locally from an unpublished build, for review only.
No Git commit, push, PR, or package upload is performed by this script.
"""
from email.parser import BytesParser
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.error
import urllib.request
from packaging.version import Version


TEST_TEMPLATE = '''# BEGIN UNIVI RELEASE API CHECK
from copy import deepcopy
import tempfile
import torch
import univi
from univi import (ClassHeadConfig, ModalityConfig, UniVIConfig, UniVIMultiModalVAE,
                   UniVIRefiner, RefinementConfig, RNAPreprocessor, ADTPreprocessor,
                   ATACPreprocessor, save_reference, load_reference,
                   predict_feature_perturbation, make_loader, stack_embeddings,
                   load_scnmt_gastrulation_genebody_triplet)

assert univi.__version__ == "__RELEASE_VERSION__"
torch.set_num_threads(1)
model = UniVIMultiModalVAE(UniVIConfig(
    latent_dim=3, encoder_batchnorm=False, decoder_batchnorm=True,
    modalities=[ModalityConfig("rna", 4, [8], [8]), ModalityConfig("adt", 3, [8], [8])]))
model.freeze_decoders().train()
assert not model.decoders["rna"].training
before = deepcopy(model.decoders.state_dict())
model.add_classification_head(ClassHeadConfig(
    "celltype", 2, hidden_dims=[4], batchnorm=False, layernorm=True),
    label_names=["A", "B"])
assert all(torch.equal(value, model.decoders.state_dict()[name]) for name, value in before.items())
with tempfile.TemporaryDirectory() as directory:
    save_reference(directory, model, metadata={"check": "conda-forge"})
    restored, _, metadata = load_reference(directory)
    assert restored.head_label_names["celltype"] == ["A", "B"]
    assert metadata["check"] == "conda-forge"
print("UniVI public API and reference bundle checks passed.")
# END UNIVI RELEASE API CHECK
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedstock', required=True, type=Path)
    parser.add_argument('--version', required=True)
    parser.add_argument('--sdist', type=Path)
    args = parser.parse_args()
    version = str(Version(args.version))
    recipe = args.feedstock / 'recipe/meta.yaml'
    text = recipe.read_text(encoding='utf-8')
    old = re.search(r'{%\s*set version\s*=\s*[\"\']([^\"\']+)[\"\']\s*%}', text)
    if old is None or not re.search(r'{%\s*set name\s*=\s*[\"\']univi[\"\']\s*%}', text):
        raise SystemExit('Expected the existing Jinja-format UniVI feedstock recipe.')
    if Version(old.group(1)) > Version(version):
        raise SystemExit('Refusing to downgrade a newer feedstock version.')
    if args.sdist:
        payload = args.sdist.read_bytes()
        origin = str(args.sdist.resolve())
        print('Preparing from a LOCAL archive; publish this exact sdist or refresh from PyPI before submitting.')
    else:
        try:
            with urllib.request.urlopen(f'https://pypi.org/pypi/univi/{version}/json', timeout=30) as response:
                info = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise SystemExit(f'UniVI {version} is not on PyPI yet. Upload the release first.') from error
            raise
        if info['info']['name'].lower() != 'univi' or info['info']['version'] != version:
            raise SystemExit('Unexpected PyPI project/version metadata.')
        sources = [item for item in info['urls'] if item['packagetype'] == 'sdist' and not item.get('yanked')]
        if len(sources) != 1:
            raise SystemExit('Expected one non-yanked source distribution on PyPI.')
        source = sources[0]
        if not source['url'].startswith('https://files.pythonhosted.org/'):
            raise SystemExit('Unexpected PyPI source host.')
        with urllib.request.urlopen(source['url'], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != source['digests']['sha256']:
            raise SystemExit('Downloaded sdist SHA-256 does not match PyPI metadata.')
        origin = source['url']
    digest = hashlib.sha256(payload).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        metadata = BytesParser().parsebytes(archive.extractfile(f'univi-{version}/PKG-INFO').read())
    if metadata['Name'] != 'univi' or metadata['Version'] != version:
        raise SystemExit('The source archive is not the requested UniVI version.')
    text = text[:old.start()] + '{% set version = "' + version + '" %}' + text[old.end():]
    text, count = re.subn(r'(?m)^  sha256:.*$', '  sha256: ' + digest, text)
    if count != 1:
        raise SystemExit('Expected exactly one source SHA-256 field; inspect the recipe manually.')
    text, count = re.subn(r'(?m)^(build:\s*\n)  number:.*$', r'\g<1>  number: 0', text)
    if count != 1:
        raise SystemExit('Expected one build number directly under build:.')
    if not re.search(r'(?m)^    - joblib(?:\s|$)', text):
        text, count = re.subn(r'(?m)^(    - tqdm[^\n]*\n)', r'\g<1>    - joblib >=1.3\n', text)
        if count != 1:
            raise SystemExit('Cannot locate the runtime dependency block.')
    else:
        text = re.sub(r'(?m)^    - joblib[^\n]*$', '    - joblib >=1.3', text)
    imports = ['univi.refinement', 'univi.preprocessing', 'univi.workflows', 'univi.datasets', 'univi.perturbation']
    additions = ''.join('    - ' + module + '\n' for module in imports if '    - ' + module + '\n' not in text)
    text = text.replace('    - univi\n', '    - univi\n' + additions, 1)
    test_file = recipe.parent / 'run_test.py'
    test_text = test_file.read_text(encoding='utf-8') if test_file.exists() else ''
    block = TEST_TEMPLATE.replace('__RELEASE_VERSION__', version)
    if '# BEGIN UNIVI RELEASE API CHECK' in test_text:
        test_text = re.sub(r'# BEGIN UNIVI RELEASE API CHECK.*?# END UNIVI RELEASE API CHECK\n?',
                           lambda _: block, test_text, flags=re.DOTALL)
    else:
        test_text = test_text.rstrip() + ('\n\n' if test_text.strip() else '') + block
    recipe.write_text(text, encoding='utf-8')
    test_file.write_text(test_text, encoding='utf-8')
    print(json.dumps({'version': version, 'source': origin, 'sha256': digest,
                      'recipe': str(recipe.resolve()), 'test': str(test_file.resolve())}, indent=2))


if __name__ == '__main__':
    main()
