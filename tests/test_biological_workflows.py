from copy import deepcopy
import importlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData

from univi import ClassHeadConfig, ModalityConfig, UniVIConfig, UniVIMultiModalVAE
from univi.preprocessing import RNAPreprocessor, ADTPreprocessor, ATACPreprocessor, split_by_label
from univi.refinement import RefinementConfig, UniVIRefiner, _supervised_sums, predict_heads_adata
from univi.workflows import make_loader, save_reference, load_reference, stack_embeddings

torch.set_num_threads(1)


def model():
    return UniVIMultiModalVAE(UniVIConfig(
        latent_dim=3, encoder_batchnorm=True, decoder_batchnorm=True,
        encoder_dropout=0.2, decoder_dropout=0.4,
        modalities=[ModalityConfig('rna', 6, [8], [8]), ModalityConfig('adt', 4, [8], [8])]))


def data(n=30, d=6, seed=0):
    a = AnnData(sp.csr_matrix(np.random.default_rng(seed).poisson(3, (n, d)).astype(np.float32)),
                obs=pd.DataFrame(index=[f'cell{i}' for i in range(n)]),
                var=pd.DataFrame(index=[f'f{i}' for i in range(d)]))
    a.layers['counts'] = a.X.copy()
    return a


def test_freeze_preserves_weights_buffers_and_dropout_across_train():
    m = model().freeze_decoders()
    before = deepcopy(m.decoders.state_dict())
    x = {'rna': torch.randn(10, 6), 'adt': torch.randn(10, 4)}
    m.train()
    assert not m.decoders['rna'].training
    optim = torch.optim.AdamW(m.parameters(), lr=0.01)
    loss = m(x, epoch=1)['loss'].mean()
    loss.backward()
    assert any(p.grad is not None for p in m.encoders.parameters())
    assert all(p.grad is None for p in m.decoders.parameters())
    optim.step()
    assert all(torch.equal(before[k], v) for k, v in m.decoders.state_dict().items())
    m.unfreeze_decoders(['rna']).train()
    assert m.decoders['rna'].training and not m.decoders['adt'].training


def test_head_attachment_defaults_and_existing_reference_unchanged():
    m = model()
    before = deepcopy(m.state_dict())
    m.add_classification_head(ClassHeadConfig('celltype', 3))
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError):
        m.add_classification_head(ClassHeadConfig('celltype', 3))
    m.add_classification_head(ClassHeadConfig('NPM1', 2, head_type='binary', layernorm=True, batchnorm=False))
    m.eval()
    probs = m.predict_heads({'rna': torch.ones(4, 6)})
    assert probs['NPM1'].shape == (4, 1)
    assert torch.allclose(probs['celltype'].sum(1), torch.ones(4))


def test_missing_targets_do_not_affect_loss_and_invalid_targets_fail():
    m = model()
    m.add_classification_head(ClassHeadConfig('mutation', 2, head_type='binary', batchnorm=False, dropout=0))
    z = torch.randn(4, 3, requires_grad=True)
    y = torch.tensor([0., 1., -1., float('nan')])
    loss, count = _supervised_sums(m, z, {'mutation': y})
    expected, _ = _supervised_sums(m, z[:2], {'mutation': y[:2]})
    assert count == 2 and torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(z.grad[2:], torch.zeros(2, 3))
    with pytest.raises(ValueError):
        _supervised_sums(m, z, {'mutation': torch.tensor([0, 1, 2, -1])})


@pytest.mark.parametrize('kind', ['categorical', 'binary'])
def test_staged_refinement_and_checkpoint_roundtrip(tmp_path, kind):
    torch.manual_seed(3)
    m = model()
    m.add_classification_head(ClassHeadConfig('target', 2, head_type=kind,
                             hidden_dims=[6], layernorm=True, batchnorm=False, dropout=0),
                             label_names=['WT', 'MUT'])
    a = data()
    labels = {'target': np.arange(a.n_obs) % 2}
    train = make_loader({'rna': a[:20]}, labels={'target': labels['target'][:20]}, batch_size=10)
    val = make_loader({'rna': a[20:]}, labels={'target': labels['target'][20:]}, batch_size=10)
    before = deepcopy(m.decoders.state_dict())
    enc_before = deepcopy(m.encoders['rna'].state_dict())
    refiner = UniVIRefiner(m, [train], [val], encoder_modalities=['rna'],
        config=RefinementConfig(max_epochs=3, warmup_epochs=1, latent_weight=1, log_every=0))
    report = refiner.fit()
    assert len(report['history']) == 3
    assert all(torch.equal(before[k], v) for k, v in m.decoders.state_dict().items())
    # The best checkpoint can legitimately be the warmup checkpoint.
    assert report['best_epoch'] in [0, 1, 2]
    prep = RNAPreprocessor(n_hvg=None).fit(a[:20])
    save_reference(tmp_path, m, preprocessors={'rna': prep}, metadata={'split': 'cell-level'})
    loaded, transforms, metadata = load_reference(tmp_path)
    for k, v in m.state_dict().items():
        assert torch.equal(v.cpu(), loaded.state_dict()[k])
    assert loaded.head_label_names['target'] == ['WT', 'MUT']
    assert metadata['split'] == 'cell-level'
    assert transforms['rna'].features_ == prep.features_
    assert np.allclose(predict_heads_adata(m, a, 'rna')['target'],
                       predict_heads_adata(loaded, a, 'rna')['target'])


def test_refinement_replay_and_all_missing_validation():
    m = model().add_classification_head(ClassHeadConfig('y', 2, dropout=0, batchnorm=False))
    a, b = data(20), data(20, 4)
    labels = {'y': np.arange(20) % 2}
    loader = make_loader({'rna': a}, labels=labels, batch_size=10)
    paired = make_loader({'rna': a, 'adt': b}, batch_size=10)
    r = UniVIRefiner(m, [loader], [loader], replay_loader=paired,
        config=RefinementConfig(max_epochs=1, warmup_epochs=0, replay_weight=0.1, log_every=0))
    r.fit()
    missing = make_loader({'rna': a}, labels={'y': np.full(20, -1)}, batch_size=10)
    with pytest.raises(ValueError, match='no observed labels'):
        UniVIRefiner(m, [loader], [missing]).fit()


@pytest.mark.parametrize('factory', [
    lambda: RNAPreprocessor(n_hvg=None, scale=True),
    lambda: ADTPreprocessor(scale=True),
    lambda: ATACPreprocessor(3, drop_first=True),
    lambda: ATACPreprocessor(3, method='signac'),
    lambda: ATACPreprocessor(3, method='tea'),
])
def test_preprocessing_uses_fixed_training_transform_and_feature_order(factory):
    a = data()
    transform = factory().fit(a[:20])
    normal = transform.transform(a[20:]).X
    shuffled = transform.transform(a[20:, ::-1]).X
    assert np.allclose(normal, shuffled)
    # Query rows are transformed independently; adding another query cannot refit a scaler.
    one = transform.transform(a[20:21]).X
    assert np.allclose(np.asarray(normal)[0], np.asarray(one)[0])
    assert np.array_equal(a.layers['counts'].toarray(), data().X.toarray())
    with pytest.raises((ValueError, KeyError)):
        transform.transform(a[:, 1:])


def test_pairing_splits_and_stacked_rows():
    a, b = data(), data(d=4)
    splits = split_by_label(np.arange(30) % 3, train_cap=4, val_cap=1)
    ids = np.concatenate(list(splits.values()))
    assert sorted(ids) == list(range(30)) and len(np.unique(ids)) == 30
    with pytest.raises(ValueError):
        make_loader({'rna': a, 'adt': b[::-1]})
    stacked = stack_embeddings(model(), [('bridge', 'rna', a), ('bridge', 'adt', b)])
    assert stacked.n_obs == 60 and stacked.obs_names.is_unique
    assert stacked.obs.cell_id.tolist()[:30] == a.obs_names.tolist()


def test_lazy_module_import_is_not_recursive():
    import univi
    assert univi.evaluation is importlib.import_module('univi.evaluation')


def test_new_public_top_level_exports_resolve_to_their_documented_modules():
    import univi
    for name, module in univi._WORKFLOW_EXPORTS.items():
        assert name in univi.__all__
        assert getattr(univi, name) is getattr(importlib.import_module(f'univi.{module}'), name)


def test_fractional_feature_edit_preserves_integer_input_and_matches_explicit_edit():
    from univi.perturbation import predict_feature_perturbation
    from univi.evaluation import cross_modal_predict
    m = model().eval()
    a = data(10)
    a.X = a.X.astype(np.int32)
    original = a.X.copy()
    result = predict_feature_perturbation(m, a, source_modality='rna',
        target_modality='adt', features='f0', mode='set', value=0.5)
    manual = a.copy()
    x = manual.X.astype(np.float32).tolil()
    x[:, 0] = 0.5
    manual.X = x.tocsr()
    expected = cross_modal_predict(m, manual, 'rna', 'adt')
    assert np.allclose(result['perturbed'], expected)
    assert np.allclose(result['delta'], expected - result['baseline'])
    assert (a.X != original).nnz == 0 and a.X.dtype == np.int32
    with pytest.raises(KeyError):
        predict_feature_perturbation(m, a, source_modality='rna',
            target_modality='adt', features=['nonexistent'])


def test_failed_reference_overwrite_leaves_existing_model_unchanged(tmp_path):
    m = model()
    save_reference(tmp_path, m, preprocessors={'rna': RNAPreprocessor(None).fit(data())})
    before = (tmp_path / 'model.pt').read_bytes()
    with pytest.raises(ValueError, match='stale'):
        save_reference(tmp_path, model())
    assert (tmp_path / 'model.pt').read_bytes() == before


def test_pool_cap_is_applied_before_splitting_and_freeze_accepts_one_name():
    split = split_by_label(np.repeat(['a', 'b'], 100), max_per_label=20, seed=42)
    selected = np.concatenate(list(split.values()))
    # With a pool cap (unused_to_test=True convention), overflow stays in test.
    assert len(selected) == len(np.unique(selected)) == 200
    assert len(split['train']) == 32 and len(split['val']) == 4 and len(split['test']) == 164
    m = model().freeze_decoders('rna').train()
    assert not m.decoders['rna'].training and m.decoders['adt'].training
    before = set(m.class_heads)
    with pytest.raises(ValueError, match='LayerNorm'):
        m.add_classification_head(ClassHeadConfig('invalid', 2, layernorm=True))
    assert set(m.class_heads) == before
