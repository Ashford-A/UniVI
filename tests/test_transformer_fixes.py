"""Regression tests for the tokenizer / transformer fixes released in 1.2.0."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from univi import ModalityConfig, UniVIConfig, UniVIMultiModalVAE
from univi.config import TokenizerConfig, TransformerConfig
from univi.models.encoders import _VectorToTokens

T = TransformerConfig(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0.0, attn_dropout=0.0)


def _tok(n, **kw):
    return TokenizerConfig(mode="topk_channels", n_tokens=6, channels=("value", "rank"), n_features=n,
                           use_feature_embedding=True, feature_emb_dim=8, **kw)


def _fused_model(**cfg_kw):
    mods = [ModalityConfig("rna", 20, [16], [16], tokenizer=_tok(20)),
            ModalityConfig("atac", 30, [16], [16], tokenizer=_tok(30))]
    cfg = UniVIConfig(latent_dim=4, modalities=mods, fused_encoder_type="multimodal_transformer",
                      fused_transformer=T, **cfg_kw)
    return UniVIMultiModalVAE(cfg, loss_mode="v2", recon_normalize_by_dim=False)


def test_feature_embedding_is_a_field_and_reaches_the_fused_encoder():
    model = _fused_model()
    for name in ("rna", "atac"):
        assert model.fused_encoder.vec2tok[name].id_emb is not None


def test_feature_embedding_reaches_a_per_modality_transformer():
    mods = [ModalityConfig("rna", 20, [16], [16], encoder_type="transformer", transformer=T, tokenizer=_tok(20)),
            ModalityConfig("atac", 30, [16], [16])]
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=mods))
    assert model.encoders["rna"].vec2tok.id_emb is not None


def test_save_and_load_reference_round_trip_with_feature_embeddings(tmp_path):
    from univi.workflows import load_reference, save_reference
    torch.manual_seed(0)
    model = _fused_model().eval()
    x = {"rna": torch.randn(5, 20), "atac": torch.randn(5, 30)}
    save_reference(tmp_path / "ref", model)
    loaded = load_reference(tmp_path / "ref")
    restored = loaded[0] if isinstance(loaded, tuple) else getattr(loaded, "model", loaded)
    restored.eval()
    with torch.no_grad():
        a = model.encode_fused(x, use_mean=True)[0]
        b = restored.encode_fused(x, use_mean=True)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_coordinate_embedding_depends_on_position():
    tok = TokenizerConfig(mode="topk_channels", n_tokens=4, channels=("value",), n_features=4,
                          use_coord_embedding=True, n_chroms=1)
    v = _VectorToTokens(input_dim=4, tok=tok)
    v.set_feature_coords(torch.zeros(4, dtype=torch.long), torch.tensor([0., 5e4, 5e6, 5e7]),
                         torch.tensor([500., 5e4 + 500, 5e6 + 500, 5e7 + 500]))
    idx = torch.arange(4).view(1, 4)
    out = v._apply_coord_emb(torch.zeros(1, 4, v.d_in), idx)[0]
    # four different positions on the same chromosome must give four different embeddings
    dists = torch.cdist(out, out)
    assert dists[~torch.eye(4, dtype=torch.bool)].min() > 1e-4


def test_topk_embed_mode_builds_and_runs_with_feature_info():
    info = {"chrom": ["chr1", "chr1", "chr2", "chr2", "chr3"], "start": [0, 100, 0, 5000, 10],
            "end": [50, 150, 50, 5050, 60]}
    tok = TokenizerConfig(mode="topk_embed", n_tokens=3, channels=("value",), n_features=5, d_model=8,
                          use_coords=True, chrom_vocab_size=3, feature_info=info)
    v = _VectorToTokens(input_dim=5, tok=tok)
    assert v.id_emb is not None and v._has_coords
    tokens, _ = v(torch.randn(2, 5), return_indices=False)
    assert tokens.shape[:2] == (2, 3)


def test_relative_position_bias_is_passed_through_and_trained():
    tcfg = TransformerConfig(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0.0,
                             attn_dropout=0.0, use_relpos_bias=True, relpos_num_bins=8, relpos_max_dist=1e5)
    tok = TokenizerConfig(mode="topk_channels", n_tokens=5, channels=("value",), n_features=10,
                          use_coord_embedding=True, n_chroms=2)
    mods = [ModalityConfig("atac", 10, [16], [16], encoder_type="transformer", transformer=tcfg, tokenizer=tok),
            ModalityConfig("rna", 6, [16], [16])]
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=mods))
    enc = model.encoders["atac"]
    relpos = [m.relpos for m in enc.encoder.modules() if getattr(m, "relpos", None) is not None]
    assert relpos, "use_relpos_bias was not passed to the transformer"
    enc.vec2tok.set_feature_coords(torch.tensor([0] * 5 + [1] * 5), torch.arange(10.) * 1e4,
                                   torch.arange(10.) * 1e4 + 500)
    mu, logvar = enc(torch.randn(3, 10))
    (mu.sum() + logvar.sum()).backward()
    assert all(r.bias.grad is not None and r.bias.grad.abs().sum() > 0 for r in relpos)


def test_fused_v2_per_modality_encoders_are_not_decayed_before_alignment_starts():
    """With alignment off, weight decay must not shrink the per-modality encoders (fixed in 1.2.0)."""
    from univi import TrainingConfig, UniVITrainer
    from univi.workflows import make_loader
    import anndata as ad

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs = {"rna": ad.AnnData(rng.normal(size=(64, 20)).astype(np.float32)),
           "atac": ad.AnnData(rng.normal(size=(64, 30)).astype(np.float32))}
    for a in obs.values():
        a.obs_names = [f"c{i}" for i in range(64)]
    model = _fused_model(align_anneal_start=1000, align_anneal_end=1001, kl_anneal_start=0, kl_anneal_end=1)
    before = {k: v.detach().clone() for k, v in model.encoders.named_parameters()}   # parameters only:
    # BatchNorm running statistics are buffers and change in every forward pass by design
    UniVITrainer(model, make_loader(obs, batch_size=8, shuffle=True, drop_last=True), None,
                 TrainingConfig(n_epochs=3, batch_size=8, lr=1e-2, weight_decay=1e-2, device="cpu")).fit()
    after = dict(model.encoders.named_parameters())
    assert all(torch.equal(before[k], after[k].detach().cpu()) for k in before), "per-modality encoders changed"
    fused_moved = any(p.grad is not None for p in model.fused_encoder.parameters())
    assert fused_moved, "the fused encoder should still be trained"


def test_relpos_bias_is_chromosome_aware_and_matches_float64_reference():
    """1.2.1: float32 (chromosome, position) input gives the same bins as an exact float64 computation."""
    import math
    from univi.models.transformer import GenomicRelPosBias
    torch.manual_seed(0)
    rp = GenomicRelPosBias(num_heads=2, num_bins=16, max_dist=1e6)
    with torch.no_grad():
        rp.bias.copy_(torch.arange(32, dtype=torch.float32).view(2, 16))   # bias value encodes the bin
    chrom = torch.tensor([[0, 0, 0, 1, 1, -1]], dtype=torch.float32)
    pos = torch.tensor([[1.0e8, 1.0e8 + 1500, 2.4e8, 5.0e7, 5.0e7 + 20, 0.0]], dtype=torch.float32)
    out = rp(torch.stack([chrom, pos], dim=-1))[0, 0]                     # head 0: bias == bin index
    p64 = pos.double()[0]
    d = (p64[:, None] - p64[None, :]).abs().clamp(max=1e6)
    ref = (torch.log1p(d) / math.log1p(1e6) * 15).long()
    same = chrom[0][:, None] == chrom[0][None, :]
    assert torch.equal(out.long()[same], ref[same])                      # within a chromosome: exact bins
    assert torch.all(out.long()[~same] == 15)                            # across chromosomes: farthest bin
    out.sum().backward()
    assert rp.bias.grad is not None and rp.bias.grad.abs().sum() > 0
