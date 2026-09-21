"""Workflow helpers: data loaders, stacked embeddings, and reference bundles."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import (ClassHeadConfig, ModalityConfig, UniVIConfig,
                     TransformerConfig, TokenizerConfig)
from .data import MultiModalDataset, collate_multimodal_xy_recon
from .evaluation import encode_adata
from .models import UniVIMultiModalVAE


def make_loader(adata_by_mod, *, batch_size=256, shuffle=False, labels=None,
                recon_targets_spec=None, drop_last=False, seed=0):
    """Build a CPU dataset with the required UniVI collator.

    A mapping with multiple modalities must contain genuinely paired cells.
    Use separate loaders for unrelated cohorts. BatchNorm training requires
    at least two cells per batch; use drop_last=True for a singleton remainder.
    """
    dataset = MultiModalDataset(adata_by_mod, labels=labels, device=None,
                               recon_targets_spec=recon_targets_spec)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=drop_last,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=collate_multimodal_xy_recon)


def stack_embeddings(model, datasets, *, device="cpu", batch_size=1024):
    """Encode (cohort, modality, AnnData) entries and stack modality means.

    Each input cell contributes one row per supplied modality. Features in the
    result are latent dimensions, not a union of incompatible assay features.
    ``cell_id`` retains the input barcode and ``block`` the tuple's cohort label.
    No UMAP is fit and no biological pairing is inferred by this function.
    """
    blocks = []
    names = set()
    for cohort, modality, data in datasets:
        block = f"{cohort}:{modality}"
        if block in names:
            raise ValueError("Use unique cohort/modality combinations.")
        names.add(block)
        z = encode_adata(model, data, modality=modality, device=device,
                         batch_size=batch_size, latent="modality_mean")
        obs = data.obs.copy()
        obs["cell_id"] = data.obs_names.astype(str)
        obs["cohort"], obs["modality"], obs["block"] = cohort, modality, block
        obs.index = pd.Index([f"{block}::{cell}" for cell in data.obs_names])
        blocks.append(ad.AnnData(z, obs=obs))
    if not blocks:
        raise ValueError("Supply at least one dataset.")
    result = ad.concat(blocks, merge="same", index_unique=None)
    result.obsm["X_univi"] = result.X.copy()
    return result


def save_reference(directory, model, *, preprocessors=None, metadata=None):
    """Save model configuration, objective switches, labels, and fitted transforms.

    The directory contains model.pt and optional preprocessing.joblib. Save
    cell split maps and acquisition details in metadata. No optimizer state is
    saved: this bundle is for inference/new training stages, not exact resume.
    """
    directory = Path(directory)
    if preprocessors is None and (directory / "preprocessing.joblib").exists():
        raise ValueError("Existing preprocessing.joblib would become stale; use a new output directory.")
    metadata_text = json.dumps(metadata or {}, indent=2) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    kwargs = {key: getattr(model, key) for key in (
        "loss_mode", "v1_recon", "v1_recon_mix", "normalize_v1_terms",
        "recon_normalize_by_dim", "recon_dim_power", "n_label_classes",
        "label_loss_weight", "label_moe_weight", "unlabeled_logvar",
        "label_encoder_warmup", "label_ignore_index", "classify_from_mu", "label_head_name")}
    kwargs["use_label_encoder"] = model.label_encoder is not None
    torch.save({"format": "univi-reference-1", "config": asdict(model.cfg),
                "model_kwargs": kwargs, "state_dict": model.state_dict(),
                "head_label_names": model.head_label_names,
                "label_names": getattr(model, "label_names", None),
                "requires_grad": {k: p.requires_grad for k, p in model.named_parameters()},
                "frozen_eval_modules": sorted(getattr(model, "_frozen_eval_modules", ()))},
               directory / "model.pt")
    if preprocessors is not None:
        joblib.dump(preprocessors, directory / "preprocessing.joblib")
    (directory / "metadata.json").write_text(metadata_text)


def load_reference(directory, *, device="cpu", load_preprocessors=True):
    """Load a bundle created by save_reference; use only trusted local bundles.

    Joblib preprocessing files use Python pickle. The returned tuple is
    (model, preprocessors, metadata). Restore saved transforms; do not refit them.
    """
    directory = Path(directory)
    payload = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    if payload.get("format") != "univi-reference-1":
        raise ValueError("Expected a UniVI reference bundle.")
    config = payload["config"]
    modalities = []
    for item in config["modalities"]:
        item = dict(item)
        for key, cls in (("transformer", TransformerConfig), ("tokenizer", TokenizerConfig)):
            if item.get(key) is not None:
                item[key] = cls(**item[key])
        modalities.append(ModalityConfig(**item))
    config["modalities"] = modalities
    if config.get("class_heads") is not None:
        config["class_heads"] = [ClassHeadConfig(**item) for item in config["class_heads"]]
    if config.get("fused_transformer") is not None:
        config["fused_transformer"] = TransformerConfig(**config["fused_transformer"])
    model = UniVIMultiModalVAE(UniVIConfig(**config), **payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    for name, labels in payload["head_label_names"].items():
        model.set_head_label_names(name, labels)
    if payload.get("label_names"):
        model.set_label_names(payload["label_names"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(payload["requires_grad"][name])
    model._frozen_eval_modules = set(payload["frozen_eval_modules"])
    model.eval()
    prep_path = directory / "preprocessing.joblib"
    prep = joblib.load(prep_path) if load_preprocessors and prep_path.exists() else {}
    metadata = json.loads((directory / "metadata.json").read_text())
    return model, prep, metadata
