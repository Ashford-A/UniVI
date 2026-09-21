"""Staged, label-masked refinement of an existing UniVI reference.

Refinement adds supervision (e.g. cell-type or mutation labels) to a trained
UniVI reference: classification heads are warmed up on frozen encoders, then
selected encoders are fine-tuned at a low learning rate while decoders stay
frozen. Missing labels are masked per head. This is the workflow used for
Figs. 5 and 7 of the Genome Research article; the generative training objective
itself is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F


@dataclass
class RefinementConfig:
    """Settings for head warmup followed by encoder/head fine-tuning.

    ``max_epochs`` includes warmup. ``latent_weight`` anchors posterior means
    to an eval-mode copy of the original reference. ``replay_weight`` adds the
    original generative objective on a separate paired training loader.
    """
    max_epochs: int = 1000
    warmup_epochs: int = 10
    lr_head: float = 1e-4
    lr_encoder: float = 1e-5
    weight_decay_head: float = 1e-6
    weight_decay_encoder: float = 1e-6
    latent_weight: float = 0.0
    replay_weight: float = 0.0
    replay_epoch: int = 10000
    patience: int = 50
    min_delta: float = 0.0
    grad_clip: float = 5.0
    log_every: int = 25

    def __post_init__(self):
        if self.max_epochs < 1 or not 0 <= self.warmup_epochs <= self.max_epochs:
            raise ValueError("Require max_epochs >= 1 and 0 <= warmup_epochs <= max_epochs.")
        if self.lr_head <= 0 or self.lr_encoder <= 0 or self.patience < 1:
            raise ValueError("Learning rates and patience must be positive.")
        for name in ("weight_decay_head", "weight_decay_encoder", "latent_weight",
                     "replay_weight", "min_delta", "grad_clip"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")


def _to_device(value, device):
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device)


def _posterior_mean(model, x):
    mu, lv = model.encode_modalities(x)
    return model.mixture_of_experts(mu, lv)[0]


def _supervised_sums(model, z, labels):
    """Return weighted loss sum and actual observed-label count.

    Missing entries never enter CE/BCE, including NaN and ignore_index.
    All labeled gene/cell entries carry one denominator unit.
    """
    unknown = set(labels) - set(model.class_heads)
    if unknown:
        raise KeyError(f"Targets name unregistered heads: {sorted(unknown)}")
    total = z.sum() * 0.0
    count = 0
    for name, head in model.class_heads.items():
        if name not in labels:
            continue
        cfg = model.class_heads_cfg[name]
        if cfg["loss_weight"] == 0:
            continue
        y = labels[name].reshape(-1)
        if y.shape[0] != z.shape[0]:
            raise ValueError(f"Head {name}: labels and input batch lengths differ.")
        mask = torch.isfinite(y) & (y != cfg["ignore_index"]) & (y >= 0)
        if not mask.any():
            continue
        logits = head(z)["logits"]
        target = y[mask]
        if cfg["type"] == "binary":
            if not torch.all((target == 0) | (target == 1)):
                raise ValueError(f"Head {name}: observed binary targets must be 0 or 1.")
            loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1)[mask], target.float(), reduction="sum",
                pos_weight=logits.new_tensor(cfg["pos_weight"]),
            )
        else:
            if not torch.all((target == target.long()) & (target < cfg["n_classes"])):
                raise ValueError(f"Head {name}: categorical targets must be valid integer codes.")
            loss = F.cross_entropy(logits[mask], target.long(), reduction="sum")
        total = total + cfg["loss_weight"] * loss
        count += int(mask.sum())
    return total, count


def _repeat_loader(loader):
    # Recreate the iterator each pass; unlike itertools.cycle, do not cache batches.
    while True:
        yield from loader


class UniVIRefiner:
    """Refine in place; retain a copy of the original model for comparisons.

    Loaders yield ``(x_dict, targets_dict)`` from ``MultiModalDataset`` with
    ``collate_multimodal_xy_recon``. Use separate loaders for independent
    RNA-only / ATAC-only / ADT-only cohorts. Shorter training loaders cycle so
    each cohort contributes the same number of optimization steps per epoch.
    Validation uses every labeled entry once. Labels must not be modalities.

    The supported path is the analytic fusion model without a label encoder or
    adversarial heads. Routers and all generative decoders stay frozen.
    """

    def __init__(self, model, train_loaders, val_loaders, *, config=None,
                 device="cpu", encoder_modalities=None, replay_loader=None):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cfg = config or RefinementConfig()
        self.train_loaders = list(train_loaders)
        self.val_loaders = list(val_loaders)
        self.replay_loader = replay_loader
        self.encoder_modalities = list(model.modality_names if encoder_modalities is None
                                       else encoder_modalities)
        if not self.train_loaders or not self.val_loaders:
            raise ValueError("Provide nonempty training and validation loader lists.")
        if any(len(loader) == 0 for loader in self.train_loaders + self.val_loaders):
            raise ValueError("Every refinement loader must contain at least one batch.")
        if not model.class_heads:
            raise ValueError("Attach classification heads before constructing UniVIRefiner.")
        if model.fused_encoder_type != "moe" or model.label_encoder is not None:
            raise ValueError("Refinement supports analytic fusion without a label encoder.")
        if set(self.encoder_modalities) - set(model.modality_names):
            raise ValueError("Unknown encoder modality.")
        for cfg in model.class_heads_cfg.values():
            if cfg["adversarial"] or not cfg["from_mu"] or cfg["warmup"] != 0:
                raise ValueError("Use non-adversarial mean heads with head warmup=0; stage warmup is configured in RefinementConfig.")
        if self.cfg.replay_weight and (replay_loader is None or len(replay_loader) == 0):
            raise ValueError("replay_weight > 0 requires a nonempty paired training replay_loader.")
        self.teacher = deepcopy(model).eval()
        self.teacher.requires_grad_(False)
        self.history = []
        self.best_epoch = None
        self.best_val_loss = float("inf")

    def _stage(self, fine_tune):
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.grad = None
        self.model.freeze_encoders().freeze_decoders()
        self.model.class_heads.requires_grad_(True)
        head_params = list(self.model.class_heads.parameters())
        groups = [{"params": head_params, "lr": self.cfg.lr_head,
                   "weight_decay": self.cfg.weight_decay_head}]
        if fine_tune:
            self.model.unfreeze_encoders(self.encoder_modalities)
            head_ids = {id(p) for p in head_params}
            enc_params = [p for p in self.model.parameters()
                          if p.requires_grad and id(p) not in head_ids]
            if enc_params:
                groups.append({"params": enc_params, "lr": self.cfg.lr_encoder,
                               "weight_decay": self.cfg.weight_decay_encoder})
        self.optimizer = torch.optim.AdamW(groups)

    def _train_mode(self):
        self.model.train()
        # Keep frozen routers and any other fully frozen modules deterministic.
        for module in self.model.modules():
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()

    def _batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("Supervised batches must be (x_dict, targets_dict).")
        x, y = batch
        if not isinstance(x, Mapping) or not isinstance(y, Mapping):
            raise ValueError("Use a dictionary of targets keyed by head name.")
        return _to_device(x, self.device), _to_device(y, self.device)

    @torch.no_grad()
    def evaluate(self):
        """Observed-label-weighted validation CE/BCE; fail on all-unlabeled data."""
        self.model.eval()
        numerator, denominator = 0.0, 0
        for loader in self.val_loaders:
            for batch in loader:
                x, y = self._batch(batch)
                loss, count = _supervised_sums(self.model, _posterior_mean(self.model, x), y)
                numerator += float(loss)
                denominator += count
        if denominator == 0:
            raise ValueError("Validation has no observed labels for active heads.")
        return numerator / denominator

    def fit(self):
        """Train both stages and restore the lowest supervised validation loss."""
        self.evaluate()  # reject invalid validation before any weight update
        best_state = None
        bad = 0
        steps = max(len(loader) for loader in self.train_loaders)
        for epoch in range(self.cfg.max_epochs):
            fine = epoch >= self.cfg.warmup_epochs
            if epoch == 0 or epoch == self.cfg.warmup_epochs:
                self._stage(fine)
                bad = 0
            self._train_mode()
            streams = [_repeat_loader(loader) for loader in self.train_loaders]
            replay = _repeat_loader(self.replay_loader) if self.replay_loader is not None else None
            losses, observed = [], 0
            for _ in range(steps):
                for stream in streams:
                    x, y = self._batch(next(stream))
                    self.optimizer.zero_grad(set_to_none=True)
                    z = _posterior_mean(self.model, x)
                    numerator, count = _supervised_sums(self.model, z, y)
                    observed += count
                    if count == 0:
                        continue
                    loss = numerator / count
                    if fine and self.cfg.latent_weight:
                        with torch.no_grad():
                            target_z = _posterior_mean(self.teacher, x)
                        loss = loss + self.cfg.latent_weight * F.mse_loss(z, target_z)
                    if fine and self.cfg.replay_weight:
                        batch = next(replay)
                        if isinstance(batch, Mapping):
                            rx, rt = batch, None
                        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                            rx, rt = batch
                        else:
                            raise ValueError("Replay must contain inputs, optionally reconstruction targets; no labels.")
                        kwargs = {} if rt is None else {"recon_targets": _to_device(rt, self.device)}
                        output = self.model(_to_device(rx, self.device), epoch=self.cfg.replay_epoch, **kwargs)
                        loss = loss + self.cfg.replay_weight * output["loss"].mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite refinement loss.")
                    loss.backward()
                    if self.cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
                    self.optimizer.step()
                    losses.append(float(loss.detach()))
            if observed == 0:
                raise ValueError("Training has no observed labels for active heads.")
            val = self.evaluate()
            if not np.isfinite(val):
                raise FloatingPointError("Nonfinite validation loss.")
            row = {"epoch": epoch, "stage": "encoders+heads" if fine else "heads",
                   "train_loss": float(np.mean(losses)), "val_supervised_loss": val}
            self.history.append(row)
            if val < self.best_val_loss - self.cfg.min_delta:
                self.best_val_loss, self.best_epoch = val, epoch
                best_state = {k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if self.cfg.log_every and epoch % self.cfg.log_every == 0:
                print(f"Refinement {epoch}: {row['stage']}, validation={val:.5f}")
            # Always complete warmup before considering early stopping.
            if fine and bad >= self.cfg.patience:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return {"history": self.history, "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss}


@torch.no_grad()
def predict_heads_adata(model, adata, modality, *, device="cpu", batch_size=512):
    """Return per-head probabilities in the input AnnData row order, from .X."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.eval()
    pieces = {name: [] for name in model.class_heads}
    for start in range(0, adata.n_obs, batch_size):
        x = adata.X[start:start + batch_size]
        x = x.toarray() if sp.issparse(x) else np.asarray(x)
        probabilities = model.predict_heads(
            {modality: torch.as_tensor(x, dtype=torch.float32, device=device)},
            use_mean=True, inject_label_expert=False,
        )
        for name in pieces:
            pieces[name].append(probabilities[name].cpu().numpy())
    return {name: np.concatenate(chunks) if chunks else
            np.empty((0, model.class_heads_cfg[name]["out_dim"]), dtype=np.float32)
            for name, chunks in pieces.items()}
