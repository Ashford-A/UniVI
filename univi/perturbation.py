"""Model-sensitivity experiments in an explicitly named input feature space."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from .evaluation import cross_modal_predict


def predict_feature_perturbation(model, adata, *, source_modality,
                                target_modality, features, mode="off", value=0.,
                                device="cpu", batch_size=256):
    """Compare baseline and edited-input decoder means without retraining.

    ``adata.X`` must already be in the model's input space. ``features`` are exact
    source var_names, not latent coordinates or genes mapped heuristically to
    peaks. Supported edits are off, set, add, scale. This reports model sensitivity,
    not a causal perturbation effect. The input object is never modified.
    """
    if isinstance(features, str):
        features = [features]
    else:
        features = list(features)
    if not features or not adata.var_names.is_unique:
        raise ValueError("Provide nonempty features and unique feature identifiers.")
    if mode not in {"off", "set", "add", "scale"}:
        raise ValueError("mode must be off, set, add, or scale.")
    if not np.isfinite(value):
        raise ValueError("value must be finite.")
    indices = adata.var_names.get_indexer(features)
    if (indices < 0).any():
        raise KeyError("Some requested source features are missing.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Perturbation feature names must be unique.")
    altered = adata.copy()
    # Raw count matrices may be integer typed; fractional set/scale operations
    # must not be silently truncated by assignment back into the input dtype.
    x = (altered.X.astype(np.float32).tolil(copy=True) if sp.issparse(altered.X)
         else np.array(altered.X, dtype=np.float32, copy=True))
    before = x[:, indices].toarray() if sp.issparse(x) else x[:, indices].copy()
    after = (np.zeros_like(before) if mode == "off" else
             np.full_like(before, value) if mode == "set" else
             before + value if mode == "add" else before * value)
    likelihood = model.mod_cfg_by_name[source_modality].likelihood
    if likelihood in {"nb", "zinb", "poisson", "bernoulli"} and (after < 0).any():
        raise ValueError("Count/binary inputs cannot become negative.")
    if likelihood == "bernoulli" and not np.isin(after, [0., 1.]).all():
        raise ValueError("Bernoulli inputs must stay binary.")
    x[:, indices] = after
    altered.X = x.tocsr() if sp.issparse(x) else x
    baseline = cross_modal_predict(model, adata, source_modality, target_modality,
                                   device=device, batch_size=batch_size)
    prediction = cross_modal_predict(model, altered, source_modality, target_modality,
                                     device=device, batch_size=batch_size)
    return {"baseline": baseline, "perturbed": prediction, "delta": prediction-baseline,
            "features": features, "mode": mode, "value": float(value)}
