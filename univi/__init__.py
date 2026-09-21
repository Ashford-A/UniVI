# univi/__init__.py

from __future__ import annotations

from typing import Any, List

__version__ = "0.5.0"

# Eager (fast/light) public API
from .config import ModalityConfig, UniVIConfig, TrainingConfig, ClassHeadConfig
from .models import UniVIMultiModalVAE
from . import matching

_WORKFLOW_EXPORTS = {
    "RefinementConfig": "refinement",
    "UniVIRefiner": "refinement",
    "predict_heads_adata": "refinement",
    "RNAPreprocessor": "preprocessing",
    "ADTPreprocessor": "preprocessing",
    "ATACPreprocessor": "preprocessing",
    "split_by_label": "preprocessing",
    "make_loader": "workflows",
    "stack_embeddings": "workflows",
    "save_reference": "workflows",
    "load_reference": "workflows",
    "predict_feature_perturbation": "perturbation",
    "load_scnmt_gastrulation_genebody_triplet": "datasets",
    "build_univi_inputs_from_scnmt_triplet": "datasets",
}

__all__ = [
    "__version__",
    # configs
    "ModalityConfig",
    "ClassHeadConfig",
    "UniVIConfig",
    "TrainingConfig",
    # model
    "UniVIMultiModalVAE",
    # lightweight module
    "matching",
    # model state
    "save_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
    # lazy exports
    "UniVITrainer",
    "write_univi_latent",
    "MultiModalDataset",
    "pipeline",
    "diagnostics",
    # modules
    "evaluation",
    "plotting",
    # eval convenience (optional)
    "encode_adata",
    "evaluate_alignment",
    # interpretability
    "interpretability",
    "fused_encode_with_meta_and_attn",
    "feature_importance_for_head",
    "top_cross_modal_feature_pairs_from_attn",
    *_WORKFLOW_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """
    Lazy exports keep `import univi` fast/light and avoid heavy deps unless needed.
    """
    if name in _WORKFLOW_EXPORTS:
        from importlib import import_module
        value = getattr(import_module(f".{_WORKFLOW_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value

    # ---- training ----
    if name == "UniVITrainer":
        from .trainer import UniVITrainer
        return UniVITrainer

    # ---- IO ----
    if name == "write_univi_latent":
        from .utils.io import write_univi_latent
        return write_univi_latent

    # ---- data ----
    if name == "MultiModalDataset":
        from .data import MultiModalDataset
        return MultiModalDataset

    # ---- model state ----
    if name in {"save_checkpoint", "load_checkpoint", "restore_checkpoint"}:
        from .utils.io import save_checkpoint, load_checkpoint, restore_checkpoint
        return {"save_checkpoint": save_checkpoint, "load_checkpoint": load_checkpoint, "restore_checkpoint": restore_checkpoint}[name]

    # ---- modules (avoid recursive from-list resolution in __getattr__) ----
    if name in {"pipeline", "diagnostics", "evaluation", "plotting", "interpretability"}:
        from importlib import import_module
        return import_module(f".{name}", __name__)

    if name in {
        "fused_encode_with_meta_and_attn",
        "feature_importance_for_head",
        "top_cross_modal_feature_pairs_from_attn",
    }:
        from .interpretability import (
            fused_encode_with_meta_and_attn,
            feature_importance_for_head,
            top_cross_modal_feature_pairs_from_attn,
        )
        return {
            "fused_encode_with_meta_and_attn": fused_encode_with_meta_and_attn,
            "feature_importance_for_head": feature_importance_for_head,
            "top_cross_modal_feature_pairs_from_attn": top_cross_modal_feature_pairs_from_attn,
        }[name]

    # ---- eval convenience functions (re-export) ----
    if name in {"encode_adata", "evaluate_alignment"}:
        from .evaluation import encode_adata, evaluate_alignment
        return {"encode_adata": encode_adata, "evaluate_alignment": evaluate_alignment}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)
