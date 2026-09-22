"""Datasets used in the UniVI tutorials and the Genome Research article.

Quick use::

    import univi.datasets as uds

    uds.list_datasets()                    # what is registered / downloadable
    data = uds.pbmc_multiome_10k()         # {"rna": AnnData, "atac": AnnData}
    data = uds.load("pbmc_multiome_10k")   # same thing, by name

Files are cached under ``UNIVI_DATA_DIR`` (default ``~/.cache/univi``) and
verified against registered checksums. The scNMT-seq bundle readers used for
Supplemental Fig. S4 are also available from this module. Ready-to-load
RNA/CpG/GpC objects are available with ``scnmt_gastrulation()`` or
``load("scnmt_gastrulation")`` from https://zenodo.org/records/22885463
(DOI: 10.5281/zenodo.22885463).
"""
from ._export import export_dataset
from ._registry import (
    DatasetNotAvailableError,
    dataset_info,
    fetch,
    get_data_dir,
    hao_citeseq_pbmc,
    list_datasets,
    load,
    pbmc_multiome_10k,
    register_dataset,
    scnmt_gastrulation,
)
from .scnmt import (
    attach_metadata_and_align_modalities,
    build_methyl_like_anndata_from_long_feature,
    build_univi_inputs_from_scnmt_triplet,
    filter_methyl_features_by_coverage,
    filter_rna_genes_basic,
    list_tar_members,
    load_scnmt_gastrulation_genebody_triplet,
    read_feature_level_long_from_tar,
    read_metadata_from_tar,
    read_rna_counts_from_tar,
)

__all__ = [
    "DatasetNotAvailableError",
    "export_dataset",
    "list_datasets",
    "dataset_info",
    "register_dataset",
    "get_data_dir",
    "fetch",
    "load",
    "pbmc_multiome_10k",
    "hao_citeseq_pbmc",
    "scnmt_gastrulation",
    "list_tar_members",
    "read_rna_counts_from_tar",
    "read_feature_level_long_from_tar",
    "read_metadata_from_tar",
    "build_methyl_like_anndata_from_long_feature",
    "attach_metadata_and_align_modalities",
    "filter_rna_genes_basic",
    "filter_methyl_features_by_coverage",
    "load_scnmt_gastrulation_genebody_triplet",
    "build_univi_inputs_from_scnmt_triplet",
]
