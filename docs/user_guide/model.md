# How the model works

## Encoders, posteriors, decoders

Every modality $m$ has an encoder that maps a cell's measurements $x_m$ to a Gaussian posterior over a shared latent space,
$q_m(z \mid x_m) = \mathcal{N}(\mu_m, \operatorname{diag}\sigma_m^2)$, and a decoder $p_m(x_m \mid z)$ with a likelihood chosen for that modality. Because all encoders target the same space and all decoders read from it, any encoder can be paired with any decoder: encoding ATAC and decoding RNA is cross-modal prediction.

## Fusing modalities

When several modalities are observed for a cell, their posteriors are combined per latent dimension by precision weighting (a product of Gaussian experts):

$$
\mu_{\text{fused}} = \frac{\sum_m \tau_m \mu_m}{\sum_m \tau_m}, \qquad \tau_m = \sigma_m^{-2}, \qquad \sigma^2_{\text{fused}} = \Big(\sum_m \tau_m\Big)^{-1}.
$$

A modality that is confident about a cell (small posterior variance) contributes more. With a single observed modality, the fused posterior is just that modality's posterior.

## The training objective (`loss_mode="v1"`)

The default objective, used for every analysis in the paper, has three parts:

- **Reconstruction.** With `v1_recon="avg"` (default), every observed modality is reconstructed from its own latent sample (self-reconstruction) and from every other modality's sample (cross-reconstruction). With $K$ modalities, self terms have weight $0.5/K$ each and cross terms $0.5/(K(K-1))$ each, so both kinds contribute half.
- **KL to the prior**, weighted by `beta`: keeps each posterior close to $\mathcal{N}(0, I)$, averaged over modalities.
- **Cross-modal alignment**, weighted by `gamma`: the KL divergence between the posteriors of every ordered pair of modalities for the same cell, averaged over pairs. This is what pulls a cell's RNA and ATAC embeddings together.

`beta` and `gamma` can ramp linearly from 0 to their full values between `kl_anneal_start`/`kl_anneal_end` and `align_anneal_start`/`align_anneal_end` (in epochs). Validation loss always uses the full values.

Other reconstruction choices (`v1_recon`): `"self"` (self-reconstruction only), `"moe"` (reconstruct every modality from the fused sample), `"avg_z"` (from the average of the modality samples), `"src:<name>"` (from one source modality).

`loss_mode="v2"` ("lite") instead reconstructs all modalities from the fused posterior, with one KL term for the fused posterior and an L2 penalty between modality means; by default it also divides each reconstruction term by the modality's feature count. It is the mode to use with the multimodal transformer encoder and can suit loosely paired data, but it generally gives looser cell-to-cell correspondence between modalities than `"v1"`, which is the objective used throughout the paper.

## Which latent representation to use

| Representation | How to get it | Use it for |
| --- | --- | --- |
| Per-modality posterior mean | `encode_adata(model, adata, modality="rna", latent="modality_mean")` | comparing modalities, paired-cell metrics, unimodal query data |
| Fused posterior mean | `encode_fused_adata_pair(model, {"rna": ..., "atac": ...})` | one point per cell when all modalities were measured: clustering, annotation, UMAPs |
| Stacked embeddings | `stack_embeddings(model, [(cohort, modality, adata), ...])` | one row per cell per modality: joint UMAPs that show modality mixing |

`encode_adata(..., latent="moe_mean")` (its default) fuses whatever modalities it is given. Given one modality, that is the same as `"modality_mean"`. `"modality_sample"` and `"moe_sample"` draw a sample instead of the mean.

Do not assign the same fused embedding to both modalities and then measure their alignment: paired distances are zero by construction.

(modality-weights-and-gating)=
## Modality weights and gating

Two different quantities describe how much each modality contributes to a fused embedding.

**Precision weights** are always available. `encode_moe_gates_from_tensors(model, x_dict, kind="effective_precision")` returns, per cell, each modality's share of the total posterior precision (precisions summed over latent dimensions, then normalized across modalities). These are the modality contributions shown in the paper.

**Learned gates** come from an optional gating network (`UniVIConfig(use_moe_gating=True)`) that re-weights the precisions per cell before fusion. Read them with `kind="router_x_precision"`. The gating network only learns when the fused posterior is part of the loss:

- `loss_mode="v1"` with `v1_recon="moe"`, or
- `loss_mode="v2"`, or
- classification heads, which read the fused mean.

With the default `v1_recon="avg"` and no heads, the gating network receives no gradient and keeps its random initial weights; UniVI warns about this when the model is built.

Without a gating network, the `gates` returned by `encode_fused_adata_pair` are uniform (for example 0.5/0.5); use the precision weights above instead.

## Architecture settings

`ModalityConfig(name, input_dim, encoder_hidden, decoder_hidden, likelihood=...)` defines each modality. Encoders and decoders are MLPs by default; `encoder_type="transformer"` swaps in a transformer encoder (see [Advanced features](advanced.md)). Model-wide settings live in `UniVIConfig`: `latent_dim`, `beta`, `gamma`, dropout (`encoder_dropout`, `decoder_dropout`), BatchNorm (`encoder_batchnorm`, default on; `decoder_batchnorm`, default off), and the annealing windows.

`ModalityConfig.recon_weight` rescales one modality's reconstruction term. `UniVIMultiModalVAE(..., recon_normalize_by_dim=True, recon_dim_power=0.5)` divides each reconstruction term by (number of features)^power, which keeps a 20,000-gene modality from overwhelming a 50-protein one.

### Implemented likelihoods

`gaussian` (alias `normal`), `gaussian_diag`, `nb` (`negative_binomial`; gene-wise or global dispersion via `dispersion`), `zinb`, `poisson`, `bernoulli`, `beta`, `binomial`, `beta_binomial`, `logistic_normal`, and `categorical`. Count decoders predict counts directly; library-size offsets are not modeled, so the `use_library_size` and `library_key` fields of `ModalityConfig` currently have no effect.
