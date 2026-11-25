"""The two observability-guided regularizers of OG-SH.

* `modewise_shrinkage_loss`  - mode-wise SH shrinkage (Sec. II-D, Eqs. 7-8)
* `opacity_dropout`          - observability-guided opacity dropout (Sec. II-E, Eqs. 10-11)
"""
import torch

from .observability import degree_band

__all__ = ["modewise_shrinkage_loss", "opacity_dropout"]


def modewise_shrinkage_loss(sh_features: torch.Tensor, eig: list, max_degree: int, tau: float = 1e-2) -> torch.Tensor:
    """L_OG = (1/M) sum_i sum_{l>=1} tr(F_{i,l}^T U_{i,l} F_{i,l}),  U = tau (G + tau I)^-1 = Q diag(1 - o_k) Q^T.

    Evaluated in the eigenbasis without forming U:
        tr(F^T U F) = sum_k tau / (lambda_k + tau) * || q_k^T F ||^2.
    sh_features: [M, (L+1)^2, 3] (GaussianModel.get_features); eig: output of `observability.modewise_eig`
    for the same M Gaussians. Gradients flow only into sh_features. Degree 0 is not regularized."""
    terms = []
    for ell in range(1, max_degree + 1):
        start, end = degree_band(ell)
        lam, Q = eig[ell - 1]                                     # [M,w], [M,w,w]
        proj = torch.einsum('nwk,nwc->nkc', Q, sh_features[:, start:end, :])   # q_k^T F_l   [M,w,3]
        weight = tau / (lam + tau)                                # 1 - o_{i,l,k}
        terms.append((weight * (proj ** 2).sum(dim=-1)).sum(dim=-1))
    return torch.stack(terms, dim=0).sum(dim=0).mean()


def opacity_dropout(obs: torch.Tensor, p_max: float = 0.2, alloc: str = "guided") -> torch.Tensor:
    """Per-Gaussian opacity multiplier for one training iteration.

    obs: [N, L] band observability. o_i = mean_l o_{i,l}; guided rule p_i = p_max (1 - o_i)          (Eq. 10)
    k_i ~ Bernoulli(1 - p_i), returned multiplier k_i / (1 - p_i), i.e. alpha_hat = mult * alpha    (Eq. 11)

    `alloc` (ablation only) keeps the mean drop rate of the guided rule but changes where drops go:
      'inverted': p_i proportional to o_i (well-observed Gaussians dropped more)
      'uniform' : p_i = mean_j p_max (1 - o_j) for every Gaussian
    p_i is capped at 0.95."""
    o = obs.mean(dim=1)
    guided = (1.0 - o) * p_max
    if alloc == "guided":
        p = guided
    elif alloc == "inverted":
        p = o * (guided.mean() / o.mean().clamp_min(1e-12))
    elif alloc == "uniform":
        p = torch.full_like(o, float(guided.mean()))
    else:
        raise ValueError(f"unknown alloc: {alloc!r}")
    p = p.clamp(0.0, 0.95)
    keep = torch.bernoulli(1.0 - p)
    return keep / (1.0 - p)
