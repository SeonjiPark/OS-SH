"""Directional observability of SH coefficients (OG-SH, Sec. II-B/C).

For a Gaussian i with centre mu_i and V training cameras o_v, the SH design matrix is
    A_i = [Y(omega_{i,v})^T]_v,   omega_{i,v} = (o_v - mu_i) / ||o_v - mu_i||        (Eq. 3)
For each degree l >= 1 the degree-l block is residualised against all lower degrees,
    A^cond_{i,l} = (I - P_{i,<l}) A_{i,l}                                             (Eq. 4)
and its Gram matrix G_{i,l} = A^cond^T A^cond = Q diag(lambda) Q^T gives the modes   (Eq. 5)
with mode-wise observability o_{i,l,k} = lambda_k / (lambda_k + tau)                (Eq. 6).

Everything here is computed without gradient from a detached snapshot of the Gaussian centres.
"""
import torch

__all__ = ["eval_sh_basis", "observability", "modewise_eig"]

# Real SH constants (same convention as the 3DGS rasterizer; degree l occupies indices [l^2, l^2 + 2l]).
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]


def eval_sh_basis(max_deg: int, dirs: torch.Tensor) -> torch.Tensor:
    """Real SH basis values. dirs: [..., 3] unit vectors -> [..., (max_deg + 1)^2]."""
    assert 0 <= max_deg <= 3, "only degrees 0-3 are implemented"
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = [torch.full_like(x, C0)]
    if max_deg > 0:
        out += [-C1 * y, C1 * z, -C1 * x]
    if max_deg > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        out += [C2[0] * xy, C2[1] * yz, C2[2] * (2.0 * zz - xx - yy),
                C2[3] * xz, C2[4] * (xx - yy)]
    if max_deg > 2:
        xx, yy, zz = x * x, y * y, z * z
        xy = x * y
        out += [C3[0] * y * (3 * xx - yy),
                C3[1] * xy * z,
                C3[2] * y * (4 * zz - xx - yy),
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                C3[4] * x * (4 * zz - xx - yy),
                C3[5] * z * (xx - yy),
                C3[6] * x * (xx - 3 * yy)]
    return torch.stack(out, dim=-1)


def degree_band(ell: int) -> tuple:
    """Column range [start, end) of degree ell in the stacked SH basis."""
    return ell * ell, ell * ell + 2 * ell + 1


def _view_directions(points: torch.Tensor, cam_centers: torch.Tensor) -> torch.Tensor:
    """points [N,3], cam_centers [V,3] -> unit directions from each Gaussian to each camera [N,V,3]."""
    diff = cam_centers[None, :, :] - points[:, None, :]
    return diff / (diff.norm(dim=-1, keepdim=True) + 1e-12)


def _conditional_grams(points, cam_centers, max_degree, rank_tol_rel=1e-4):
    """Yield (ell, G_{i,ell}) for ell = 1..max_degree, G = (A^cond)^T A^cond of shape [N, 2l+1, 2l+1].

    The projector P_{<l} uses a truncated pseudo-inverse of A_{<l}^T A_{<l}: eigenvalues below
    rank_tol_rel * lambda_max are discarded (Sec. II-F)."""
    basis = eval_sh_basis(max_degree, _view_directions(points, cam_centers))   # [N,V,(L+1)^2]
    for ell in range(1, max_degree + 1):
        start, end = degree_band(ell)
        A_low, A_l = basis[..., :start], basis[..., start:end]
        G_low = A_low.transpose(-1, -2) @ A_low
        evals, evecs = torch.linalg.eigh(G_low)
        tol = evals.max(dim=-1, keepdim=True).values.clamp(min=1e-12) * rank_tol_rel
        inv = torch.where(evals > tol, 1.0 / evals.clamp(min=1e-12), torch.zeros_like(evals))
        G_low_pinv = (evecs * inv.unsqueeze(-2)) @ evecs.transpose(-1, -2)
        A_cond = A_l - A_low @ (G_low_pinv @ (A_low.transpose(-1, -2) @ A_l))      # (I - P_<l) A_l
        yield ell, A_cond.transpose(-1, -2) @ A_cond


@torch.no_grad()
def observability(points: torch.Tensor, cam_centers: torch.Tensor, max_degree: int,
                  tau: float = 1e-2, rank_tol_rel: float = 1e-4, chunk: int = 100_000) -> torch.Tensor:
    """Band observability o_{i,l} = mean_k lambda_k / (lambda_k + tau), l = 1..max_degree.

    points [N,3], cam_centers [V,3] -> [N, max_degree] in [0, 1]. Processed in chunks of `chunk` Gaussians."""
    out = []
    for s in range(0, points.shape[0], chunk):
        scores = []
        for _, G in _conditional_grams(points[s:s + chunk], cam_centers, max_degree, rank_tol_rel):
            lam = torch.linalg.eigvalsh(G).clamp(min=0.0)
            scores.append((lam / (lam + tau)).mean(dim=-1))
        out.append(torch.stack(scores, dim=-1).clamp(0.0, 1.0))
    return torch.cat(out, dim=0)


@torch.no_grad()
def modewise_eig(points: torch.Tensor, cam_centers: torch.Tensor, max_degree: int,
                 rank_tol_rel: float = 1e-4) -> list:
    """Eigen-decomposition of G_{i,l} for l = 1..max_degree.

    Returns a list of (eigvals [N, 2l+1] ascending and >= 0, eigvecs [N, 2l+1, 2l+1]); column k of eigvecs is
    the mode q_{i,l,k}. Cached between observability refreshes and consumed by `modewise_shrinkage_loss`."""
    eig = []
    for _, G in _conditional_grams(points, cam_centers, max_degree, rank_tol_rel):
        lam, Q = torch.linalg.eigh(G)
        eig.append((lam.clamp(min=0.0), Q))
    return eig
