"""Training-loop glue for OG-SH (Algorithm 1): observability cache -> opacity dropout -> mode-wise SH shrinkage."""
import torch

from .observability import observability, modewise_eig
from .regularizers import modewise_shrinkage_loss, opacity_dropout


class OGSH:
    """Holds the observability cache and exposes the two per-iteration hooks.

    Usage inside a 3DGS training loop:
        ogsh = OGSH(train_cam_centers, p_max=0.2, weight=0.01)
        ...
        opacity_mult = ogsh.dropout(gaussians, iteration)          # before render (None when inactive)
        image = render(..., opacity_mult=opacity_mult)
        loss = render_loss + ogsh.shrinkage_loss(gaussians)        # 0 when inactive
    """

    def __init__(self, cam_centers: torch.Tensor, p_max=0.2, weight=0.01, tau=1e-2, warmup=2000,
                 interval=500, max_shrink_pts=200_000, alloc="guided"):
        self.cams = cam_centers.float().cuda()
        self.p_max, self.weight, self.tau = p_max, weight, tau
        self.warmup, self.interval, self.max_shrink_pts, self.alloc = warmup, interval, max_shrink_pts, alloc
        self.cache = None
        self.active = False
        self.drop_hist, self.last_loss = [], None

    def _refresh(self, gaussians, iteration):
        n, deg = gaussians.get_xyz.shape[0], gaussians.active_sh_degree
        c = self.cache
        if c is not None and c["n"] == n and c["degree"] == deg and iteration % self.interval != 0:
            return
        with torch.no_grad():
            pts = gaussians.get_xyz.detach()
            obs = observability(pts, self.cams, deg, tau=self.tau)                         # [N, deg], Step 1
            idx = torch.randperm(n, device="cuda")[:self.max_shrink_pts] if n > self.max_shrink_pts else None
            eig = modewise_eig(pts if idx is None else pts[idx], self.cams, deg)              # shrinkage set S
        self.cache = {"n": n, "degree": deg, "obs": obs, "idx": idx, "eig": eig}

    def dropout(self, gaussians, iteration):
        """Step 1 (refresh if needed) + Step 2. Returns the per-Gaussian opacity multiplier, or None."""
        self.active = iteration > self.warmup and gaussians.active_sh_degree >= 1
        if not self.active:
            return None
        self._refresh(gaussians, iteration)
        mult = opacity_dropout(self.cache["obs"], p_max=self.p_max, alloc=self.alloc)
        self.drop_hist.append(float((mult == 0).float().mean()))
        return mult

    def shrinkage_loss(self, gaussians):
        """Step 3: lambda_OG * L_OG on the cached shrinkage set (0 when inactive)."""
        if not self.active or self.weight <= 0:
            return 0.0
        idx = self.cache["idx"]
        feats = gaussians.get_features if idx is None else gaussians.get_features[idx]
        loss = modewise_shrinkage_loss(feats, self.cache["eig"], gaussians.active_sh_degree, tau=self.tau)
        self.last_loss = float(loss.detach())
        return self.weight * loss

    def summary(self):
        if not self.drop_hist:
            return "OG-SH inactive"
        return (f"OG-SH drop fraction last {self.drop_hist[-1]:.1%} / mean {sum(self.drop_hist) / len(self.drop_hist):.1%}"
                f" | L_OG {self.last_loss}")
