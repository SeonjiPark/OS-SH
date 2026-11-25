"""OG-SH: Observability-Guided Spherical Harmonic Shrinkage for sparse-view 3D Gaussian Splatting."""
from .observability import eval_sh_basis, observability, modewise_eig
from .regularizers import modewise_shrinkage_loss, opacity_dropout
from .trainer import OGSH

__all__ = ["OGSH", "observability", "modewise_eig", "modewise_shrinkage_loss", "opacity_dropout", "eval_sh_basis"]
