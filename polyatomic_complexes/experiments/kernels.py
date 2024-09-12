"""
The Tanimoto Kernel implementation coes from the Gauche library for chemistry,
this implementation of the Tanimoto Kernel comes from the GAUCHE library for Chemistry
# Credit: https://github.com/leojklarner/gauche
# Original Paper: https://arxiv.org/abs/2212.04450
# We make use of this code per the MIT license from the original repo
"""

from gpytorch.kernels import Kernel
import torch


class TanimotoKernel(Kernel):
    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def batch_tanimoto_sim(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        if x1.ndim < 2 or x2.ndim < 2:
            raise ValueError("Tensors must have a batch dimension")
        dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
        x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
        x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)
        tan_similarity = (dot_prod + eps) / (
            eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
        )
        return tan_similarity.clamp_min_(
            0
        )  # zero out negative values for numerical stability

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(self, x1, x2, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        return self.batch_tanimoto_sim(x1, x2)


class WalkKernel(Kernel):
    def __init__(self, **kwargs):
        super(WalkKernel, self).__init__(**kwargs)

    def batch_walk_sim(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:

        return


class GraphKernel(Kernel):
    def __init__(self, **kwargs):
        super(GraphKernel, self).__init__(**kwargs)

    def batch_graph_sim(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:

        return
