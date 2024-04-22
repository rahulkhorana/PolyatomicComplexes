import torch
import numpy as np


class CRPS:
    def __init__(self, x_test: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
        self.pred = torch.from_numpy(x_test)
        self.truth = torch.from_numpy(y_test)

    def crps(self):
        """
        credit: https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html
        adapted from pyro.ops.stats
        :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
            This should have shape ``(num_samples,) + truth.shape``.
        :param torch.Tensor truth: A tensor of true observations.
        :return: A tensor of shape ``truth.shape``.
        :rtype: torch.Tensor
        :used per the Apache-2.0 license
        """
        opts = dict(device=self.pred.device, dtype=self.pred.dtype)
        num_samples = self.pred.size(0)
        if num_samples == 1:
            return (self.pred[0] - self.truth).abs()

        self.pred = self.pred.sort(dim=0).values
        diff = self.pred[1:] - self.pred[:-1]
        weight = torch.arange(1, num_samples, **opts) * torch.arange(
            num_samples - 1, 0, -1, **opts
        )
        assert isinstance(weight, torch.Tensor)
        weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

        return (self.pred - self.truth).abs().mean(0) - (diff * weight).sum(
            0
        ) / num_samples**2
