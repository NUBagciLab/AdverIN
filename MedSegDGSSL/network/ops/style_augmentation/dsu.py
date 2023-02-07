"""
Implementation of Batch-Instance Normalization
https://arxiv.org/abs/2202.03958

Based on the official implementation:
https://github.com/lixiaotong97/DSU.git
"""
import torch
import torch.nn as nn
import numpy as np


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        prob   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, prob=0.5, factor:float=1., eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = prob
        self.factor = factor
    
    def __repr__(self):
        return f'DSU(p={self.p}, eps={self.eps})'

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        x_flatten = torch.flatten(x, start_dim=2)
        mean = x_flatten.mean(dim=2, keepdim=False)
        std = (x_flatten.var(dim=2, keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        shape = [s if i<2 else 1 for i, s in enumerate(list(x.size()))]

        x = (x - mean.reshape(*shape)) / std.reshape(*shape)
        x = x * gamma.reshape(*shape) + beta.reshape(*shape)

        return x
