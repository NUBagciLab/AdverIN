"""
The implementation of pAdaIN:
https://arxiv.org/abs/2010.05785

Implementation of the pAdaIN layer. This layer can be added after every convolutional layer and acts
as a regularization which increases overall performance.
It is only applied during training.
"""

import torch
import torch.nn as nn
import numpy as np


class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, prob=0.01, eps=1e-5):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = prob
        self.eps = eps

    def forward(self, x):
        if np.random.random()>self.p or not self.training:
            return x

        perm_indices = torch.randperm(x.size()[0])
        return adaptive_instance_normalization(x, x[perm_indices], self.eps)

    def extra_repr(self) -> str:
        return 'adain p={}'.format(
            self.p
        )


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)