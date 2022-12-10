""" Here is to implement the input augmentation
Including:

Mixup: https://arxiv.org/abs/1710.09412

"""

import torch
import torch.nn as nn
import numpy as np


class MixUp(nn.Module):
    """Random Mixup Based input augmentation
    https://arxiv.org/abs/2007.13003

    args:
        prob: float, the possibility for random mixup, default 0.5
        alpha: determine the shape for augmentation
        preserve_order: bool, to determine whether preserve order 
    """
    def __init__(self, alpha=0.5, prob=0.5, preserve_order=False):
        super().__init__()
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.prob = prob
        self.preserve_order = preserve_order
    
    def forward(self, x1, x2, y1, y2):
        # determine the sample shape
        if not self.training or np.random.random() > self.prob:
            return x1, 1, y1, y2

        sample_shape = [s if i==0 else 1 for i, s in enumerate(x1.shape)]
        lmda = self.beta.sample(sample_shape=sample_shape)
        if self.preserve_order:
            lmda = torch.max(lmda, 1 - lmda)

        xmix1 = x1*lmda + x2 * (1-lmda)
        return xmix1, lmda, y1, y2
