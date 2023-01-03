"""
Implementation of RSC
Self-Challenging Improves Cross-Domain Generalization

https://arxiv.org/abs/2202.03958

Based on the official implementation:
https://github.com/lixiaotong97/DSU.git
"""
import torch
import torch.nn as nn
import numpy as np


class RSC(nn.Module):
    def __init__(self, percentile=0.95) -> None:
        super().__init__()
        self.is_challenge = False
        self.percentile = percentile
        self.x_mask = None
    
    def forward(self, x):
        if not self.training or not self.is_challenge:
            return x
        #######
        # Mask the value using the gradient
        #######
        x = self.x_mask * x
        return x

    def backward(self, x_grad):
        if not self.is_challenge:
            x_size = x_grad.shape
            x_mask = torch.flatten(x_grad, dims=1)
            x_index = int(x_mask.size(1)*self.percentile)
            x_percentile = torch.sort(x_mask, dim=1)[:, x_index].unsqueeze(1)
            x_mask = (x_percentile-x_mask)>0
            self.x_mask = torch.reshape(x_mask, x_size)
        else:
            x_grad = self.x_mask * x_grad
        return x_grad

    def set_challenge(self, status:bool=True):
        self.is_challenge = status
