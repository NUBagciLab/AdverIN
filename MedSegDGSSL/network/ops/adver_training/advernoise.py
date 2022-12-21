"""
Implement the adversarial noise attacking for increasing model's robustness

Based on "Virtual Adversarial Training: 
A Regularization Method for Supervised and Semi-Supervised Learning"

https://arxiv.org/abs/1704.03976
"""

import torch
import torch.nn as nn 

from .reverse_grad import ReverseNormGrad
from MedSegDGSSL.network.build import NETWORK_REGISTRY

class AdverNoise(nn.Module):
    """ Adversarial Noise attacking
    Note that the attacking is based on the case level
    args:
        input_size: the input size with batch dimension
        p: L_p distance for norm calculation, default 2
        grad_norm: changed grad norm
    """
    def __init__(self, input_size, p:int=2, grad_norm:float=1.):
        super().__init__()
        self.input_size = input_size
        self.p = p
        self.grad_norm = grad_norm
        self.reverse_grad = ReverseNormGrad()
        self.params = nn.parameter.Parameter(torch.zeros(size=self.input_size),
                                             requires_grad=True)
        self.axis = tuple(range(1, len(input_size)))
    
    def forward(self, x):
        if not self.training:
            return x
        params = self.reverse_grad(self.params, self.p,
                                   self.grad_norm, self.axis)
        x = x + params
        # reset the param after adversarial attacking
        # Note this will not influence the storaged grad
        self.params.data = torch.zeros_like(self.params.data)
        return x


@NETWORK_REGISTRY.register()
def adver_noise(cfg):
    model_cfg = cfg.MODEL
    input_size = (cfg.DATALOADER.TRAIN_X.BATCH_SIZE, cfg.MODEL.IN_CHANNELS, *(cfg.MODEL.PATCH_SIZE))
    adv = AdverNoise(input_size=input_size,
                     p=model_cfg.ADVER_NOISE.P,
                     grad_norm=model_cfg.ADVER_NOISE.GRAD_NORM)
    return adv
