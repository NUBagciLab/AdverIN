import torch
import torch.nn as nn
import numpy as np

from .reverse_grad import reverse_grad
from MedSegDGSSL.network.build import NETWORK_REGISTRY

class AdverStyleBlock(nn.Module):
    def __init__(self, batch_size, num_channels, prob=0.5, eps=1e-6) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.eps = eps
        self.prob = prob
        self.params = nn.parameter.Parameter(torch.zeros(2, self.batch_size, self.num_channels),
                                             requires_grad=True)
    
    def forward(self, x):
        if not self.training or np.random.random() > self.prob:
            return x

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        mu_adver = torch.reshape(self.params[0], mu.shape)
        var_adver = torch.reshape(self.params[1], var.shape)
        sig = (var + self.eps).sqrt()
        sig_adver = (var_adver + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed =(x-mu) / sig 
        x =  (sig + reverse_grad(sig_adver))*x_normed+ (mu+reverse_grad(mu_adver))
        return x

    def reset(self):
        # reset the param after adversarial attacking
        # Note this will not influence the storaged grad
        self.params.data = torch.zeros_like(self.params.data)


class AdverStyleEncoder(nn.Module):
    def __init__(self, batch_size, channels:list, prob:float=0.5, adver_block:list=None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        if adver_block is None:
            adver_block = [True]*len(self.channels)

        self.adver_block = adver_block
        
        self.adver_block = nn.ModuleList([AdverStyleBlock(batch_size, channel, prob) if adver_block[i] else nn.Identity() for i, channel in enumerate(channels)])
    
    def forward(self, x, i):
        return self.adver_block[i](x)
    
    def reset(self):
        for i, adver in enumerate(self.adver_block):
            if adver:
                self.adver_block[i].reset()


@NETWORK_REGISTRY.register()
def adver_style(cfg):
    adv = AdverStyleEncoder(batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                            channels=cfg.MODEL.FEATURES,
                            prob=cfg.MODEL.ADVER_STYLE.PROB)
    return adv
