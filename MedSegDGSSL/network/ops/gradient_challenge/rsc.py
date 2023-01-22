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

from torch.autograd import Variable

class RSC(object):
    def __init__(self, challenge_list, percentile=0.95) -> None:
        super().__init__()
        self.challenge_list = challenge_list
        self.percentile = percentile

    def wrap_variable(self, feature_list):
        feature_list = [Variable(feature) if is_challenge else feature for feature, is_challenge in zip(feature_list, self.challenge_list)]
        return feature_list
    
    def generate_mask(self, feature_list):
        feature_masks = []
        for idx, feature in enumerate(feature_list):
            if not self.challenge_list[idx]:
                x_grad = feature.grad
                x_size = x_grad.shape
                x_mask = torch.flatten(x_grad, dims=1)
                x_index = int(x_mask.size(1)*self.percentile)
                x_percentile = torch.sort(x_mask, dim=1)[:, x_index].unsqueeze(1)
                x_mask = (x_percentile-x_mask)>0
                x_mask = torch.reshape(x_mask, x_size)
            else:
                x_mask = torch.ones_like(feature)
            feature_masks.append(x_mask)
        
        return feature_masks
    
    def mask_forward(self, feature_list, feature_masks):
        feature_list = [feature*mask for feature, mask in zip(feature_list, feature_masks)]
        return feature_list
