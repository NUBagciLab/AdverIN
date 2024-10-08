"""
Implement the adversarial histogram attacking for increasing model's robustness
"""

import torch
import torch.nn as nn
import numpy as np

from .reverse_grad import ReverseNormGrad
from MedSegDGSSL.network.build import NETWORK_REGISTRY

def interp1d(y, xi, nbins):
    """ To calculate the simple inner 1d interpolation
    input data range should be scaled within 0-1, just for making things easy
    """
    index = xi*(nbins-1)
    # print('min', torch.min(index), 'max', torch.max(index), 'nbins', nbins)
    index_round = torch.floor(index).to(torch.long)
    index_round_pluse_one = torch.clamp_max(index_round + 1, nbins-1).to(torch.long)
    index_left, index_right = index - index_round, index_round_pluse_one-index
    y, index_round, index_round_pluse_one = torch.flatten(y, start_dim=0, end_dim=1), \
                                            torch.flatten(index_round, start_dim=0, end_dim=1), \
                                            torch.flatten(index_round_pluse_one, start_dim=0, end_dim=1),
    select_left = torch.reshape(torch.gather(y, dim=1, index=torch.flatten(index_round, start_dim=1)),
                                shape=index.shape)
    select_right = torch.reshape(torch.gather(y, dim=1, index=torch.flatten(index_round_pluse_one, start_dim=1)),
                                shape=index.shape)

    yi = index_right*select_left + index_left*select_right

    return yi


class AdverHist(nn.Module):
    """Adversarial histogram attacking
    Input:
        prob: the possibility for applying the adversial training
        data_min, data_max: the min, max value to control the data range
        num_control_point: how many control points for the histogram changing
    """
    def __init__(self, batch_size, num_channels, prob:float=0.5, data_min:float=-1.0, data_max:float=1.0,
                 num_control_point:int=11, grad_norm:float=1., p:int=2, ):
        super().__init__()
        self.prob = prob
        self.data_min = data_min
        self.data_max = data_max
        self.num_control_point = num_control_point
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.p = p
        self.grad_norm = grad_norm*(num_control_point-1)
        self.reverse_grad = ReverseNormGrad()
        self.params = nn.parameter.Parameter(torch.zeros(self.batch_size, self.num_channels, self.num_control_point),
                                             requires_grad=True)
        self.scale = nn.parameter.Parameter(torch.zeros(self.batch_size, self.num_channels, 2),
                                             requires_grad=True)
        self.axis = tuple([2])

    def forward(self, x):
        if not self.training:
            return x

        # print(torch.softmax(self.params, dim=0))
        map_point =  torch.cumsum(torch.softmax(self.params, dim=-1), dim=-1)
        map_point_min, _ = torch.min(map_point, dim=-1, keepdim=True)
        map_point_max, _ = torch.max(map_point, dim=-1, keepdim=True)
        map_point = (map_point-map_point_min) / (map_point_max-map_point_min)
        map_point = self.reverse_grad(map_point, p=self.p,
                                      grad_norm=self.grad_norm,
                                      axis=self.axis)
        ### 1st order interpolation
        x_shape = x.shape
        x_flat = torch.flatten(x, start_dim=2)
        x_min, _ = torch.min(x_flat, dim=2)
        x_max, _ = torch.max(x_flat, dim=2)
        x_min, x_max = torch.reshape(x_min, (*x_shape[:2], *((1,)*(len(x_shape)-2)))),\
                        torch.reshape(x_max, (*x_shape[:2], *((1,)*(len(x_shape)-2))))
        x = interp1d(map_point, (x-x_min)/(x_max-x_min), self.num_control_point)
        x_scale = torch.sigmoid(self.scale)
        x_min, x_max = x_min + (x_max - x_min)/2*torch.reshape(x_scale[:, :, 0], (*x_shape[:2], *((1,)*(len(x_shape)-2)))), \
                       x_max - (x_max - x_min)/2*torch.reshape(x_scale[:, :, 1], (*x_shape[:2], *((1,)*(len(x_shape)-2))))
        # x_min, x_max = x_min + (x_max-x_min)/4*np.random.random(), \
        #                    x_max - (x_max-x_min)/4*np.random.random()
        
        x = x_min + (x_max - x_min)*x

        return x

    def get_entropy(self):
        dist = torch.softmax(self.params, dim=0)
        entropy = -torch.sum(dist*torch.log(dist))
        return entropy

    def reset(self):
        # reset the param after adversarial attacking
        # Note this will not influence the storaged grad or influence the optimizer updating
        self.params.data = torch.zeros_like(self.params.data)
    
    def reset_random(self):
        self.params.data = torch.randn_like(self.params.data)


@NETWORK_REGISTRY.register()
def adver_hist(cfg):
    model_cfg = cfg.MODEL
    adv = AdverHist(batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    num_channels=cfg.MODEL.IN_CHANNELS,
                    prob=model_cfg.ADVER_HIST.PROB,
                    data_min=model_cfg.ADVER_HIST.DATA_MIN,
                    data_max=model_cfg.ADVER_HIST.DATA_MAX,
                    num_control_point=model_cfg.ADVER_HIST.CONTROL_POINT,
                    grad_norm=model_cfg.ADVER_HIST.GRAD_NORM)
    return adv
