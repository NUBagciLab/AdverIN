""" Here is to implement the input augmentation
Including:

Random Conv: https://arxiv.org/abs/2007.13003

Note BigAug is implemented in data augmentation for dataset
"""

import torch
import torch.nn as nn
import numpy as np


class RandConv(nn.Module):
    """Random Convolution Based input augmentation
    https://arxiv.org/abs/2007.13003

    args:
        input_channel:int
        output_channel:int
        prob: float, the possibility for random conv, default 0.5
        n_dim: convolution dimension
        kernel_size: kernel size for convolution
        distribution: the distribution for random initialization
                      supports: kaiming_normal, kaiming_uniform, xavier_normal
    """
    def __init__(self, input_channel:int, output_channel:int, prob:float=0.5,
                       n_dim:int=2, kernel_size:int=3, distribution='kaiming_normal'):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.n_dim = n_dim
        self.prob = prob
        self.kernel_size = kernel_size
        self.distribution = distribution

        if n_dim == 2:
            self.conv = nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                  kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        elif n_dim == 3:
            self.conv = nn.Conv3d(in_channels=input_channel, out_channels=output_channel,
                                  kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        else:
            raise NameError(f"Random Conv {n_dim} Not implement now")
        self.register_buffer("conv", self.conv)
        self.random_func = self.get_random()

    @torch.no_grad()
    def forward(self, input):
        if not self.training or np.random.random() > self.prob:
            return input

        self.reset_conv()
        return self.conv(input)

    def get_random(self):
        if self.distribution == 'kaiming_uniform':
            random_func = nn.init.kaiming_uniform_
        elif self.distribution == 'kaiming_normal':
            random_func = nn.init.kaiming_normal_
        elif self.distribution == 'xavier_normal':
            random_func = nn.init.xavier_normal_
        else:
            raise NotImplementedError("Initialization method not support")
        return random_func

    def reset_conv(self):
        self.random_func(self.conv.weight)

