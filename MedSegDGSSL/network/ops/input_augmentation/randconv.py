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
                       n_dim:int=2, kernel_size_list:list=[1, 3, 5, 7], distribution='kaiming_normal'):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.n_dim = n_dim
        self.prob = prob
        self.kernel_size_list = kernel_size_list
        self.distribution = distribution
        
        # self.register_buffer("rand_conv", self.conv)
        self.reset_conv()
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
        kernel_size = np.random.choice(self.kernel_size_list)
        if self.n_dim == 2:
            self.conv = nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel,
                                  kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        elif self.n_dim == 3:
            self.conv = nn.Conv3d(in_channels=self.input_channel, out_channels=self.output_channel,
                                  kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        else:
            raise NameError(f"Random Conv {self.n_dim} Not implement now")
        self.random_func(self.conv.weight)
