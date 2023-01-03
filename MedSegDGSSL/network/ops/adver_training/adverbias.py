"""
Implement the adversarial bias attacking for increasing model's robustness

Based on "Realistic Adversarial Data Augmentation for MR Image Segmentation"

https://arxiv.org/abs/2006.13322
Github repo
https://github.com/cherise215/AdvBias
"""

import torch
import torch.nn as nn

from MedSegDGSSL.network.build import NETWORK_REGISTRY

def get_gassuian_smooth(spatial_dim:int,
                        kernel_size:int=3,
                        sigma:float=1):
    x_corr = torch.arange(kernel_size)
    grid = torch.stack(torch.meshgrid(*([x_corr]*spatial_dim),
                                      indexing='ij'))
    mean = kernel_size // 2
    gaussian_kernel = torch.exp(-torch.sum(((grid-mean)/sigma)**2, dim=0))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.unsqueeze_(0).unsqueeze_(0)
    if spatial_dim==2:
        conv = nn.Conv2d(in_channels=1, out_channels=1,
                         kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
    elif spatial_dim==3:
        conv = nn.Conv3d(in_channels=1, out_channels=1,
                         kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
    else:
        raise NotImplementedError('not supported')

    conv.weight.data = gaussian_kernel
    conv.weight.requires_grad = False
    return conv


class AdverBias(nn.Module):
    """ Adversarial Bias Attacking
    args:
        input_size: the input image size, [NCHW..]
        control_spacing: the control spacing for each control point
        magnitude: the magnitude for controling the bias
    """
    def __init__(self, input_size, control_spacing,
                       magnitude:float=0.3, smooth_kernel_size:int=3):
        super().__init__()
        self.input_size = input_size
        self.control_spacing = control_spacing
        self.spacial_dims = len(control_spacing)
        self.smooth_kernel_size = smooth_kernel_size
        assert len(input_size) == (len(control_spacing)+2), 'Match the control spacing with the input'
        control_size = [input_size[0], 1]
        for i in range(self.spacial_dims):
            assert self.input_size[i+2] % self.control_spacing[i] ==0, "Watch you control spacing"
            control_size.append(self.input_size[i+2]//self.control_spacing[i])
        self.magnitude = magnitude
        interp_mode = 'trilinear' if self.spacial_dims==3 else 'bilinear'
        self.smooth = get_gassuian_smooth(spatial_dim=self.spacial_dims,
                                          kernel_size=smooth_kernel_size,
                                          sigma=1.)
        self.upsample = nn.Upsample(scale_factor=control_spacing,
                                    mode=interp_mode, align_corners=False)
        self.params = nn.parameter.Parameter(torch.ones(size=control_size),
                                             requires_grad=True)

    def forward(self, x):
        if not self.training:
            return x
        bias = self.generate_bias(self.params)
        x = x * bias
        # reset the param after adversarial attacking
        # Note this will not influence the storaged grad
        self.params.data = torch.ones_like(self.params.data)
        return x

    def generate_bias(self, control_point):
        bias = self.upsample(self.smooth(control_point))
        bias = torch.clamp_(bias, min=(1-self.magnitude),
                            max=(1+self.magnitude))
        return bias

@NETWORK_REGISTRY.register()
def adver_bias(cfg):
    model_cfg = cfg.MODEL
    input_size = (cfg.DATALOADER.TRAIN_X.BATCH_SIZE, cfg.MODEL.IN_CHANNELS, *(cfg.MODEL.PATCH_SIZE))
    adv = AdverBias(input_size=input_size,
                    control_spacing=model_cfg.ADVER_BIAS.SPACING,
                    magnitude=model_cfg.ADVER_BIAS.MAGNITUDE)
    return adv