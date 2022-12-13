"""
UNet monai version
"""
import monai
from monai.networks.nets import UNet

from MedSegDGSSL.network.segnet.build import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
def monaiunet(model_cfg):
    unet = UNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels=model_cfg.IN_CHANNELS,
                out_channels=model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                num_res_units=2,
                norm=model_cfg.NORM,
                dropout=model_cfg.DROPOUT)
    return unet