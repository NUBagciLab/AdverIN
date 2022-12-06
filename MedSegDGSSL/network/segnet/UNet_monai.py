"""
UNet monai version
"""
import monai
from monai.networks.nets import UNet

from .build import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
def monaiunet(model_cfg):
    unet = UNet(spatial_dims=2,
                in_channels=1,
                out_channels=2,
                channels=(32, 32, 64, 64, 128, 128, 256),
                strides=(2, 2, 2, 2, 2, 2),
                num_res_units=2,
                norm='BATCH',
                dropout=0.)
    return unet