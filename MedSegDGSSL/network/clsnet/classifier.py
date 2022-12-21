"""
To build the classifier for the domain classification network

Here we adopt the classification network implementation from MONAI
"""

import torch
import torch.nn as nn
import monai
from monai.networks import nets
from monai.networks.nets import Classifier, ResNet

from MedSegDGSSL.network.build import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
def classify_net(model_cfg, num_classes):
    in_shape = (model_cfg.IN_CHANNELS, *model_cfg.PATCH_SIZE)
    classifer = Classifier(in_shape=in_shape,
                           classes=num_classes,
                           channels=model_cfg.FEATURES,
                           strides=model_cfg.STRIDES,
                           kernel_size=3, 
                           num_res_units=1,
                           norm=model_cfg.NORM)
    return classifer

########################
# Large size network is not prefered
########################
@NETWORK_REGISTRY.register()
def resnet18(model_cfg, num_classes):
    resnet = nets.resnet18(spatial_dims=model_cfg.SPATIAL_DIMS,
                           n_input_channels=model_cfg.IN_CHANNELS,
                           num_classes=num_classes)
    return resnet