from .defaults import _C
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for Align FeaturesDG
###########################
_C.MODEL.DOMAIN_ALIGNMENT = CN()
_C.MODEL.DOMAIN_ALIGNMENT.LOSS_NAME = "MMD"
_C.MODEL.DOMAIN_ALIGNMENT.LOSS_WEIGHT = 0.05

@CONFIG_REGISTRY.register()
def AlignFeaturesDG():
    return _C.clone()
