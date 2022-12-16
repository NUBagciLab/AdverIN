from .defaults import _C
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for RandConv
###########################
# Mixstyle
_C.MODEL.RANDCONV = CN()
_C.MODEL.RANDCONV.DIST = 'kaiming_normal'
_C.MODEL.RANDCONV.KERNEL_SIZE = 3

@CONFIG_REGISTRY.register()
def RandConvDG():
    return _C.clone()
