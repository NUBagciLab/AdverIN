from .defaults import _C as _C_ORG
import copy
_C = copy.deepcopy(_C_ORG)
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
_C.MODEL.RANDCONV.KERNEL_SIZE_LIST = [1, 3, 5, 7]

@CONFIG_REGISTRY.register()
def RandConvDG():
    return _C.clone()
