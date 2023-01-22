from .defaults import _C as _C_ORG
import copy
_C = copy.deepcopy(_C_ORG)
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for Mixup
###########################
# Mixstyle
_C.MODEL.MIX_UP = CN()
_C.MODEL.MIX_UP.ALPHA = 0.5
_C.MODEL.MIX_UP.PROB = 0.5
_C.MODEL.MIX_UP.ORDER = True


@CONFIG_REGISTRY.register()
def MixUpDG():
    return _C.clone()
