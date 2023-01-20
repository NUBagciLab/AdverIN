from .defaults import _C as _C_ORG
import copy
_C = copy.deepcopy(_C_ORG)
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for RSC
###########################
_C.MODEL.RSC = CN()
_C.MODEL.RSC.CHALLENGE_LIST = [1, 1, 1, 1, 1, 1]
_C.MODEL.RSC.PERCENTILE = 0.95

@CONFIG_REGISTRY.register()
def RSCDG():
    return _C.clone()
