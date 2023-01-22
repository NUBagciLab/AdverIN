from .defaults import _C as _C_ORG
import copy
_C = copy.deepcopy(_C_ORG)
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for Diffusion Model
###########################

_C.MODEL.NAME = 'diff_unet'
_C.MODEL.TIME_STEPS = 400
_C.MODEL.EMBED_CHANNEL = 32

_C.MODEL.DIFFUSION = CN()
_C.MODEL.DIFFUSION.LOSS = "l1"


@CONFIG_REGISTRY.register()
def IntraDiffusionTrainer():
    return _C.clone()

@CONFIG_REGISTRY.register()
def DiffusionTrainer():
    return _C.clone()
