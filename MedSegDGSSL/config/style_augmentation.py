from .defaults import _C
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Model specifics for style mixing or augmentation
# Will apply this to change the encoder of segmentation enginee
# List the potential parameters here for parameter updating
###########################
# Mixstyle
_C.MODEL.MIXSTYLE = CN()
_C.MODEL.MIXSTYLE.ALPHA = 0.5
_C.MODEL.MIXSTYLE.PROB = 0.5

# DSU
_C.MODEL.DSU = CN()
_C.MODEL.DSU.ALPHA = 0.5
_C.MODEL.DSU.PROB = 0.5

# PADAIN
_C.MODEL.PADAIN = CN()
_C.MODEL.PADAIN.PROB = 0.5

# BatchInstanceNorm
_C.MODEL.BIN = CN()

@CONFIG_REGISTRY.register()
def StyleAugDG():
    return _C.clone()
