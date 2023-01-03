from .defaults import _C
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Specify the parameters for adeversarial training
###########################
# AdverHist
_C.MODEL.ADVER_MODEL_NAME = ''
_C.MODEL.ADVER_RATE = 0.001
_C.MODEL.ADVER_HIST = CN()
# Hyper-parameter to limit the entropy
_C.MODEL.ADVER_HIST.DATA_MIN = -1.
_C.MODEL.ADVER_HIST.DATA_MAX = 1.
_C.MODEL.ADVER_HIST.CONTROL_POINT = 21  
_C.MODEL.ADVER_HIST.ALPHA = 0.01
_C.MODEL.ADVER_HIST.PROB = 0.5
_C.MODEL.ADVER_HIST.GRAD_NORM = 1.


# AdverNoise
_C.MODEL.ADVER_NOISE = CN()
# Hyper-parameter to limit the entropy
_C.MODEL.ADVER_NOISE.P = 2
_C.MODEL.ADVER_NOISE.GRAD_NORM = 1.

# AdverBias
_C.MODEL.ADVER_BIAS = CN()
# Hyper-parameter to limit the entropy
_C.MODEL.ADVER_BIAS.SPACING = (64, 64)
_C.MODEL.ADVER_BIAS.MAGNITUDE = 0.3

@CONFIG_REGISTRY.register()
def AdverTraining():
    return _C.clone()
