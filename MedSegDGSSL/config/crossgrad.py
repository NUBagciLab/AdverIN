from .defaults import _C
from .build import CONFIG_REGISTRY
from yacs.config import CfgNode as CN

###########################
# Add Config definition
###########################

###########################
# Build the network including the segmentation and domain discriminator
###########################
_C.MODEL = CN()
_C.MODEL.SEG_MODEL = CN()
_C.MODEL.DIS_MODEL = CN()
# Path to model weights for initialization
_C.MODEL.SEG_MODEL.INIT_WEIGHTS = ''
_C.MODEL.SEG_MODEL.NAME = 'basicunet'
_C.MODEL.SEG_MODEL.SPATIAL_DIMS = 2
_C.MODEL.SEG_MODEL.PATCH_SIZE = (384, 384)
_C.MODEL.SEG_MODEL.IN_CHANNELS = 1
_C.MODEL.SEG_MODEL.OUT_CHANNELS = 2
_C.MODEL.SEG_MODEL.FEATURES = [32, 32, 64, 96, 128, 256]
_C.MODEL.SEG_MODEL.STRIDES = [2, 2, 2, 2, 2]
_C.MODEL.SEG_MODEL.NORM = "BATCH"
_C.MODEL.SEG_MODEL.DROPOUT = 0.3
_C.MODEL.SEG_MODEL.RETURN_FEATURES = False
_C.MODEL.SEG_MODEL.PRETRAINED = False

_C.MODEL.DIS_MODEL.INIT_WEIGHTS = ''
_C.MODEL.DIS_MODEL.NAME = 'resnet18'
_C.MODEL.DIS_MODEL.SPATIAL_DIMS = 2
_C.MODEL.DIS_MODEL.PATCH_SIZE = (384, 384)
_C.MODEL.DIS_MODEL.IN_CHANNELS = 1
_C.MODEL.DIS_MODEL.FEATURES = [32, 32, 64, 96, 128, 256]
_C.MODEL.DIS_MODEL.STRIDES = [2, 2, 2, 2, 2]
_C.MODEL.DIS_MODEL.NORM = "BATCH"
_C.MODEL.DIS_MODEL.DROPOUT = 0.3
_C.MODEL.DIS_MODEL.RETURN_FEATURES = False
_C.MODEL.DIS_MODEL.PRETRAINED = False

@CONFIG_REGISTRY.register()
def CrossGradDG():
    return _C.clone()
