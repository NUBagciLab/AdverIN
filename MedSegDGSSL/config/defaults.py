from yacs.config import CfgNode as CN
from .build import CONFIG_REGISTRY

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

# Directory to save the output files
_C.OUTPUT_DIR = './output'
# Path to a directory where the files were saved
_C.RESUME = ''
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = 1
_C.USE_CUDA = True
# Print detailed information (e.g. what trainer,
# dataset, backbone, etc.)
_C.VERBOSE = True
_C.DATA_IS_3D = True
_C.TRAINING_IS_2D = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (384, 384)
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Default mean and std come from ImageNet
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Padding for random crop
_C.INPUT.CROP_PADDING = 4
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
_C.DATASET.KEYS = ('data', 'seg')
# List of names of source domains
_C.DATASET.SOURCE_DOMAINS = ()
# List of names of target domains
_C.DATASET.TARGET_DOMAINS = ()
# Specify the fold for cross-fold training
_C.DATASET.FOLD = 0
# Number of labeled instances for the SSL setting
_C.DATASET.NUM_LABELED = 250
# Percentage of validation data (only used for SSL datasets)
# Set to 0 if do not want to use val data
# Using val data for hyperparameter tuning was done in Oliver et al. 2018
_C.DATASET.VAL_PERCENT = 0.1
# Fold index for STL-10 dataset (normal range is 0 - 9)
# Negative number means None
_C.DATASET.STL10_FOLD = -1
# CIFAR-10/100-C's corruption type and intensity level
_C.DATASET.CIFAR_C_TYPE = ''
_C.DATASET.CIFAR_C_LEVEL = 1
# Use all data in the unlabeled data set (e.g. FixMatch)
_C.DATASET.ALL_AS_UNLABELED = False
_C.DATASET.AUGMENT = 'baseline_augmentation'

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# Setting for train_x data-loader
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32
# Parameter for RandomDomainSampler
# 0 or -1 means sampling from all domains
_C.DATALOADER.TRAIN_X.N_DOMAIN = 0
# Setting for train_u data-loader
_C.DATALOADER.TRAIN_U = CN()
# Set to false if you want to have unique
# data loader params for train_u
_C.DATALOADER.TRAIN_U.SAME_AS_X = True
_C.DATALOADER.TRAIN_U.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_U.BATCH_SIZE = 32
_C.DATALOADER.TRAIN_U.N_DOMAIN = 0

# Setting for test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = 'RandomSampler'
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Setting for the data augmentation
###########################
_C.AUGMENTATION = CN()
_C.AUGMENTATION.PROB = 0.3

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights for initialization
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.NAME = 'basicunet'
_C.MODEL.SPATIAL_DIMS = 2
_C.MODEL.PATCH_SIZE = (384, 384)
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 2
_C.MODEL.FEATURES = [32, 32, 64, 96, 128, 256]
_C.MODEL.STRIDES = [2, 2, 2, 2, 2]
_C.MODEL.NORM = "BATCH"
_C.MODEL.DROPOUT = 0.3
_C.MODEL.RETURN_FEATURES = False
_C.MODEL.PRETRAINED = False

###########################
# Setting for the loss
###########################
_C.LOSS = "DiceCELoss"


###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'adam'
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.99
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = 'single_step'
_C.OPTIM.STEPSIZE = (10, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to disable
_C.TRAIN.CHECKPOINT_FREQ = 5
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = 'train_x'

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = 'Segmentation'
_C.TEST.FINAL_EVALUATOR = 'FinalSegmentation'
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
_C.TEST.FINAL_MODEL = 'best_val'
# How often (epoch) to do testing during training
# Set to 0 or negative value to disable
_C.TEST.EVAL_FREQ = 1
# Use 'test' set or 'val' set for evaluation
_C.TEST.SPLIT = 'test'

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ''


@CONFIG_REGISTRY.register()
def default():
    return _C.clone()
