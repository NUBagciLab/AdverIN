# Domain Generalization Framework for Medical Image Segmentation

## Introduction
**One PyTorch based Domain Generalization (DG) framework for Medical Image Segmentation**, including the data preprocessing, data augmentation, varies network implementations, different training methods and standard performance evaluation.

**Why this framework?** Well, unlike the natural image tasks, the medical image segmentation performance for domain generalization might be very different when using different data preprocessing, data augmentation or different training methods. Many previous works lack fair comparasion and results can hardly be repeated by community. We want to achieve one fair evaluation for varies domain generalization methods.

### Supported Datasets
Multidomain medical image segmentation tasks are limited. We collect the high quality datasets under multi-center setting in previous research as much as possible. We reorganize the dataset into the standard format for preprocessing and training. The supported dataset are listed as follow:

+ Multi-domain Prostate MRI Segmentation Dataset
+ Multi-domain Fundus Optic Cup and Optic Disc Segmentation Dataset
+ Multi-domain Fundus Vessel Segmentation Dataset
+ Multi-domain Covid-19 Segmentation Dataset
+ Multi-domain Cardiac Segmentation Dataset

### Data Preprocessing

Standard data preprocessing is the basis for any fair comparation. In this work, we adpot the widely used data preprocessing strategy which supports varies data format, including but not limited to 2D: PNG, JPG... 3D: NIFTI, DICOM... We transfer all the raw data to numpy array according to different settings like 2D network for 2D data, 2D network for 3D data, 3D network for 3D data. More detailed tutorials are available in []

### Data Augmentation
Different data augemnetation methods can influence the trained model's generalization ability dramatically, as shown in \BigAug. We standardalize the data augmentation based on the batch-generator. For every baseline, we basically adopt the data augmentation of the default setting of nnUNet. More detailed tutorials are available in []

### Network Implementation and Network Training
We support varies network structure and, actually, you can merge any segmentation enginee as you like. In this work, many network implementations are based on MONAI Network. One thing you need to notice is that for some domain generalization methods, you need to return the imtermidiate feature for training. Make sure your network structure supports this. More detailed tutorials are available in []

Varies optimizers including, SGD, Adam, AdamW, AMSGrad, RMSProp are supported.

Different learning rate schedulers including Cosine, Multi-Step, Single-Step are supported.

Segmentation Losses including Dice, DiceCE, DiceFocal are also supported.

### Standard Evaluation
The standard evaluation methods are essential for domain generalization ability measurement. We provide automatic evaluation using varies metrics including  "Dice", "Jaccard", "Precision", "Recall", "Accuracy", "Hausdorff Distance 95", "Avg. Symmetric Surface Distance". Note that for fair comparasion, evaluation is based on case-level. For example, one complete 3D case rather than the slices of 3D is considered as one sample.

## Get started

### Installization
Clone the repo, create a conda environment and install the package
```bash
# Clone the repo
git clone git@github.com:freshman97/MedSeg_Generalization_Framework.git
cd MedSeg_Generalization_Framework
# Create the environment
conda create -y -n medsegdg python=3.8
conda activate medsegdg
conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia
# Install the required packages
pip install -r requirements.txt
python setup.py install
```

### Run Training
One basic training example using the Prostate MRI dataset under domain generalization setting, more detailed tutorials for config setting are available in [].

```bash
#!/bin/bash
DATA=path_to_dir
DATASET=ProstateMRI
D1=BIDMC
D2=BMC
D3=HK
D4=I2CVB
D5=UCL
D6=RUNMC

SEED=0
method=baseline
cuda_device=0

# train with 5 domain and test on the rest
CUDA_VISIBLE_DEVICES=${cuda_device} python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer Vanilla \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} \
--target-domains ${D6} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D6}
```
### Implemented Domain Generalization Method
Here we implement several previous domain generalization methods, including

**Data Augmentation**
+ 1
+ 2
+ 3

**Style Mixing**
+ 1
+ 2
+ 3

**Feature Alignment**
+ 1
+ 2
+ 3

**Meta-Learning**
+ 1

**Self-Challenging**
+ 1

### Add New Network
We can add the new segmentation network easily (we use the unet implementation of MONAI as example) as follow:

```Python
import monai
from monai.networks.nets import UNet

from MedSegDGSSL.network.segnet.build import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
def monaiunet(model_cfg):
    unet = UNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels=model_cfg.IN_CHANNELS,
                out_channels=model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                num_res_units=2,
                norm=model_cfg.NORM,
                dropout=model_cfg.DROPOUT)
    return unet
```

### Add New Dataset
We can add the new dataset as follow, the dataset discription should includes name, domain, labels, and data type information:

```Python
from MedSegDGSSL.dataset.build import DATASET_REGISTRY
from MedSegDGSSL.dataset.data_base import Datum, DatasetBase


@DATASET_REGISTRY.register()
class ProstateMRI(DatasetBase):
    """Prostate Segmentation

    Statistics:
        - 6 domains: "BMC", "HK", "I2CVB", "UCL", "RUNMC", "BIDMC"
        - Prostate Segmentation
    """
    dataset_name = 'ProstateMRI'
    domains = ["BMC", "HK", "I2CVB", "UCL", "RUNMC", "BIDMC"]
    labels = {"0": "Background", "1": "Prostate"}
    data_shape = "3D"
    def __init__(self, cfg):

        super().__init__(data_dir=cfg.DATASET.ROOT,
                         train_domains=cfg.DATASET.SOURCE_DOMAINS,
                         test_domains=cfg.DATASET.TARGET_DOMAINS)
        self._lab2cname = self.labels
        self.num_classes = len(self.labels)
```

### Add New Trainer
We can add the new trainer easily as follow. Make sure you read the implementation of SimpleTrainer first before adding new trainer to understand where to change.

```Python
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX

@TRAINER_REGISTRY.register()
class NewTrainer(TrainerX):
    """Trainer description.
    """
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = self.loss_func(output, label)
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item()}
```
## Citation
```
Citation here
```

## Acknowledgement
We acknowledge the DASSL, MONAI, and BatchGenerator for their great contribution. 