## Segmentation Generalization Framework for Medical Image

**One PyTorch based Domain Generalization (DG) framework for Medical Image Segmentation**, including the data preprocessing, data augmentation, varies network implementations, different training methods and standard performance evaluation.

**Why this framework?** Well, unlike the natural image tasks, the medical image segmentation performance for domain generalization might be very different when using different data preprocessing, data augmentation or different training methods. Many previous works lacks fair comparasion and results can hardly be repeated by community. We want to achieve one fair evaluation for varies domain generalization methods.

### Supported Datasets
Multidomain medical image segmentation tasks are limited. We collect the high quality datasets under multi-center setting in previous research as much as possible. We reorganize the dataset into the standard format for preprocessing and training. The supported dataset are listed as follow:

+ Multi-domain Prostate MRI Segmentation Dataset
+ Multi-domain Fundus Optic Cup and Optic Disc Segmentation Dataset
+ Multi-domain Fundus Vessel Segmentation Dataset
+ Multi-domain Covid-19 Segmentation Dataset
+ Multi-domain Cardiac Segmentation Dataset

### Data Preprocessing

Standard data preprocessing is the basis for any fair comparation. In this work, we adpot the widely used data preprocessing strategy which supports varies data format, including but not limited to 2D: PNG, JPG... 3D: NIFTI, DICOM... We transfer all the raw data to numpy array according to different settings like 2D training for 2D data, 2D training for 3D data, 3D training for 3D data. More detailed tutorials are available in []

### Data Augmentation
Different data augemnetation methods can influence the trained model's generalization ability dramatically, as shown in \BigAug. We standardalize the data augmentation based on the batch-generator. For every baseline, we basically adopt the data augmentation of the default setting of nnUNet. More detailed tutorials are available in []

### Network Implementation and Network Training
We support varies network structure and, actually, you can merge any segmentation enginee as you like. In this work, many network implementations are based on MONAI Network. One thing you need to notice is that for some domain generalization methods, you need to return the imtermidiate feature for training. Make sure your network structure supports this. More detailed tutorials are available in []

Varies optimizers including, SGD, Adam, AdamW, AMSGrad, RMSProp are supported.

Different learning rate schedulers including Cosine, Multi-Step, Single-Step are supported.

Segmentation Losses including Dice, DiceCE, DiceFocal are also supported.

### Standard Evaluation
The standard evaluation methods are essential for domain generalization ability measurement. We provide automatic evaluation using varies metrics including  "Dice", "Jaccard", "Precision", "Recall", "Accuracy", "Hausdorff Distance 95", "Avg. Symmetric Surface Distance". Note that for fair comparasion evaluation is based on case-level, for example, each 3D case rather than the slice of 3D is considered as one sample.