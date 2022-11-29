"""
Here to define the basic augmentation method for medical image segmentation

The augmentation will be based on 2D and using the MONAI augmentation format
"""
import monai
import monai.transforms as transforms

basic_image_key = ("image", "label")

def get_basic_2d_augmentation(num_classes, prob:float=0.3):
    basic_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropd(keys=basic_image_key,
                                    roi_size=(-1, -1, 1)),
        transforms.SqueezeDimd(keys=basic_image_key, dim=3),
        transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=0.5, upper=99.5,
                                                   b_min=-1., b_max=1., clip=True),
        transforms.RandFlipd(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.AsDiscreted(keys="label", to_onehot=num_classes),
    ])
    return basic_compose

def get_unlabel_2d_augmentation(num_classes, prob:float=0.3):
    basic_compose = transforms.Compose([
        transforms.AddChanneld(keys=("image")),
        transforms.RandSpatialCropd(keys=("image"),
                                    roi_size=(-1, -1, 1)),
        transforms.SqueezeDimd(keys=("image"), dim=3),
        transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=0.5, upper=99.5,
                                                   b_min=-1., b_max=1., clip=True),
        transforms.RandFlipd(keys=("image"), prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=("image"), prob=prob, spatial_axis=(0, 1))
    ])
    return basic_compose

def get_dense_2d_augmentation(num_classes, prob:float=0.3):
    dense_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropd(keys=basic_image_key,
                                    roi_size=(-1, -1, 1)),
        transforms.SqueezeDimd(keys=basic_image_key, dim=3),
        transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=0.5, upper=99.5,
                                                   b_min=-1., b_max=1., clip=True),
        transforms.RandFlipd(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.AsDiscreted(keys="label", to_onehot=num_classes),
    ])
    return dense_compose


def get_evaluation_2d_augmentation(num_classes):
    eval_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropd(keys=basic_image_key,
                                    roi_size=(-1, -1, 1)),
        transforms.SqueezeDimd(keys=basic_image_key, dim=3),
        transforms.AsDiscreted(keys="label", to_onehot=num_classes),
    ])
    return eval_compose

