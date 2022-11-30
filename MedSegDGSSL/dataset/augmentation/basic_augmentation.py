"""
Here to define the basic augmentation method for medical image segmentation

The augmentation will be based on 2D and using the MONAI augmentation format
"""
import monai
import monai.transforms as transforms

basic_image_key = ("image", "label")
NUM_SAMPLES_2D = 8

def get_basic_2d_augmentation(num_classes, prob:float=0.3):
    basic_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropSamplesd(keys=basic_image_key, num_samples=NUM_SAMPLES_2D,
                                           roi_size=(1, -1, -1), random_size=False),
        transforms.SqueezeDimd(keys=basic_image_key, dim=0),
        transforms.RandFlipd(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=basic_image_key, prob=prob, spatial_axes=(0, 1)),
    ])
    return basic_compose

def get_unlabel_2d_augmentation(num_classes, prob:float=0.3):
    basic_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropSamplesd(keys=basic_image_key, num_samples=NUM_SAMPLES_2D,
                                           roi_size=(1, -1, -1), random_size=False),
        transforms.SqueezeDimd(keys=basic_image_key, dim=0),
        transforms.RandFlipd(keys=("image"), prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=("image"), prob=prob, spatial_axes=(0, 1))
    ])
    return basic_compose

def get_dense_2d_augmentation(num_classes, prob:float=0.3):
    dense_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropSamplesd(keys=basic_image_key, num_samples=NUM_SAMPLES_2D,
                                           roi_size=(1, -1, -1), random_size=False),
        transforms.SqueezeDimd(keys=basic_image_key, dim=0),
        transforms.RandFlipd(keys=basic_image_key, prob=prob, spatial_axis=(0, 1)),
        transforms.RandRotate90d(keys=basic_image_key, prob=prob, spatial_axes=(0, 1)),
    ])
    return dense_compose


def get_evaluation_2d_augmentation(num_classes):
    eval_compose = transforms.Compose([
        transforms.AddChanneld(keys=basic_image_key),
        transforms.RandSpatialCropSamplesd(keys=basic_image_key, num_samples=NUM_SAMPLES_2D,
                                           roi_size=(1, -1, -1), random_size=False),
        transforms.SqueezeDimd(keys=basic_image_key, dim=0),
    ])
    return eval_compose

