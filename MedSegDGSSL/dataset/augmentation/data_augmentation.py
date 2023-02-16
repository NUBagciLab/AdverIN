"""
Here to define the basic augmentation method for medical image segmentation

The augmentation will be based on 2D and using the MONAI augmentation format
"""

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform,SpatialTransform_2
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from MedSegDGSSL.dataset.augmentation.augmentation_params import params_dict
from MedSegDGSSL.dataset.augmentation.custom_transforms import RandCropByPosNegRatio, RandAdjustResolution, MinMaxNormalization, MinMaxNormalization


def get_baseline_train_augmentation(patch_size, params_key='3D'):
    params = params_dict[params_key]
    tr_transforms = []
    if params.get("do_crop_by_pn_ratio"):
        tr_transforms.append(RandCropByPosNegRatio(patch_size,
                                                   pos=params.get("pos_ratio"), neg=params.get("neg_ratio")))

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=-1,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("do_res"):
        tr_transforms.append(RandAdjustResolution(params.get("p_resolution"), params.get("gaussian_range"),
                             params.get("sharp_range")))

    tr_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    # tr_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_default_train_augmentation(patch_size, params_key='3D'):
    params = params_dict[params_key]
    tr_transforms = []
    if params.get("do_crop_by_pn_ratio"):
        tr_transforms.append(RandCropByPosNegRatio(patch_size,
                                                   pos=params.get("pos_ratio"), neg=params.get("neg_ratio")))

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=-1,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    """tr_transforms.append(GaussianNoiseTransform(p_per_sample=params.get("p_gaussian_noise")))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=params.get("contrast_range"), p_per_sample=params.get("p_contrast")))
    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))"""

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("do_res"):
        tr_transforms.append(RandAdjustResolution(params.get("p_resolution"), params.get("gaussian_range"),
                             params.get("sharp_range")))

    tr_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    # tr_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_online_eval_augmentation(patch_size, params_key='3D'):
    params = params_dict[params_key]
    val_transforms = []
    if params.get("do_crop_by_pn_ratio"):
        val_transforms.append(RandCropByPosNegRatio(patch_size,
                                                   pos=params.get("pos_ratio"), neg=params.get("neg_ratio")))
    val_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    # val_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    val_transforms = Compose(val_transforms)
    return val_transforms

def get_dense_augmentation(patch_size, params_key='3D'):
    params = params_dict[params_key]
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("do_crop_by_pn_ratio"):
        tr_transforms.append(RandCropByPosNegRatio(patch_size,
                                                   pos=params.get("pos_ratio"), neg=params.get("neg_ratio")))

    tr_transforms.append(SpatialTransform_2(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        deformation_scale=params.get("eldef_deformation_scale"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=-1,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
        p_independent_scale_per_axis=params.get("p_independent_scale_per_axis")
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.15))  # inverted gamma

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("do_res"):
        tr_transforms.append(RandAdjustResolution(params.get("p_resolution"), params.get("gaussian_range"),
                             params.get("sharp_range")))

    tr_transforms.append(MinMaxNormalization(norm_range=params.get('norm_range')))
    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms = Compose(tr_transforms)
   
    return tr_transforms
