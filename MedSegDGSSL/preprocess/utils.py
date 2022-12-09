"""
Basic Utils function for the proprocessing
Including: 

file reader for 3d medical image and 2d medical image
resize for 3d or 2d image to target size
normalization for 3d or 2d image to [-1, 1] with clipping

"""

import os
import numpy as np
from skimage import io as skio
from skimage import transform
import SimpleITK as sitk


def image_preprocessor_2d(file_dict, out_dir, target_size, clip_percent=(0.5, 99.5)):
    ### This one is to support the png, jpg ... 2d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    out_dict = {}
    image = np.transpose(skio.imread(file_dict["image"]), axes=(2, 0, 1)).astype(np.float32)
    target_size_image = (image.shape[0], *target_size)

    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape[1:]
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size
    meta_dict["spacing"] = (1, 1)
    meta_dict["meta"] = {}

    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0]), np.percentile(image_resize, q=clip_percent[1])
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    out_dict["data"] = image_resize.astype(np.float32)

    num_classes = file_dict['num_classes']
    meta_dict["meta"]['num_classes'] = num_classes
    if "label" in file_dict.keys():
        # For PNG you need to transfer them to corresponding label
        seg = skio.imread(file_dict["label"], as_gray=True).astype(np.float32) / 255
        seg = np.round(seg * (num_classes - 1))
        seg_resize = transform.resize(seg, target_size, order=0)
        out_dict["seg"] = np.expand_dims(seg_resize.astype(np.int64), 0)
    
    meta_dict["pos_match"] = {}
    meta_dict["pos_match"][case_name+".npz"] = case_name+".npz"
    np.savez(out_dir+".npz", **out_dict)
    return meta_dict

def image_preprocessor_3d_slice(file_dict, out_dir, target_size, clip_percent=(0.5, 99.5), num_slice:int=1):
    ### This one is to support the nii, dicom ... 3d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    # note that the resize transform will be limited with xy plane

    # This one is designed for 2d network training with 3d data
    out_dict = {}
    image_org= sitk.ReadImage(file_dict["image"])
    image = sitk.GetArrayFromImage(image_org)
    target_size_image = (image.shape[0], *target_size)
    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size_image
    meta_dict["spacing"] = image_org.GetSpacing()[::-1]
    meta_dict["meta"] = {}
    meta_dict["pos_match"] = {}
    for key in image_org.GetMetaDataKeys():
        meta_dict["meta"][key] = image_org.GetMetaData(key)
    # For the z dimension, the axis size should not be changed
    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0], axis=(1, 2), keepdims=True), \
                            np.percentile(image_resize, q=clip_percent[1], axis=(1, 2), keepdims=True)
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    out_dict["data"] = image_resize.astype(np.float32)
    if "label" in file_dict.keys():
        seg = sitk.GetArrayFromImage(sitk.ReadImage(file_dict["label"]))
        # This is just for some disgusting image issue
        seg[seg<0] = 0
        seg_resize = transform.resize(seg, target_size_image, order=0)
        out_dict["seg"] = seg_resize.astype(np.int64)
        positive_slice = list(np.nonzero(np.sum(seg_resize, axis=(1, 2)) > 20.)[0])
        len_pos = len(positive_slice)

    depth = image_resize.shape[0]
    for i in range(depth):
        slice_list = [max(min(i-num_slice//2+idx, depth-1), 0) for idx in range(num_slice)]
        out_dict_slice = {"data": image_resize[slice_list]}
        if "label" in file_dict.keys():
            out_dict_slice.update({"seg": np.expand_dims(seg_resize[i], 0)})
            if i not in positive_slice:
                meta_dict["pos_match"][case_name + "_slice{:03d}.npz".format(i)] = \
                    case_name + "_slice{:03d}.npz".format(positive_slice[i%len_pos])
            else:
                meta_dict["pos_match"][case_name + "_slice{:03d}.npz".format(i)] = \
                    case_name + "_slice{:03d}.npz".format(i)
        else:
            meta_dict["pos_match"][case_name + "_slice{:03d}.npz".format(i)] = \
                    case_name + "_slice{:03d}.npz".format(i)
        np.savez(out_dir+"_slice{:03d}.npz".format(i), **out_dict_slice)

    meta_dict["depth"] = depth
    return meta_dict

def image_preprocessor_3d(file_dict, out_dir, target_size, clip_percent=(0.5, 99.5)):
    ### This one is to support the nii, dicom ... 3d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    # This is defined for 3d network training

    out_dict = {}
    image_org = sitk.ReadImage(file_dict["image"])
    image = sitk.GetArrayFromImage(image_org)
    org_shape = image_org.GetSize()

    target_size_image = tuple([org_shape[-1], *target_size[:-1]])
    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size_image
    meta_dict["spacing"] = image_org.GetSpacing()[::-1]
    meta_dict["meta"] = {}
    for key in image_org.GetMetaDataKeys():
        meta_dict["meta"][key] = image_org.GetMetaData(key)
    # For the z dimension, the axis size should not be changed
    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0]), np.percentile(image_resize, q=clip_percent[1])
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    out_dict["data"] = np.expand_dims(image_resize.astype(np.float32), 0)
    if "label" in file_dict.keys():
        seg = sitk.GetArrayFromImage(sitk.ReadImage(file_dict["label"]))
        seg_resize = transform.resize(seg, target_size_image, order=0)
        out_dict["seg"] = np.expand_dims(seg_resize.astype(np.int64), 0)

    meta_dict["pos_match"] = {}
    meta_dict["pos_match"][case_name+".npz"] = case_name+".npz"
    np.savez(out_dir+".npz", **out_dict)
    return meta_dict

def image_preprocessor_3d_space(file_dict, out_dir, target_space, clip_percent=(0.5, 99.5)):
    ### This one is to support the nii, dicom ... 3d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    # This is defined for 3d network training

    out_dict = {}
    image_org = sitk.ReadImage(file_dict["image"])
    image = sitk.GetArrayFromImage(image_org)
    org_shape = image_org.GetSize()
    org_spacing = image_org.GetSpacing()

    target_size_image = tuple([int(org_shape[i]*org_spacing[i]/target_space[i]) for i in reversed(range(3))])
    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size_image
    meta_dict["meta"] = {}
    for key in image_org.GetMetaDataKeys():
        meta_dict["meta"][key] = image_org.GetMetaData(key)
    # For the z dimension, the axis size should not be changed
    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0]), np.percentile(image_resize, q=clip_percent[1])
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    out_dict["data"] = np.expand_dims(image_resize.astype(np.float32), 0)
    if "label" in file_dict.keys():
        seg = sitk.GetArrayFromImage(sitk.ReadImage(file_dict["label"]))
        seg_resize = transform.resize(seg, target_size_image, order=0)
        out_dict["seg"] = np.expand_dims(seg_resize.astype(np.int64), 0)

    meta_dict["pos_match"] = {}
    meta_dict["pos_match"][case_name+".npz"] = case_name+".npz"
    np.savez(out_dir+".npz", **out_dict)
    return meta_dict

if __name__ =='__main__':
    file_dict = {'image':'/home/zze3980/project/AdverHistAug/Data/ProstateMRI/raw_data/Domain1/imagesTr/Case00_0000.nii.gz',
                 'label':'/home/zze3980/project/AdverHistAug/Data/ProstateMRI/raw_data/Domain1/labelsTr/Case00.nii.gz'}
    out_dir = '/home/zze3980/project/AdverHistAug/Data/ProstateMRI/processed/Domain1/Case00.npz'
    _ = image_preprocessor_3d_space(file_dict, out_dir, target_space=(1.0, 1., 1.))
