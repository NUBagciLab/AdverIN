"""
To implement the custom transform

"""

import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop

rng = np.random.default_rng()

def get_lbs_for_foreground_crop(seg, crop_size, data_shape, margins):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    max_label = np.max(seg)
    if max_label == 1:
        select_class = 1
    else:
        select_class = np.random.randint(1, max_label)

    select_pos = rng.choice(np.argwhere(seg==select_class))
    lbs = []
    for i in range(len(data_shape) - 2):
        if data_shape[i+2] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(select_pos[i])
        else:
            lbs.append((data_shape[i+2] - crop_size[i]) // 2)
    return lbs

def crop_foreground(data, seg=None, crop_size=128, margins=(0, 0, 0),
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes
    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        lbs = get_lbs_for_foreground_crop(seg[b], crop_size, data_shape_here, margins)

        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]

        slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


class RandCropByPosNegRatio(AbstractTransform):
    """ Random Crop volume or spatial according to the positive / negative ratio
    """
    def __init__(self, crop_size=128, pos=1., neg=1., 
                       margins=(0, 0, 0), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.margins = margins
        self.crop_size = crop_size
        self.pos_neg_ratio = pos / (pos + neg)
        pass

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        
        if np.random.rand(1) < self.pos_neg_ratio and seg is not None:
            data, seg = crop_foreground(data, seg, self.crop_size, self.margins)
        else:
            data, seg = random_crop(data, seg, self.crop_size, self.margins)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


if __name__ == "__main__":
    seg = np.zeros((1, 1, 32, 256, 256))
    seg[:,:, 12:18, :128, 128:] = 1
    data_dict = {"data": np.random.randn(1, 1, 32, 256, 256),
                 "seg": seg}
    crop_trans = RandCropByPosNegRatio(crop_size=(16, 128, 128), pos=1., neg=1.)
    out = crop_trans(**data_dict)
    for key in out:
        print(key, out[key].shape)

