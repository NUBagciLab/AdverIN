"""
Include the dataset supporting
2D Training -> 2D data image and slice of 3D data volume
3D Training -> 3D data volume
2D Evaluation-> 2D data image
3D Evaluation -> 3D data volume
"""
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class TrainDatasetWarpper(Dataset):
    def __init__(self, data_files:list, transform, pos_ratio:float=0.33, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.transform = transform

        # Pos_ratio is to balance the positive samples
        # and negative samples ratio during training
        self.pos_ratio = pos_ratio
        self.keys = keys
        self.dtype_dict ={'data': torch.float, 'seg': torch.long}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        temp_dict = self.data_files[index]
        out_dict = {}
        
        if np.random.random() < self.pos_ratio:
            temp_data = np.load(temp_dict["positive"])
        else:
            temp_data = np.load(temp_dict["data"])

        for key in self.keys:
            out_dict[key] = np.expand_dims(temp_data[key], axis=0)
        out_dict = self.transform(**out_dict)
        for key in self.keys:
            out_dict[key] = torch.from_numpy(out_dict[key][0]).to(self.dtype_dict[key])
        out_dict["domain"] = torch.tensor([temp_dict["domain"]])
        return out_dict


class EvalDatasetWarpper(Dataset):
    def __init__(self, data_files:list, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.domains_map = {file['data']:file['data'].rsplit('/', 2)[-2] for file in data_files}
        self.domains = list(set(self.domains_map.values()))
        self.folder_dir = data_files[0]["data"].rsplit('/', 2)[0]
        self.meta_data_maps = {}
        self.case_names = []
        for domain in self.domains:
            with open(os.path.join(self.folder_dir, domain, 'meta.pickle'), 'rb') as f:
                meta_data = pickle.load(f)
            self.meta_data_maps[domain] = meta_data

        self.keys = keys
        self.dtype_dict ={'data': torch.float, 'seg': torch.long}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        temp_dict = self.data_files[index]
        temp_split = temp_dict['data'].rsplit('/', 2)
        domain = temp_split[-2]
        case_name = temp_split[-1].split('.', 1)[0]
        out_dict = {}
        temp_data = np.load(temp_dict["data"])

        for key in self.keys:
            out_dict[key] = torch.from_numpy(temp_data[key]).to(self.dtype_dict[key])

        out_dict["meta"] = self.meta_data_maps[domain]["case_info"][case_name]
        out_dict["meta"]["domain"] = domain
        return out_dict


class Eval3DDatasetWarpperFrom2D(Dataset):
    """Well this is not the style I like,looks dirty
    """
    def __init__(self, data_files:list, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.domains_map = {file['data'].rsplit('_', 1)[0]:file['data'].rsplit('/', 2)[-2] for file in data_files}
        self.domains = list(set(self.domains_map.values()))
        self.folder_dir = data_files[0]["data"].rsplit('/', 2)[0]

        self.meta_data_maps = {}
        for domain in self.domains:
            with open(os.path.join(self.folder_dir, domain, 'meta.pickle'), 'rb') as f:
                meta_data = pickle.load(f)
            self.meta_data_maps[domain] = meta_data

        self.case_names = list(self.domains_map.keys())
        self.keys = keys
        self.dtype_dict ={'data': torch.float, 'seg': torch.long}

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        temp_dir = self.case_names[index]
        temp_split = temp_dir.rsplit('/', 2)
        domain, case_name = temp_split[-2], temp_split[-1]

        out_dict = {}
        for key in self.keys:
            out_dict[key] = []
        for i in range(self.meta_data_maps[domain]["case_info"][case_name]["depth"]):
            temp_data = np.load(os.path.join(self.folder_dir, domain,
                                             case_name+"_slice{:03d}.npz".format(i)))
            for key in self.keys:
                out_dict[key].append(temp_data[key])
        for key in self.keys:
            out_dict[key] = torch.from_numpy(np.stack(out_dict[key])).to(self.dtype_dict[key])

        out_dict["meta"] = self.meta_data_maps[domain]["case_info"][case_name]
        out_dict["meta"]["domain"] = domain
        return out_dict


class MetaDatasetWarpper(Dataset):
    def __init__(self, data_files:list, transform, pos_ratio:float=0.33, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.transform = transform
        # Pos_ratio is to balance the positive samples
        # and negative samples ratio during training
        self.pos_ratio = pos_ratio
        self.task = keys
        self.keys = ("data", "seg")
        self.dtype_dict ={'data': torch.float, 'seg': torch.long}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        out_dict = {}
        for idx, i in enumerate(index):

            temp_dict = self.data_files[i]
            inner_dict = {}
            
            if np.random.random() < self.pos_ratio:
                temp_data = np.load(temp_dict["positive"])
            else:
                temp_data = np.load(temp_dict["data"])

            for key in self.keys:
                inner_dict[key] = np.expand_dims(temp_data[key], axis=0)
            inner_dict = self.transform(**inner_dict)

            for key in self.keys:
                inner_dict[key] = torch.from_numpy(inner_dict[key][0]).to(self.dtype_dict[key])
            inner_dict["domain"] = torch.tensor([temp_dict["domain"]])

            out_dict[self.task[idx]] = inner_dict
        return out_dict

data_key = ("data", "seg")

def collect_eval_fn(data):
    out_dict = {}
    for key in data[0].keys():
        if key in data_key:
            out_dict[key] = torch.stack([item[key] for item in data], dim=0)
        else:
            out_dict[key] = [item[key] for item in data]
    return out_dict

if __name__ == '__main__':
    from MedSegDGSSL.dataset.augmentation.data_augmentation import get_default_train_augmentation
    transform = get_default_train_augmentation((256, 256), '2D')
    ds = TrainDatasetWarpper(data_files=[{'data': '/home/zze3980/project/AdverHistAug/data/ProstateMRI/processed/2DSlice/BMC/Case00_slice000.npz',
                                          'domain': 0}], 
                                         transform=transform)
    print(ds[0])