"""
Include the dataset supporting
2D Training -> 2D data image and slice of 3D data volume
3D Training -> 3D data volume
2D Evaluation-> 2D data image
3D Evaluation -> 3D data volume
"""
import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class TrainDatasetWarpper(Dataset):
    def __init__(self, data_files:list, transform, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.transform = transform
        self.keys = keys

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        temp_dict = self.data_files[index]
        out_dict = {}
        out_dict["domain"] = temp_dict["domain"]
        temp_data = np.load(temp_dict["data"])

        for key in self.keys:
            out_dict[key] = np.expand_dims(temp_data[key], axis=0)

        out_dict = self.transform(out_dict)

        for key in self.keys:
            out_dict[key] = out_dict[key][0]
        return out_dict


class EvalDatasetWarpper(Dataset):
    def __init__(self, data_files:list, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        self.folder_dir = data_files[0].rsplit('/', 1)[0]
        with open(os.path.join(self.folder_dir, 'meta.pickle')) as f:
            self.meta_data = pickle.load(f)
        self.case_names = list(self.meta_data.keys())
        self.keys = keys

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        temp_dict = self.data_files[index]
        case_name = self.data_files[index].rsplit('/', 1)[-1].rsplit('.', 1)[0]
        out_dict = {}
        temp_data = np.load(temp_dict["data"])

        for key in self.keys:
            out_dict[key] = temp_data[key]

        out_dict["meta"] = self.meta_data[case_name]
        return out_dict


class Eval3DDatasetWarpperFrom2D(Dataset):
    """Well this is not the style I like
    """
    def __init__(self, data_files:list, transform, keys=("data", "seg")):
        super().__init__()
        self.data_files = data_files
        # Just folder dir is enough this time
        self.folder_dir = data_files[0].rsplit('/', 1)[0]
        with open(os.path.join(self.folder_dir, 'meta.pickle')) as f:
            self.meta_data = pickle.load(f)
        self.case_names = list(self.meta_data.keys())
        self.transform = transform
        self.keys = keys

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        case_name = self.case_names[index]
        out_dict = {}
        for key in self.keys:
            out_dict[key] = []
        for i in self.meta_data[case_name]["depth"]:
            temp_data = np.load(os.path.join(self.folder_dir,
                                          case_name+"_slice{:03d}.npz".format(i)))
            for key in self.keys:
                out_dict[key].append(temp_data[key])
        for key in self.keys:
            out_dict[key] = np.vstack(out_dict[key])

        out_dict["meta"] = self.meta_data["meta"]
        return out_dict
