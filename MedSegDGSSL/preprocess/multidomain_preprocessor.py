"""
Multidomain preprocessor for 2D, 3D data
"""

import os
import json
import yaml
import functools as func

from multiprocessing.pool import Pool

from MedSegDGSSL.preprocess.utils import (image_preprocessor_2d, 
                              image_preprocessor_3d,
                              image_preprocessor_3d_space)
from MedSegDGSSL.utils.tools import mkdir_if_missing

TWO_DIM_DTYPE = ["png", "jpg"]
THREE_DIM_DTYPE = ["dicom", "nii", "nii.gz"]

class Preprocessor(object):
    """Dataset preprocessor using the dataset configuration file
    
    """
    NUM_THREAD = 16
    def __init__(self, dataset_config_file:str):
        self.data_config_file = dataset_config_file
        
        with open(dataset_config_file, 'r') as f:
            self.data_preprocess_config = yaml.safe_load(f)

        self.dataset_dir = self.data_preprocess_config["data_dir"]
        self.out_tag = self.data_preprocess_config["out_tag"]
        self.raw_dir = os.path.join(self.dataset_dir, "raw_data")
        self.domains = sorted([domain for domain in os.listdir(self.raw_dir) \
                                if not os.path.isfile(os.path.join(self.raw_dir, domain))])
        self.processed_dir = os.path.join(self.dataset_dir, "processed", self.out_tag)
        mkdir_if_missing(self.processed_dir)
        self.file_type = self.data_preprocess_config["file_type"]
        if self.file_type in TWO_DIM_DTYPE:
            self.is_threeD_data = False
        elif self.file_type in THREE_DIM_DTYPE:
            self.is_threeD_data = True
        else:
            raise NotImplementedError(f"file type {self.file_type} is not supported")

        self.is_threeD_training = False
        if "threeD_training" in self.data_preprocess_config.keys():
            self.is_threeD_training = True
            assert "target_space" in self.data_preprocess_config.keys(), ValueError("When use threeD training, you should specify the space size")
            self.target_space = tuple(self.data_preprocess_config["target_space"])
        else:
            self.target_size = tuple(self.data_preprocess_config["target_size"])

    def generate_mapfiles(self):
        file_list = []
        outdir_list = []
    
        for domain in self.domains:
            mkdir_if_missing(os.path.join(self.processed_dir, domain))
            with open(os.path.join(self.raw_dir, domain, "dataset.json"), 'r') as f:
                temp_dataset_json = json.load(f)
            
            temp_file = temp_dataset_json["training"]
            if "label" in temp_file[0].keys():
                temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, domain, item["image"])),
                                    "label": os.path.abspath(os.path.join(self.raw_dir, domain, item["label"]))} for item in temp_file]
            else:
                temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, domain, item["image"]))} for item in temp_file]
            
            temp_outdir_list = [os.path.abspath(os.path.join(self.processed_dir, domain, item["image"].split('/')[-1].split('.')[0] + ".npz")) for item in temp_file]

            file_list.extend(temp_file_list)
            outdir_list.extend(temp_outdir_list)
        
        return file_list, outdir_list
    

    def __call__(self):
        file_list, outdir_list = self.generate_mapfiles()
        if self.is_threeD_data:
            if self.is_threeD_training:
                map_func = func.partial(image_preprocessor_3d_space, target_space=self.target_space,
                                                               clip_percent=(0.5, 99.5))
            else:
                map_func = func.partial(image_preprocessor_3d, target_size=self.target_size,
                                                               clip_percent=(0.5, 99.5))
        else:
            map_func = func.partial(image_preprocessor_2d, target_size=self.target_size,
                                                               clip_percent=(0.5, 99.5))
        
        with Pool(processes=self.NUM_THREAD) as pool:
            pool.starmap(map_func, zip(file_list, outdir_list))


if __name__ == '__main__':
    data_dir = '/home/zze3980/project/AdverHistAug/configs/preprcossors/ProstateMRI.yaml'
    propressor = Preprocessor(data_dir)
    propressor()
