"""
Multidomain preprocessor for 2D, 3D data
"""

import os
import json
import pickle
import yaml
import functools as func
import argparse

from multiprocessing.pool import Pool

from MedSegDGSSL.preprocess.utils import (image_preprocessor_2d, 
                              image_preprocessor_3d, image_preprocessor_3d_slice, 
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
        file_dict = {}
        outdir_dict = {}
        dataset_meta_dict = {}
    
        for domain in self.domains:
            mkdir_if_missing(os.path.join(self.processed_dir, domain))
            with open(os.path.join(self.raw_dir, domain, "dataset.json"), 'r') as f:
                temp_dataset_json = json.load(f)
            
            temp_file = temp_dataset_json.pop("training")
            num_classes = len(temp_dataset_json['labels'])
            if "label" in temp_file[0].keys():
                temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, domain, item["image"])),
                                    "label": os.path.abspath(os.path.join(self.raw_dir, domain, item["label"])),
                                    "num_classes": num_classes} for item in temp_file]
            else:
                temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, domain, item["image"])),
                                    "num_classes": num_classes} for item in temp_file]
            
            temp_outdir_list = [os.path.abspath(os.path.join(self.processed_dir, domain, item["image"].split('/')[-1].split('.')[0])) for item in temp_file]

            file_dict[domain] = temp_file_list
            outdir_dict[domain] = temp_outdir_list

            _ =temp_dataset_json.pop("test")
            dataset_meta_dict[domain] = temp_dataset_json
        return file_dict, outdir_dict, dataset_meta_dict
    

    def __call__(self):
        file_dict, outdir_dict, dataset_meta_dict = self.generate_mapfiles()

        if self.is_threeD_data:
            if self.is_threeD_training:
                map_func = func.partial(image_preprocessor_3d_space, target_space=self.target_space,
                                                               clip_percent=(0.5, 99.5))
            else:
                assert "num_slice" in self.data_preprocess_config.keys(), "You should define the slice number"
                num_slice = self.data_preprocess_config["num_slice"]
                map_func = func.partial(image_preprocessor_3d_slice, target_size=self.target_size,
                                                               clip_percent=(0.5, 99.5), num_slice=num_slice)
        else:
            map_func = func.partial(image_preprocessor_2d, target_size=self.target_size,
                                                               clip_percent=(0.5, 99.5))
        
        for domain in self.domains:
            with Pool(processes=self.NUM_THREAD) as pool:
                meta_list = pool.starmap(map_func, zip(file_dict[domain], outdir_dict[domain]))
            
            meta_dict, pos_match_dict = {}, {}
            for item in meta_list:
                meta_dict.update({item['case_name']:item})
                pos_match_dict.update(item["pos_match"])
            out_dict = {'case_info':meta_dict,
                        'dataset_info': dataset_meta_dict[domain],
                        'positive_match':pos_match_dict}
            with open(os.path.join(self.processed_dir, domain, "meta.pickle"), 'wb') as f:
                pickle.dump(out_dict, f)

        print("Finished preprocessing")

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='', help='path to dataset config')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()
    propressor = Preprocessor(args.config_dir)
    propressor()
