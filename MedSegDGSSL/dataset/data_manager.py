import torch
import numpy as np
from torch.utils.data import Dataset

from monai.data import DataLoader
from MedSegDGSSL.dataset.build import build_dataset
from MedSegDGSSL.dataset.samplers import build_sampler
from MedSegDGSSL.dataset.augmentation.basic_augmentation import *


class DatasetWarpper(Dataset):
    def __init__(self, data_files:list, transform, keys=("image", "label")):
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
            out_dict[key] = temp_data[key]
        out_dict = self.transform(out_dict)
        return out_dict

def build_data_loader(
    cfg,
    sampler_type='SequentialSampler',
    data_source=None,
    batch_size=64,
    n_domain=0,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain
    )

    # Build data loader
    data_loader = DataLoader(
        dataset_wrapper(data_source, transform=tfm),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        custom_tfm_unlabel=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = get_basic_2d_augmentation(cfg.DATASET.NUM_CLASSES, prob=cfg.AUGMENTATION.PROB)
        else:
            print('* Using custom transform for training')
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = get_evaluation_2d_augmentation(cfg.DATASET.NUM_CLASSES)
        else:
            print('* Using custom transform for testing')
            tfm_test = custom_tfm_test
        
        if custom_tfm_unlabel is None:
            tfm_unlabel = get_unlabel_2d_augmentation(cfg.DATASET.NUM_CLASSES, prob=cfg.AUGMENTATION.PROB)
        else:
            print('* Using custom transform for unlabel data')
            tfm_unlabel = custom_tfm_unlabel
        
        if dataset_wrapper is None:
            dataset_wrapper = DatasetWarpper
        else:
            print('* Using custom dataset wrapper')

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                tfm=tfm_unlabel,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    def show_dataset_summary(self, cfg):
        print('***** Dataset statistics *****')

        print('  Dataset: {}'.format(cfg.DATASET.NAME))

        if cfg.DATASET.SOURCE_DOMAINS:
            print('  Source domains: {}'.format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print('  Target domains: {}'.format(cfg.DATASET.TARGET_DOMAINS))

        print('  # classes: {}'.format(self.num_classes))

        print('  # train_x: {:,}'.format(len(self.dataset.train_x)))

        if self.dataset.train_u:
            print('  # train_u: {:,}'.format(len(self.dataset.train_u)))

        if self.dataset.val:
            print('  # val: {:,}'.format(len(self.dataset.val)))

        print('  # test: {:,}'.format(len(self.dataset.test)))
