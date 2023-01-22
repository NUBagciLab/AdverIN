import copy
import torch
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.optim import build_optimizer, build_lr_scheduler
from MedSegDGSSL.network import build_network
from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.dataset.dataset import MetaDatasetWarpper
from MedSegDGSSL.utils import count_num_param
import pdb

from MedSegDGSSL.metrics import compute_dice

@TRAINER_REGISTRY.register()
class MetaLearning(TrainerX):

    def build_model(self):
        cfg = self.cfg

        # build model1 for forward meta-train and get updated parameter to do meta-test
        self.model1 = build_network(cfg.MODEL.NAME, model_cfg=cfg.MODEL)
        self.model1.to(self.device)
        print(f"# params1: {count_num_param(self.model1):,}")
        self.optim1 = build_optimizer(self.model1, cfg.OPTIM.UPDATE_OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM.UPDATE_OPTIM)
        self.register_model("model1", self.model1, self.optim1, self.sched1)

        # build model for foward meta-test and update original parameter
        self.model = build_network(cfg.MODEL.NAME, model_cfg=cfg.MODEL)
        self.model.to(self.device)
        print(f"# params2: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM.META_OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM.META_OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        # make model1 and model have same parameter
        self.model1.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def build_data_loader(self):
        """Create essential data-related attributes.
        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, train_dataset_wrapper=MetaDatasetWarpper)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.final_test_loader = dm.final_test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

        self.dm = dm

    def forward_backward(self, batch):
        
        loss_summary = {}
        keys = list(batch.keys())

        self.model_zero_grad("model")

        for idx, task in enumerate(batch):

            # get meta-test data
            if task == "test":
                test_input, test_label, test_domain = batch[keys[idx]].values()
                test_input = test_input.to(self.device)
                test_label = test_label.to(self.device)
                
                # just for zero setting the gradient to zeros
                loss = torch.sum(0*self.model(test_input))
                loss.backward()
                self.model.zero_grad()
                continue

            # get meta-train data
            train_input, train_label, train_domain = batch[keys[idx]].values()
            train_input = train_input.to(self.device)
            train_label = train_label.to(self.device)

            # forward meta-train to model1 and get inner loss regarding meta-train
            train_output = self.model1(train_input)
            inner_loss = self.loss_func(train_output, train_label)
            loss_summary["loss_metatrain{}".format(idx)] = inner_loss

            # backward and update model1 to get updated parameter to do metatest
            self.model_backward_and_update(inner_loss, names='model1')

            # do metatest and get outer loss
            test_output = self.model1(test_input)
            outer_loss = self.loss_func(test_output, test_label)
            dice_value = compute_dice(test_output, test_label)
            for i in range(self.num_classes-1):
                loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()
            
            # copy the BN set up
            """            
            for key in self.model1.state_dict().keys():
                if 'ADN.N' in key:
                    self.model.state_dict()[key] = self.model1.state_dict()[key].detach().clone()"""
                    
            # backward model1 and trasfer gradient from model1 to model
            self.model_zero_grad("model1")
            self.model_backward(outer_loss)

            for pp, qq in zip(self.model1.parameters(), self.model.parameters()):
                if pp.grad is not None:
                    if qq.grad is not None:
                        qq.grad += copy.deepcopy(pp.grad) / (len(self.cfg.DATASET.SOURCE_DOMAINS) - 1)
 

            self.model1.load_state_dict(copy.deepcopy(self.model.state_dict()))


        # update model parameter and make parameter of model1 and model consistent
        self.model_update("model")
        self.model1.load_state_dict(self.model.state_dict())

        # return loss
        loss_summary["loss_metatest"] = outer_loss

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



