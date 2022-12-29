from lib2to3.pgen2.literals import test
from xml.sax.handler import DTDHandler
import torch
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
import torch.nn as nn
from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.dataset.dataset import MetaDatasetWarpper
from collections import OrderedDict
import pdb



@TRAINER_REGISTRY.register()
class MetaLearning(TrainerX):

    def build_data_loader(self):
        """Create essential data-related attributes.
        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, dataset_wrapper=MetaDatasetWarpper)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.final_test_loader = dm.final_test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

        self.dm = dm

    def forward_backward(self, batch):
        
        outer_loss = torch.tensor(0., device=self.device)
        loss_summary = {}
        keys = list(batch.keys())

        for idx, task in enumerate(batch):
            if idx == 0:
                test_input, test_label, test_domain = batch[keys[idx]].values()
                test_input = test_input.to(self.device)
                test_label = test_label.to(self.device)

            train_input, train_label, train_domain = batch[keys[idx]].values()
            train_input = train_input.to(self.device)
            train_label = train_label.to(self.device)

            train_output = self.model(train_input)
            inner_loss = self.loss_func(train_output, train_label)
            loss_summary["loss_metatrain{}".format(idx)] = inner_loss

            self.model_zero_grad('model')
            updated_params = self.gradient_update(inner_loss, self.model)

            test_output = self.model(test_input, params=updated_params)
            outer_loss += self.loss_func(test_output, test_label)
        
        outer_loss.div(idx)
        self.model_backward(outer_loss)
        self.model_update('model')

        # return loss
        loss_summary["loss_metatest"] = outer_loss

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def gradient_update(self, loss, model):
        grads = torch.autograd.grad(loss, model.parameters())
        # grads = nn.utils.clip_grad_norm_(grads, max_norm=self.cfg.CLIPGRADIENTNORM)
        updated_params = OrderedDict()
        pdb.set_trace()
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            updated_params[name] = param - self.cfg.OPTIM.LR * grad

        return updated_params



