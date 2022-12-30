
import torch
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.optim import build_optimizer, build_lr_scheduler
from MedSegDGSSL.network import build_network
from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.dataset.dataset import MetaDatasetWarpper
from MedSegDGSSL.utils import count_num_param
import pdb



@TRAINER_REGISTRY.register()
class MetaLearning(TrainerX):

    def build_model(self):
        cfg = self.cfg

        # build model1 for forward meta-train and get updated parameter to do meta-test
        self.model1 = build_network(cfg.MODEL.NAME, model_cfg=cfg.MODEL)
        self.model1.to(self.device)
        print(f"# params1: {count_num_param(self.model1):,}")
        self.optim1 = build_optimizer(self.model1, cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.register_model("model1", self.model1, self.optim1, self.sched1)

        # build model2 for foward meta-test and update original parameter
        self.model2 = build_network(cfg.MODEL.NAME, model_cfg=cfg.MODEL)
        self.model2.to(self.device)
        print(f"# params2: {count_num_param(self.model2):,}")
        self.optim2 = build_optimizer(self.model2, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("model2", self.model2, self.optim2, self.sched2)
        
        # make model1 and model2 have same parameter
        self.model1.load_state_dict(self.model2.state_dict())
        self.model = self.model2
        

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
        
        outer_loss = torch.tensor(0., device=self.device)
        loss_summary = {}
        keys = list(batch.keys())

        for idx, task in enumerate(batch):

            # get meta-test data
            if task == "test":
                test_input, test_label, test_domain = batch[keys[idx]].values()
                test_input = test_input.to(self.device)
                test_label = test_label.to(self.device)
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

            # backward model1 and trasfer gradient from model1 to model2
            self.model_zero_grad("model1")
            self.model_backward(outer_loss)
            for pp, qq in zip(self.model1.parameters(), self.model2.parameters()):
                if pp.grad is not None:
                    if qq.grad is not None:
                        qq.grad += pp.grad / (len(self.cfg.DATASET.SOURCE_DOMAINS) - 1)    
                    else:
                        qq.grad = pp.grad / (len(self.cfg.DATASET.SOURCE_DOMAINS) - 1)    
        

        # update model2 parameter and make parameter of model1 and model2 consistent
        self.model_update("model2")
        self.model1.load_state_dict(self.model2.state_dict())
        
        # return loss
        loss_summary["loss_metatest"] = outer_loss

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



