import torch
from torch.nn import functional as F


from MedSegDGSSL.dataset.data_manager import DataManager
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.metrics import compute_dice


@TRAINER_REGISTRY.register()
class IntraTrainer(TrainerX):
    ''' To implement intra domain training process
    '''
    def build_data_loader(self):
        """Create essential data-related attributes.
        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, set_kfold=True)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.final_test_loader = dm.final_test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

        self.dm = dm

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        #print(input.shape, torch.sum(label))
        loss = self.loss_func(output, label)
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item()}
        dice_value = compute_dice(output, label)
        for i in range(self.num_classes-1):
            loss_summary[f'dice {str(i+1)}'] = dice_value[i+1].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['data']
        label = batch['seg']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def final_evaluation(self):
        """A generic final evaluation pipeline."""
        super().final_evaluation(extra_name=f'_fold_{self.cfg.DATASET.FOLD}')
