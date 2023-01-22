import torch
from torch.nn import functional as F
import pdb
from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX
from MedSegDGSSL.metrics import compute_dice
#### Need to modify the distance here
from MedSegDGSSL.network.ops import MaximumMeanDiscrepancy, CrossEntropyDistance


@TRAINER_REGISTRY.register()
class AlignFeaturesDG(TrainerX):
    """AlignFeatures baseline."""
    def get_loss_func(self):
        if self.cfg.MODEL.DOMAIN_ALIGNMENT.LOSS_NAME == "MMD":
            self.feature_align_loss = MaximumMeanDiscrepancy()
        elif self.cfg.MODEL.DOMAIN_ALIGNMENT.LOSS_NAME == "CrossEntropy":
            self.feature_align_loss = CrossEntropyDistance()
        else:
            raise NotImplementedError(f'{self.cfg.MODEL.DOMAIN_ALIGNMENT.LOSS_NAME} not supported')

        self.alignment_weight = self.cfg.MODEL.DOMAIN_ALIGNMENT.LOSS_WEIGHT
        return super().get_loss_func()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output, features = self.model(input)
        #print(input.shape, torch.sum(label))
        
        loss = self.loss_func(output, label)
        features = torch.mean(torch.flatten(features, start_dim=2), dim=-1)
        #############
        # think about this is one better one
        #############
        n = features.size(0)
        features_domain1, features_domain2 = features[:n//2], features[n//2:]
        align_loss = self.feature_align_loss(features_domain1, features_domain2)
        self.model_backward_and_update(loss + self.alignment_weight*align_loss)
        loss_summary = {
            'loss': loss.item(),
            'align loss': align_loss.item()}

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
