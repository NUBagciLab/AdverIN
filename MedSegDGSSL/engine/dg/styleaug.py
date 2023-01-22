import torch
from torch.nn import functional as F

from MedSegDGSSL.engine import TRAINER_REGISTRY, TrainerX

from MedSegDGSSL.metrics import compute_dice


@TRAINER_REGISTRY.register()
class StyleAugDG(TrainerX):
    """StyleAugDG baseline."""

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
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
