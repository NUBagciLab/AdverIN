from torch.nn import functional as F

from engine.trainer import TRAINER_REGISTRY, TrainerX
from metrics import compute_dice


@TRAINER_REGISTRY.register()
class Vanilla(TrainerX):
    """Vanilla baseline."""

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = self.loss_func(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'dice': compute_dice(output, label).item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['image']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
